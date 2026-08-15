"""Guardrail metrics: non-inferiority checks on the things a win must not break.

A guardrail is not a second success metric. It answers a different question --
"did this break anything?" -- and asking it with an ordinary significance test
is a well-known way to ship harm.

Why guardrails are non-inferiority tests
----------------------------------------
The naive check is "is the refund-rate difference significant?" and if not,
ship. That treats *failure to detect* harm as *evidence of no harm*, which is
exactly backwards: an underpowered guardrail always passes. Guardrails are
therefore framed as non-inferiority tests against an explicit tolerance margin:

    H0: the metric degraded by at least ``margin``   (guilty until proven innocent)
    H1: any degradation is smaller than ``margin``

Passing requires positive evidence that harm is below tolerance. When the data
cannot support either conclusion the verdict is INCONCLUSIVE, which blocks the
launch just as a failure does -- but tells the team to gather more data rather
than to abandon the feature.

Why there is no multiplicity correction
---------------------------------------
Bonferroni across guardrails would be actively harmful. Correction makes each
test *less* likely to fire, and for a safety check the expensive error is the
missed regression, not the false alarm. The asymmetry is deliberate: alpha is
spent generously on detecting harm and stingily on declaring wins.

The margin is a product decision, not a statistical one
-------------------------------------------------------
``margin`` encodes how much degradation the business will accept in exchange
for the primary win. It belongs in the experiment intake document, agreed
before the experiment starts. Setting it afterwards, once the guardrail
readings are visible, converts the check into a rationalization.

But a margin the experiment cannot resolve is useless
-----------------------------------------------------
A margin can be perfectly reasonable as a business tolerance and still be
unanswerable at the sample size available, in which case the guardrail returns
INCONCLUSIVE and blocks the launch no matter what the data says.
:func:`minimum_resolvable_margin` computes the smallest margin a given sample
can actually clear, so this is a design-time check at intake rather than a
discovery at readout. Every result carries the figure, and
:attr:`GuardrailResult.adequately_powered` says whether the declared margin was
ever answerable.

**Choose the denominator before choosing the margin.** A guardrail measured on
a conditional subpopulation -- refunds among *converters* rather than among all
assigned users -- inherits that subpopulation's much smaller sample. In this
repo's reference experiment that single choice costs a factor of ~11.6 in
precision and is the difference between a resolvable guardrail and a permanent
INCONCLUSIVE.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats

from streamly.experiment.frequentist import TestResult, two_proportion_z_test, welch_t_test


class Verdict(str, Enum):
    """Guardrail outcomes. Only PASS permits a launch."""

    PASS = "PASS"                    # harm confidently below the tolerance margin
    FAIL = "FAIL"                    # harm confidently above the margin
    INCONCLUSIVE = "INCONCLUSIVE"    # cannot distinguish; treated as blocking


class Direction(str, Enum):
    """Which way is bad for this metric."""

    LOWER_IS_BETTER = "lower_is_better"      # refund rate, latency, crash rate
    HIGHER_IS_BETTER = "higher_is_better"    # retention, engagement


@dataclass(frozen=True)
class GuardrailResult:
    """One guardrail's reading, with the evidence behind the verdict."""

    name: str
    verdict: Verdict
    control_value: float
    treatment_value: float
    harm: float                      # signed so positive ALWAYS means degradation
    harm_ci_low: float
    harm_ci_high: float
    margin: float
    non_inferiority_p: float         # p-value for H0: harm >= margin
    alpha: float
    test: TestResult
    resolvable_margin: float         # smallest margin this sample could clear

    @property
    def blocks_launch(self) -> bool:
        return self.verdict is not Verdict.PASS

    @property
    def adequately_powered(self) -> bool:
        """Whether the declared margin was answerable at this sample size.

        False means the guardrail could never have passed, so an INCONCLUSIVE
        verdict says nothing about the treatment -- it is a design defect.
        """
        return self.margin >= self.resolvable_margin

    def __str__(self) -> str:
        return (
            f"[{self.verdict.value:12s}] {self.name}: "
            f"{self.control_value:.4f} -> {self.treatment_value:.4f} "
            f"(harm {self.harm:+.4f}, {1 - self.alpha:.0%} CI "
            f"[{self.harm_ci_low:+.4f}, {self.harm_ci_high:+.4f}], "
            f"margin {self.margin:.4f}, p_ni={self.non_inferiority_p:.3g})"
        )


@dataclass(frozen=True)
class GuardrailReport:
    """All guardrails for one experiment, and the launch gate they imply."""

    results: tuple[GuardrailResult, ...]

    @property
    def passed(self) -> bool:
        return all(r.verdict is Verdict.PASS for r in self.results)

    @property
    def blocking(self) -> tuple[GuardrailResult, ...]:
        return tuple(r for r in self.results if r.blocks_launch)

    @property
    def underpowered(self) -> tuple[GuardrailResult, ...]:
        """Guardrails whose margin was never resolvable at this sample size.

        These are design defects, not findings: they block the launch while
        telling you nothing about the treatment.
        """
        return tuple(r for r in self.results if not r.adequately_powered)

    def __str__(self) -> str:
        head = "ALL GUARDRAILS PASS" if self.passed else (
            f"{len(self.blocking)} GUARDRAIL(S) BLOCKING"
        )
        return head + "\n" + "\n".join(f"  {r}" for r in self.results)


def minimum_resolvable_margin(
    base_rate: float,
    n_control: int,
    n_treatment: int,
    alpha: float = 0.05,
    power: float = 0.80,
    assumed_harm: float = 0.0,
) -> float:
    """Smallest margin this sample size can actually clear.

    A guardrail PASSes when the upper confidence bound falls below the margin.
    If the true harm is ``assumed_harm``, that happens with probability
    ``power`` only when::

        margin > assumed_harm + (z_{1-alpha/2} + z_{power}) · SE

    The two-sided critical value is used because that is what
    :func:`_classify` compares against -- a one-sided figure here would promise
    a sensitivity the verdict rule does not deliver.

    Declaring a margin below this number guarantees an INCONCLUSIVE guardrail
    and therefore a blocked launch, regardless of what the treatment does. Run
    it at intake, not at readout.
    """
    if not 0.0 < base_rate < 1.0:
        raise ValueError(f"base_rate must be in (0, 1), got {base_rate}")
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("both arms need at least one unit")

    var = base_rate * (1.0 - base_rate)
    se = np.sqrt(var / n_control + var / n_treatment)
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    return float(assumed_harm + (crit + z_power) * se)


def required_n_for_margin(
    base_rate: float,
    margin: float,
    alpha: float = 0.05,
    power: float = 0.80,
    assumed_harm: float = 0.0,
) -> int:
    """Per-arm sample size needed to resolve ``margin`` -- the inverse of above.

    Answers the intake question "we will not accept more than X degradation;
    how many users does checking that actually take?"
    """
    if margin <= assumed_harm:
        raise ValueError("margin must exceed the assumed harm to be resolvable")
    if not 0.0 < base_rate < 1.0:
        raise ValueError(f"base_rate must be in (0, 1), got {base_rate}")

    var = base_rate * (1.0 - base_rate)
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    return int(np.ceil(2.0 * var * ((crit + z_power) / (margin - assumed_harm)) ** 2))


def _classify(
    harm: float, ci_low: float, ci_high: float, margin: float
) -> Verdict:
    """Verdict from the interval's position relative to the tolerance margin.

    The whole interval below the margin is positive evidence of acceptability;
    the whole interval above it is positive evidence of unacceptable harm;
    an interval straddling the margin supports neither.
    """
    if ci_high < margin:
        return Verdict.PASS
    if ci_low > margin:
        return Verdict.FAIL
    return Verdict.INCONCLUSIVE


def _harm_orientation(direction: Direction) -> float:
    """+1 if a rising metric is harm, -1 if a falling metric is harm."""
    return 1.0 if direction is Direction.LOWER_IS_BETTER else -1.0


def guardrail_proportion(
    name: str,
    events_control: int,
    n_control: int,
    events_treatment: int,
    n_treatment: int,
    margin: float,
    direction: Direction = Direction.LOWER_IS_BETTER,
    alpha: float = 0.05,
) -> GuardrailResult:
    """Non-inferiority check on a rate (refunds, crashes, opt-outs).

    ``margin`` is in absolute rate points and must be positive: 0.01 means "a
    one-percentage-point degradation is tolerable".
    """
    if margin <= 0:
        raise ValueError("margin must be positive; it is a tolerance, not a target")

    test = two_proportion_z_test(
        events_control, n_control, events_treatment, n_treatment, alpha=alpha
    )
    sign = _harm_orientation(direction)
    harm = sign * test.absolute_effect
    # Flipping the sign reverses the interval's endpoints.
    lo, hi = sorted((sign * test.ci_low, sign * test.ci_high))

    # One-sided test of H0: harm >= margin.
    se = test.standard_error
    z = (harm - margin) / se if se > 0 else 0.0
    p_ni = float(stats.norm.cdf(z))

    base = (events_control + events_treatment) / (n_control + n_treatment)
    resolvable = (
        minimum_resolvable_margin(base, n_control, n_treatment, alpha)
        if 0.0 < base < 1.0 else 0.0
    )

    return GuardrailResult(
        name=name,
        verdict=_classify(harm, lo, hi, margin),
        control_value=test.control_mean,
        treatment_value=test.treatment_mean,
        harm=float(harm),
        harm_ci_low=float(lo),
        harm_ci_high=float(hi),
        margin=float(margin),
        non_inferiority_p=p_ni,
        alpha=alpha,
        test=test,
        resolvable_margin=float(resolvable),
    )


def guardrail_mean(
    name: str,
    control: np.ndarray,
    treatment: np.ndarray,
    margin: float,
    direction: Direction = Direction.LOWER_IS_BETTER,
    alpha: float = 0.05,
) -> GuardrailResult:
    """Non-inferiority check on a continuous metric (latency, session length).

    ``margin`` is in the metric's own units -- 10.0 for "10ms slower is fine".
    """
    if margin <= 0:
        raise ValueError("margin must be positive; it is a tolerance, not a target")

    test = welch_t_test(control, treatment, alpha=alpha)
    sign = _harm_orientation(direction)
    harm = sign * test.absolute_effect
    lo, hi = sorted((sign * test.ci_low, sign * test.ci_high))

    se = test.standard_error
    t_stat = (harm - margin) / se if se > 0 else 0.0
    df = test.degrees_of_freedom or (control.size + treatment.size - 2)
    p_ni = float(stats.t.cdf(t_stat, df))

    # For a continuous metric the standard error is observed directly, so the
    # resolvable margin needs no variance assumption.
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    resolvable = (crit + float(stats.norm.ppf(0.80))) * se

    return GuardrailResult(
        name=name,
        verdict=_classify(harm, lo, hi, margin),
        control_value=test.control_mean,
        treatment_value=test.treatment_mean,
        harm=float(harm),
        harm_ci_low=float(lo),
        harm_ci_high=float(hi),
        margin=float(margin),
        non_inferiority_p=p_ni,
        alpha=alpha,
        test=test,
        resolvable_margin=float(resolvable),
    )


def evaluate_guardrails(results: list[GuardrailResult]) -> GuardrailReport:
    """Collect guardrail readings into a single launch gate.

    No multiplicity correction is applied -- see the module docstring.
    """
    return GuardrailReport(results=tuple(results))
