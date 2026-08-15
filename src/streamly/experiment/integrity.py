"""Experiment integrity: sample-ratio mismatch and pre-experiment balance.

These checks run **before** anyone is allowed to look at the primary metric.
An experiment that fails them is not a weak result, it is an invalid one, and
the correct response is to discard the readout rather than discount it.

Why SRM is the highest-value check in an experimentation platform
-----------------------------------------------------------------
A sample-ratio mismatch means users did not arrive in the arms in the ratio the
design specified. That almost never happens by chance -- it means something in
the pipeline is selectively dropping, misrouting, or double-counting users, and
whatever it is, it is correlated with the treatment. The measured effect is
then contaminated by a selection difference of unknown size and sign. There is
no statistical repair; the only fix is to find the bug and rerun.

The alpha convention, and why it is not 0.05
--------------------------------------------
SRM is screened on **every** experiment, so it is a multiple-comparison problem
across the whole program, not a single test. At alpha=0.05 one healthy
experiment in twenty gets flagged, the team learns the alarm is noise, and the
check stops being acted on. The industry convention (Microsoft's ExP, Fabijan
et al.) is alpha in the 0.0005-0.001 range, and this module defaults to 0.001.
That is a deliberate trade of sensitivity for credibility: a fired alarm should
mean "stop and investigate", every time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# Deliberately far stricter than a hypothesis test -- see module docstring.
SRM_ALPHA: float = 0.001


@dataclass(frozen=True)
class SrmResult:
    """Outcome of a sample-ratio-mismatch check."""

    observed: dict[str, int]
    expected: dict[str, float]
    chi_square: float
    degrees_of_freedom: int
    p_value: float
    alpha: float
    passed: bool
    worst_variant: str
    worst_delta_units: float          # observed - expected, for the worst arm

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "*** SRM DETECTED ***"
        return (
            f"{verdict}  chi2={self.chi_square:.2f} (df={self.degrees_of_freedom}), "
            f"p={self.p_value:.3g} vs alpha={self.alpha}; worst arm "
            f"'{self.worst_variant}' off by {self.worst_delta_units:+,.0f} users"
        )


@dataclass(frozen=True)
class BalanceResult:
    """Pre-experiment covariate balance between arms."""

    covariate: str
    control_mean: float
    treatment_mean: float
    standardized_difference: float    # (mean_t - mean_c) / pooled_sd
    p_value: float
    passed: bool

    def __str__(self) -> str:
        return (
            f"{'PASS' if self.passed else 'IMBALANCE'}  {self.covariate}: "
            f"{self.control_mean:.4f} vs {self.treatment_mean:.4f} "
            f"(std diff {self.standardized_difference:+.4f}, p={self.p_value:.3g})"
        )


def srm_check(
    observed: dict[str, int],
    expected_split: dict[str, float] | None = None,
    alpha: float = SRM_ALPHA,
) -> SrmResult:
    """Chi-square goodness-of-fit test of arm sizes against the intended split.

    Parameters
    ----------
    observed:
        Unit counts per variant, e.g. ``{"control": 29_967, "treatment": 30_033}``.
    expected_split:
        Intended traffic shares, e.g. ``{"control": 0.5, "treatment": 0.5}``.
        Defaults to an equal split across the observed variants. Shares are
        renormalized, so passing raw weights (50/50, or 9/1) also works.
    alpha:
        Significance threshold. Defaults to 0.001 -- see the module docstring.

    Notes
    -----
    No continuity correction is applied: at experiment scale the chi-square
    approximation is excellent, and the correction is conservative in exactly
    the direction that would cause a real mismatch to be missed.
    """
    if len(observed) < 2:
        raise ValueError("SRM needs at least two variants")
    if any(v < 0 for v in observed.values()):
        raise ValueError("variant counts must be non-negative")

    variants = sorted(observed)
    counts = np.array([observed[v] for v in variants], dtype=float)
    total = counts.sum()
    if total == 0:
        raise ValueError("no units assigned; cannot check the ratio")

    if expected_split is None:
        weights: np.ndarray = np.full(len(variants), 1.0 / len(variants))
    else:
        missing = set(variants) - set(expected_split)
        if missing:
            raise ValueError(f"expected_split is missing variants: {sorted(missing)}")
        weights = np.array([expected_split[v] for v in variants], dtype=float)
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("expected_split must be non-negative and sum to a positive value")
        weights = weights / weights.sum()

    expected_counts = weights * total
    chi2 = float(((counts - expected_counts) ** 2 / expected_counts).sum())
    dof = len(variants) - 1
    p = float(stats.chi2.sf(chi2, dof))

    deltas = counts - expected_counts
    worst = int(np.argmax(np.abs(deltas)))

    return SrmResult(
        observed={v: int(observed[v]) for v in variants},
        expected={v: float(e) for v, e in zip(variants, expected_counts)},
        chi_square=chi2,
        degrees_of_freedom=dof,
        p_value=p,
        alpha=alpha,
        passed=p >= alpha,
        worst_variant=variants[worst],
        worst_delta_units=float(deltas[worst]),
    )


def srm_minimum_detectable_loss(
    intended_arm_size: int, alpha: float = SRM_ALPHA, n_variants: int = 2
) -> float:
    """Smallest fractional loss in one arm that the SRM check would flag.

    The strict alpha buys credibility at the cost of sensitivity, and that
    trade should be a stated number rather than a surprise. If one arm silently
    drops a fraction ``f`` of its units, the chi-square statistic is
    ``m f^2 / (2 - f)``; this inverts that for the critical value, so a team can
    see what magnitude of break their sample size can actually catch before
    they rely on the check.

    Example: at 30,000 users per arm and alpha=0.001, a loss below ~2.7% will
    not be flagged -- so a healthy-looking SRM result on a small experiment is
    weak evidence, not a clean bill of health.
    """
    if intended_arm_size <= 0:
        raise ValueError("intended_arm_size must be positive")
    crit = float(stats.chi2.ppf(1.0 - alpha, n_variants - 1))
    k = crit / intended_arm_size
    # Solve f^2 + k f - 2k = 0 for the positive root.
    return float((-k + np.sqrt(k * k + 8.0 * k)) / 2.0)


def covariate_balance(
    control: np.ndarray,
    treatment: np.ndarray,
    name: str = "pre_covariate",
    alpha: float = 0.001,
) -> BalanceResult:
    """Check a pre-experiment covariate is balanced across arms.

    Randomization guarantees balance *in expectation*, not in any single draw.
    A large standardized difference on a pre-period covariate is evidence the
    randomization did not happen as designed -- it is an SRM check for
    composition rather than for count, and it catches bucketing bugs that
    preserve arm sizes.

    The standardized difference is reported alongside the p-value because at
    experiment scale a trivially small imbalance will be "significant"; the
    conventional concern threshold is |std diff| > 0.1.
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if control.size < 2 or treatment.size < 2:
        raise ValueError("each arm needs at least two observations")

    m_c, m_t = float(control.mean()), float(treatment.mean())
    v_c, v_t = float(control.var(ddof=1)), float(treatment.var(ddof=1))
    pooled_sd = np.sqrt((v_c + v_t) / 2.0)
    std_diff = (m_t - m_c) / pooled_sd if pooled_sd > 0 else 0.0
    p = float(stats.ttest_ind(treatment, control, equal_var=False).pvalue)

    return BalanceResult(
        covariate=name,
        control_mean=m_c,
        treatment_mean=m_t,
        standardized_difference=float(std_diff),
        p_value=p,
        passed=p >= alpha and abs(std_diff) < 0.1,
    )


def check_assignment_reproducibility(
    unit_ids: np.ndarray,
    recorded_variants: np.ndarray,
    salt: str,
    split: float = 0.5,
) -> tuple[bool, float, np.ndarray]:
    """Re-derive assignments from the hash and compare against what was recorded.

    Catches the failure mode SRM cannot: an assignment service that drifted from
    the documented bucketing while still producing correct arm *sizes*. Returns
    ``(all_match, mismatch_rate, mismatched_unit_ids)``.
    """
    from streamly.experiment.assign import assign_many

    expected = assign_many(np.asarray(unit_ids), salt, split)
    recorded = np.asarray(recorded_variants, dtype=object)
    if expected.shape != recorded.shape:
        raise ValueError("unit_ids and recorded_variants must align")

    mismatched = expected != recorded
    rate = float(mismatched.mean())
    return not mismatched.any(), rate, np.asarray(unit_ids)[mismatched]


def duplicate_units(unit_ids: np.ndarray) -> np.ndarray:
    """Unit ids appearing more than once -- double-counting inflates significance."""
    ids = np.asarray(unit_ids)
    values, counts = np.unique(ids, return_counts=True)
    return values[counts > 1]
