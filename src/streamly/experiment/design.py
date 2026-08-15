"""Experiment design: sample size, power, MDE, and duration.

Everything here is closed-form normal-approximation, which is the right call for
Streamly's traffic: at tens of thousands of users per arm the normal
approximation to the binomial is accurate to well under a tenth of a percentage
point, and a closed form is auditable by anyone with a stats textbook. Exact
methods (Fisher, Clopper-Pearson) matter at n < 100, not here.

The four questions this module answers, which are the four a PM actually asks:

* "How many users do I need?"          -> :func:`sample_size_two_proportions`
* "How long will that take?"           -> :func:`duration_days`
* "What can I detect in two weeks?"    -> :func:`mde_two_proportions`
* "What power do I actually have?"     -> :func:`power_two_proportions`

Design decisions
----------------
* **MDE is specified in absolute percentage points**, never relative. "A 10%
  lift" is ambiguous (10pp? 10% of 8% = 0.8pp?) and that ambiguity is a common
  source of underpowered experiments. Relative helpers convert explicitly.
* **The test statistic uses a pooled variance, the CI uses unpooled** -- the
  standard convention, mirrored in :mod:`streamly.experiment.frequentist` so
  design and analysis agree about what is being estimated.
* **Unequal allocation is supported** via ``ratio``, because guardrail-heavy
  launches often run 90/10.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DesignResult:
    """A fully specified experiment design."""

    n_control: int
    n_treatment: int
    n_total: int
    baseline_rate: float
    mde_absolute: float
    mde_relative: float
    alpha: float
    power: float
    ratio: float

    def __str__(self) -> str:
        return (
            f"{self.n_total:,} users ({self.n_control:,} control / "
            f"{self.n_treatment:,} treatment) to detect {self.mde_absolute:+.2%} "
            f"({self.mde_relative:+.1%} relative) on a {self.baseline_rate:.2%} "
            f"baseline at alpha={self.alpha}, power={self.power:.0%}"
        )


def _z_alpha(alpha: float, two_sided: bool = True) -> float:
    return float(stats.norm.ppf(1.0 - alpha / (2.0 if two_sided else 1.0)))


def _validate_rate(p: float, name: str) -> None:
    if not 0.0 < p < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {p}")


def sample_size_two_proportions(
    baseline_rate: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    ratio: float = 1.0,
) -> DesignResult:
    """Per-arm sample size for a two-proportion test.

    Standard normal-approximation formula with a pooled null variance and an
    unpooled alternative variance::

        n_c = [ z_{1-a/2} sqrt((1 + 1/k) p̄ q̄) + z_{1-b} sqrt(p1 q1 + p2 q2 / k) ]^2 / Δ^2

    where ``k = ratio`` is treatment-to-control allocation and
    ``p̄ = (p1 + k p2) / (1 + k)``.

    Parameters
    ----------
    baseline_rate:
        Control conversion rate.
    mde_absolute:
        Smallest effect worth detecting, in absolute rate points (0.012 = +1.2pp).
    ratio:
        Treatment users per control user. 1.0 is a 50/50 split.
    """
    _validate_rate(baseline_rate, "baseline_rate")
    if mde_absolute == 0:
        raise ValueError("mde_absolute must be non-zero")
    if ratio <= 0:
        raise ValueError("ratio must be positive")

    p1 = baseline_rate
    p2 = p1 + mde_absolute
    _validate_rate(p2, "baseline_rate + mde_absolute")

    k = ratio
    p_bar = (p1 + k * p2) / (1.0 + k)
    z_a = _z_alpha(alpha, two_sided)
    z_b = float(stats.norm.ppf(power))

    null_sd = np.sqrt((1.0 + 1.0 / k) * p_bar * (1.0 - p_bar))
    alt_sd = np.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2) / k)
    n_control = int(np.ceil(((z_a * null_sd + z_b * alt_sd) / mde_absolute) ** 2))
    n_treatment = int(np.ceil(n_control * k))

    return DesignResult(
        n_control=n_control,
        n_treatment=n_treatment,
        n_total=n_control + n_treatment,
        baseline_rate=p1,
        mde_absolute=mde_absolute,
        mde_relative=mde_absolute / p1,
        alpha=alpha,
        power=power,
        ratio=k,
    )


def power_two_proportions(
    n_control: int,
    baseline_rate: float,
    mde_absolute: float,
    alpha: float = 0.05,
    two_sided: bool = True,
    ratio: float = 1.0,
) -> float:
    """Achieved power for a given per-arm sample size.

    Exact algebraic inverse of :func:`sample_size_two_proportions`, so the two
    are guaranteed consistent (asserted in the test suite).
    """
    _validate_rate(baseline_rate, "baseline_rate")
    p1 = baseline_rate
    p2 = p1 + mde_absolute
    _validate_rate(p2, "baseline_rate + mde_absolute")

    k = ratio
    p_bar = (p1 + k * p2) / (1.0 + k)
    z_a = _z_alpha(alpha, two_sided)
    null_sd = np.sqrt((1.0 + 1.0 / k) * p_bar * (1.0 - p_bar))
    alt_sd = np.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2) / k)

    z_b = (abs(mde_absolute) * np.sqrt(n_control) - z_a * null_sd) / alt_sd
    return float(stats.norm.cdf(z_b))


def mde_two_proportions(
    n_control: int,
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    ratio: float = 1.0,
) -> float:
    """Smallest absolute effect detectable with a fixed sample.

    Solved by bisection on :func:`power_two_proportions` rather than in closed
    form, because the pooled/unpooled variance mix makes the direct inversion
    messy and this is exact to 1e-9 in a few dozen iterations.
    """
    _validate_rate(baseline_rate, "baseline_rate")
    lo, hi = 1e-9, min(0.5, 1.0 - baseline_rate - 1e-9)
    if power_two_proportions(n_control, baseline_rate, hi, alpha, two_sided, ratio) < power:
        raise ValueError(
            f"n={n_control:,} cannot reach power={power:.0%} for any effect "
            f"on a {baseline_rate:.2%} baseline"
        )
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if power_two_proportions(n_control, baseline_rate, mid, alpha, two_sided, ratio) < power:
            lo = mid
        else:
            hi = mid
    return hi


def sample_size_means(
    sd: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """Per-arm sample size for a difference in means (continuous metric)."""
    if sd <= 0:
        raise ValueError("sd must be positive")
    if mde_absolute == 0:
        raise ValueError("mde_absolute must be non-zero")
    z_a = _z_alpha(alpha, two_sided)
    z_b = float(stats.norm.ppf(power))
    return int(np.ceil(2.0 * ((z_a + z_b) * sd / mde_absolute) ** 2))


def duration_days(
    n_total: int, daily_eligible_users: float, exposure_share: float = 1.0
) -> float:
    """Calendar days to accrue ``n_total`` assignments.

    ``exposure_share`` is the fraction of daily traffic actually entering the
    experiment (a 10% ramp is ``0.10``). Returns fractional days; round up and
    add whole weeks in practice, since weekday/weekend mix is itself a covariate.
    """
    if daily_eligible_users <= 0 or not 0 < exposure_share <= 1:
        raise ValueError("daily_eligible_users must be > 0 and exposure_share in (0, 1]")
    return n_total / (daily_eligible_users * exposure_share)


def cuped_variance_factor(rho: float) -> float:
    """Variance multiplier from CUPED given covariate correlation ``rho``.

    Variance falls by ``1 - rho^2``, so sample size for fixed power scales by the
    same factor. rho=0.6 -> 36% fewer users; rho=0.3 -> only 9%. The quadratic
    is why weak covariates are not worth the complexity.
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    return 1.0 - rho ** 2


def sample_size_with_cuped(
    baseline_rate: float,
    mde_absolute: float,
    rho: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> DesignResult:
    """Sample size after CUPED variance reduction at correlation ``rho``."""
    base = sample_size_two_proportions(baseline_rate, mde_absolute, alpha, power)
    factor = cuped_variance_factor(rho)
    n_c = int(np.ceil(base.n_control * factor))
    return DesignResult(
        n_control=n_c,
        n_treatment=n_c,
        n_total=2 * n_c,
        baseline_rate=baseline_rate,
        mde_absolute=mde_absolute,
        mde_relative=mde_absolute / baseline_rate,
        alpha=alpha,
        power=power,
        ratio=1.0,
    )
