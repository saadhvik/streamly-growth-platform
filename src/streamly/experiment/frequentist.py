"""Frequentist analysis: two-proportion z, Welch t, and CUPED.

Three tests, one convention: every function returns both a p-value **and** a
confidence interval on the effect, never a p-value alone. A bare p-value
answers "could this be zero?" when the decision actually needs "how big is it,
and how sure are we?" -- and a tight interval around a trivial effect is a very
different business outcome from a wide interval around a large one.

Conventions worth stating explicitly
------------------------------------
* **Pooled variance for the test, unpooled for the interval.** The null
  hypothesis asserts a common rate, so the test statistic pools; the interval
  estimates a difference under no such assumption, so it does not. Mixing these
  up produces intervals that disagree with their own p-value near the boundary.
  :mod:`streamly.experiment.design` uses the same convention.
* **Welch, never Student.** Equal variances are an assumption nobody checks and
  experiments routinely violate (treatment often changes spread as well as
  location). Welch costs a fraction of a degree of freedom and removes a whole
  class of silent error.
* **Effects reported absolute and relative.** Absolute drives the ship
  decision; relative is what gets quoted in the readout. Both, always.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    """Outcome of a two-sample comparison."""

    control_mean: float
    treatment_mean: float
    absolute_effect: float          # treatment - control
    relative_effect: float          # absolute / control
    standard_error: float           # SE of the difference (unpooled)
    statistic: float                # z or t
    p_value: float
    ci_low: float                   # CI on the ABSOLUTE effect
    ci_high: float
    alpha: float
    n_control: int
    n_treatment: int
    method: str
    degrees_of_freedom: float | None = None

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero at the stated alpha."""
        return self.p_value < self.alpha

    def __str__(self) -> str:
        return (
            f"{self.method}: {self.control_mean:.4f} -> {self.treatment_mean:.4f} "
            f"({self.absolute_effect:+.4f}, {self.relative_effect:+.2%}), "
            f"{1 - self.alpha:.0%} CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}], "
            f"p={self.p_value:.4g}"
        )


@dataclass(frozen=True)
class CupedResult:
    """A CUPED-adjusted comparison plus the variance reduction achieved."""

    test: TestResult
    theta: float                     # regression coefficient used
    correlation: float               # observed corr(metric, covariate), pooled
    variance_reduction: float        # empirical 1 - var(adjusted)/var(raw)
    theoretical_reduction: float     # rho^2
    unadjusted: TestResult

    @property
    def ci_width_reduction(self) -> float:
        """Fractional narrowing of the confidence interval."""
        raw = self.unadjusted.ci_high - self.unadjusted.ci_low
        adj = self.test.ci_high - self.test.ci_low
        return 1.0 - adj / raw if raw > 0 else 0.0


def two_proportion_z_test(
    conversions_control: int,
    n_control: int,
    conversions_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> TestResult:
    """Two-proportion z-test with a Wald interval on the rate difference."""
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("both arms need at least one unit")
    if not 0 <= conversions_control <= n_control or not 0 <= conversions_treatment <= n_treatment:
        raise ValueError("conversions must lie within [0, n] for each arm")

    p_c = conversions_control / n_control
    p_t = conversions_treatment / n_treatment
    diff = p_t - p_c

    # Pooled SE under H0 for the test statistic.
    p_pool = (conversions_control + conversions_treatment) / (n_control + n_treatment)
    se_pooled = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_control + 1.0 / n_treatment))
    # Unpooled SE for the interval.
    se_unpooled = np.sqrt(
        p_c * (1.0 - p_c) / n_control + p_t * (1.0 - p_t) / n_treatment
    )

    z = diff / se_pooled if se_pooled > 0 else 0.0
    if two_sided:
        p_value = float(2.0 * stats.norm.sf(abs(z)))
        crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    else:
        p_value = float(stats.norm.sf(z))
        crit = float(stats.norm.ppf(1.0 - alpha))

    return TestResult(
        control_mean=p_c,
        treatment_mean=p_t,
        absolute_effect=diff,
        relative_effect=diff / p_c if p_c > 0 else float("nan"),
        standard_error=float(se_unpooled),
        statistic=float(z),
        p_value=p_value,
        ci_low=float(diff - crit * se_unpooled),
        ci_high=float(diff + crit * se_unpooled),
        alpha=alpha,
        n_control=n_control,
        n_treatment=n_treatment,
        method="two-proportion z",
    )


def welch_t_test(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> TestResult:
    """Welch's unequal-variance t-test on two samples of a continuous metric.

    Degrees of freedom follow Welch-Satterthwaite. Matches
    ``scipy.stats.ttest_ind(equal_var=False)`` exactly (asserted in tests); the
    value added here is the interval and the effect decomposition.
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    n_c, n_t = control.size, treatment.size
    if n_c < 2 or n_t < 2:
        raise ValueError("each arm needs at least two observations for a variance estimate")

    m_c, m_t = float(control.mean()), float(treatment.mean())
    v_c, v_t = float(control.var(ddof=1)), float(treatment.var(ddof=1))
    diff = m_t - m_c

    se = np.sqrt(v_c / n_c + v_t / n_t)
    if se == 0:
        raise ValueError("zero variance in both arms; t-test is undefined")

    # Welch-Satterthwaite effective degrees of freedom.
    df = (v_c / n_c + v_t / n_t) ** 2 / (
        (v_c / n_c) ** 2 / (n_c - 1) + (v_t / n_t) ** 2 / (n_t - 1)
    )
    t = diff / se

    if two_sided:
        p_value = float(2.0 * stats.t.sf(abs(t), df))
        crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    else:
        p_value = float(stats.t.sf(t, df))
        crit = float(stats.t.ppf(1.0 - alpha, df))

    return TestResult(
        control_mean=m_c,
        treatment_mean=m_t,
        absolute_effect=float(diff),
        relative_effect=float(diff / m_c) if m_c != 0 else float("nan"),
        standard_error=float(se),
        statistic=float(t),
        p_value=p_value,
        ci_low=float(diff - crit * se),
        ci_high=float(diff + crit * se),
        alpha=alpha,
        n_control=n_c,
        n_treatment=n_t,
        method="Welch t",
        degrees_of_freedom=float(df),
    )


def cuped_theta(metric: np.ndarray, covariate: np.ndarray) -> float:
    """Variance-minimizing CUPED coefficient ``theta = cov(Y, X) / var(X)``.

    Estimated on the **pooled** sample, not per arm. Fitting theta within each
    arm lets the adjustment absorb part of the treatment effect itself, biasing
    the estimate toward zero -- the most common way CUPED is implemented wrong.
    """
    metric = np.asarray(metric, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    var_x = float(covariate.var(ddof=1))
    if var_x <= 0:
        return 0.0
    return float(np.cov(metric, covariate, ddof=1)[0, 1] / var_x)


def cuped_adjust(
    metric: np.ndarray, covariate: np.ndarray, theta: float, covariate_mean: float
) -> np.ndarray:
    """Apply ``Y_adj = Y - theta * (X - E[X])``.

    ``covariate_mean`` must be the **pooled** mean across both arms so the
    adjustment is a common recentering and leaves the between-arm difference
    unbiased.
    """
    metric = np.asarray(metric, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    return metric - theta * (covariate - covariate_mean)


def cuped_two_sample(
    control_metric: np.ndarray,
    control_covariate: np.ndarray,
    treatment_metric: np.ndarray,
    treatment_covariate: np.ndarray,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> CupedResult:
    """CUPED-adjusted two-sample comparison with the variance reduction achieved.

    The covariate must be **pre-experiment**. A covariate measured during the
    experiment can itself be moved by the treatment; adjusting for it then
    removes part of the effect being measured. Nothing in the arithmetic can
    detect that mistake, which is why it is stated here.

    Returns both the adjusted and unadjusted tests so the readout can show what
    the variance reduction actually bought.
    """
    y_c = np.asarray(control_metric, dtype=float)
    y_t = np.asarray(treatment_metric, dtype=float)
    x_c = np.asarray(control_covariate, dtype=float)
    x_t = np.asarray(treatment_covariate, dtype=float)
    if y_c.size != x_c.size or y_t.size != x_t.size:
        raise ValueError("metric and covariate arrays must align within each arm")

    y_all = np.concatenate([y_c, y_t])
    x_all = np.concatenate([x_c, x_t])
    theta = cuped_theta(y_all, x_all)
    x_mean = float(x_all.mean())

    adj_c = cuped_adjust(y_c, x_c, theta, x_mean)
    adj_t = cuped_adjust(y_t, x_t, theta, x_mean)

    unadjusted = welch_t_test(y_c, y_t, alpha, two_sided)
    adjusted = replace(welch_t_test(adj_c, adj_t, alpha, two_sided), method="Welch t (CUPED)")

    raw_var = float(y_all.var(ddof=1))
    adj_var = float(np.concatenate([adj_c, adj_t]).var(ddof=1))
    rho = float(np.corrcoef(y_all, x_all)[0, 1]) if x_all.var() > 0 else 0.0

    return CupedResult(
        test=adjusted,
        theta=theta,
        correlation=rho,
        variance_reduction=1.0 - adj_var / raw_var if raw_var > 0 else 0.0,
        theoretical_reduction=rho ** 2,
        unadjusted=unadjusted,
    )


def _main() -> None:
    """Analyze the injected paywall experiment straight from the warehouse."""
    import json

    from streamly import warehouse
    from streamly.config import EXPERIMENT, GROUND_TRUTH_DIR

    con = warehouse.connect(read_only=True)
    try:
        df = con.execute(
            "SELECT variant, primary_metric, pre_covariate FROM experiment_assignment "
            "WHERE experiment_id = ?", [EXPERIMENT.experiment_id]
        ).fetch_df()
    finally:
        con.close()

    c = df[df["variant"] == "control"]
    t = df[df["variant"] == "treatment"]

    z = two_proportion_z_test(
        int(c["primary_metric"].sum()), len(c),
        int(t["primary_metric"].sum()), len(t),
    )
    cuped = cuped_two_sample(
        c["primary_metric"].to_numpy(), c["pre_covariate"].to_numpy(),
        t["primary_metric"].to_numpy(), t["pre_covariate"].to_numpy(),
    )

    with open(GROUND_TRUTH_DIR / "ground_truth.json") as f:
        true_lift = json.load(f)["experiment"]["true_treatment_lift_abs"]

    print(f"n = {len(c):,} control / {len(t):,} treatment\n")
    print(z)
    print(cuped.unadjusted)
    print(cuped.test)
    print(f"\nCUPED  theta={cuped.theta:.4f}  rho={cuped.correlation:.4f}  "
          f"variance reduction {cuped.variance_reduction:.2%} "
          f"(theoretical {cuped.theoretical_reduction:.2%})")
    print(f"CI width reduction: {cuped.ci_width_reduction:.2%}")
    covered = z.ci_low <= true_lift <= z.ci_high
    print(f"\nTRUE injected lift {true_lift:+.4f} -> "
          f"{'COVERED by' if covered else 'OUTSIDE'} the 95% CI "
          f"[{z.ci_low:+.4f}, {z.ci_high:+.4f}]")


if __name__ == "__main__":
    _main()
