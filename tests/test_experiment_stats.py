"""Phase 4 acceptance tests: design, assignment, and frequentist analysis.

The gate is "matches closed-form / known-answer cases within tolerance", so
every statistic is checked against an *independent* source rather than against
itself:

* two-proportion z  -> ``statsmodels.proportions_ztest`` and a scipy chi-square
* Welch t           -> ``scipy.stats.ttest_ind(equal_var=False)``
* power / sample size -> Monte Carlo: the design's promised power is measured
  empirically, and Type-I error is measured under a true null
* CUPED             -> the theoretical ``1 - rho^2`` variance reduction, plus an
  unbiasedness check against a known injected lift

Simulation-based tests use fixed seeds and tolerances sized from the simulation
standard error, so they are deterministic rather than flaky.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

os.environ.setdefault("STREAMLY_DATA_DIR", tempfile.mkdtemp(prefix="streamly_test_"))

from streamly.config import EXPERIMENT  # noqa: E402
from streamly.datagen import dgp  # noqa: E402
from streamly.experiment import assign, design, frequentist  # noqa: E402

ALPHA = 0.05


# ---------------------------------------------------------------------------
# Design: sample size, power, MDE
# ---------------------------------------------------------------------------
def test_sample_size_and_power_are_exact_inverses() -> None:
    d = design.sample_size_two_proportions(0.080, 0.012, alpha=ALPHA, power=0.80)
    achieved = design.power_two_proportions(d.n_control, 0.080, 0.012, alpha=ALPHA)
    # n is rounded up, so achieved power meets or slightly exceeds the target.
    assert 0.80 <= achieved < 0.801


def test_sample_size_matches_published_reference() -> None:
    """Textbook case: 10% baseline, +2pp MDE, alpha=0.05, power=0.80.

    Standard references (Fleiss without continuity correction, Evan Miller's
    calculator) put this at ~3,835 per arm.
    """
    d = design.sample_size_two_proportions(0.10, 0.02, alpha=0.05, power=0.80)
    assert 3_700 <= d.n_control <= 3_900, d.n_control
    assert d.n_total == 2 * d.n_control
    assert d.mde_relative == pytest.approx(0.20)


def test_design_delivers_its_promised_power_in_simulation() -> None:
    """Monte Carlo: run the designed experiment 4,000 times, count rejections."""
    rng = np.random.default_rng(11)
    p_c, mde, sims = 0.08, 0.02, 4_000
    d = design.sample_size_two_proportions(p_c, mde, alpha=ALPHA, power=0.80)

    x_c = rng.binomial(d.n_control, p_c, sims)
    x_t = rng.binomial(d.n_treatment, p_c + mde, sims)
    rejects = sum(
        frequentist.two_proportion_z_test(
            int(a), d.n_control, int(b), d.n_treatment, alpha=ALPHA
        ).significant
        for a, b in zip(x_c, x_t)
    )
    empirical = rejects / sims
    # SE of a proportion at 0.8 over 4,000 sims is 0.0063; 4 SE ~ 0.025.
    assert empirical == pytest.approx(0.80, abs=0.025), empirical


def test_type_one_error_is_held_at_alpha_under_a_true_null() -> None:
    """A/A calibration: with no effect, rejection rate must sit at alpha."""
    rng = np.random.default_rng(23)
    n, p, sims = 5_000, 0.08, 4_000
    x_c = rng.binomial(n, p, sims)
    x_t = rng.binomial(n, p, sims)
    rejects = sum(
        frequentist.two_proportion_z_test(int(a), n, int(b), n, alpha=ALPHA).significant
        for a, b in zip(x_c, x_t)
    )
    assert rejects / sims == pytest.approx(ALPHA, abs=0.012)


def test_confidence_intervals_cover_the_truth_at_the_nominal_rate() -> None:
    rng = np.random.default_rng(37)
    n, p_c, lift, sims = 8_000, 0.08, 0.012, 3_000
    x_c = rng.binomial(n, p_c, sims)
    x_t = rng.binomial(n, p_c + lift, sims)
    covered = sum(
        frequentist.two_proportion_z_test(int(a), n, int(b), n, alpha=ALPHA).ci_low
        <= lift
        <= frequentist.two_proportion_z_test(int(a), n, int(b), n, alpha=ALPHA).ci_high
        for a, b in zip(x_c, x_t)
    )
    assert covered / sims == pytest.approx(0.95, abs=0.015)


def test_mde_inverts_sample_size() -> None:
    d = design.sample_size_two_proportions(0.08, 0.012, alpha=ALPHA, power=0.80)
    recovered = design.mde_two_proportions(d.n_control, 0.08, alpha=ALPHA, power=0.80)
    assert recovered == pytest.approx(0.012, rel=0.01)


def test_smaller_effects_need_quadratically_more_users() -> None:
    """Halving the MDE roughly quadruples n.

    Not exactly 4x: the alternative-hypothesis variance term depends on the
    treatment rate, which differs between the two designs (0.10 vs 0.09). The
    observed 3.8x is that second-order correction, not a formula error.
    """
    big = design.sample_size_two_proportions(0.08, 0.020).n_control
    small = design.sample_size_two_proportions(0.08, 0.010).n_control
    assert small / big == pytest.approx(4.0, rel=0.10)


def test_unequal_allocation_costs_total_sample() -> None:
    balanced = design.sample_size_two_proportions(0.08, 0.012, ratio=1.0)
    skewed = design.sample_size_two_proportions(0.08, 0.012, ratio=0.25)
    assert skewed.n_total > balanced.n_total
    assert design.power_two_proportions(
        skewed.n_control, 0.08, 0.012, ratio=0.25
    ) == pytest.approx(0.80, abs=0.005)


def test_duration_and_cuped_helpers() -> None:
    assert design.duration_days(60_000, 5_000) == pytest.approx(12.0)
    assert design.duration_days(60_000, 5_000, exposure_share=0.5) == pytest.approx(24.0)
    assert design.cuped_variance_factor(0.6) == pytest.approx(0.64)
    reduced = design.sample_size_with_cuped(0.08, 0.012, rho=0.6)
    plain = design.sample_size_two_proportions(0.08, 0.012)
    assert reduced.n_control == pytest.approx(plain.n_control * 0.64, rel=0.01)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"baseline_rate": 0.0, "mde_absolute": 0.01}, "strictly between"),
        ({"baseline_rate": 0.08, "mde_absolute": 0.0}, "non-zero"),
        ({"baseline_rate": 0.08, "mde_absolute": 0.01, "ratio": 0.0}, "positive"),
        ({"baseline_rate": 0.98, "mde_absolute": 0.05}, "strictly between"),
    ],
)
def test_design_rejects_invalid_inputs(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        design.sample_size_two_proportions(**kwargs)


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------
def test_bucketing_is_deterministic_and_in_range() -> None:
    h1 = assign.bucket_hash(12345, "salt-a")
    assert h1 == assign.bucket_hash(12345, "salt-a")
    assert 0.0 <= h1 < 1.0
    assert assign.assign(12345, "salt-a") == assign.assign(12345, "salt-a")


def test_bucketing_matches_the_generator_bit_for_bit() -> None:
    """The engine and the DGP must agree, or recorded assignments are fiction."""
    import hashlib

    for uid in (1, 2, 997, 60_000):
        digest = hashlib.sha256(f"{EXPERIMENT.salt}:{uid}".encode()).hexdigest()
        expected = int(digest[:16], 16) / 16 ** 16
        assert assign.bucket_hash(uid, EXPERIMENT.salt) == pytest.approx(expected, abs=0)


def test_bucket_hashes_are_uniform() -> None:
    h = assign.bucket_hashes(np.arange(1, 50_001), EXPERIMENT.salt)
    # Kolmogorov-Smirnov against Uniform(0,1); a structural bias would fail hard.
    assert stats.kstest(h, "uniform").pvalue > 0.01
    assert h.mean() == pytest.approx(0.5, abs=0.01)


def test_split_is_respected_and_srm_free() -> None:
    variants = assign.assign_many(np.arange(1, 60_001), EXPERIMENT.salt, split=0.5)
    assert assign.observed_split(variants) == pytest.approx(0.5, abs=0.005)
    ramp = assign.assign_many(np.arange(1, 60_001), EXPERIMENT.salt, split=0.9)
    assert assign.observed_split(ramp) == pytest.approx(0.9, abs=0.005)


def test_different_salts_produce_independent_assignments() -> None:
    """Concurrent experiments must not confound each other."""
    ids = np.arange(1, 40_001)
    a = assign.assign_many(ids, "experiment-one")
    b = assign.assign_many(ids, "experiment-two")
    agreement = float((a == b).mean())
    assert agreement == pytest.approx(0.5, abs=0.01), "salts must decorrelate bucketing"


def test_multivariate_assignment_hits_its_weights() -> None:
    spec = [
        assign.VariantSpec("control", 0.5),
        assign.VariantSpec("variant_a", 0.3),
        assign.VariantSpec("variant_b", 0.2),
    ]
    v = assign.assign_multivariate(np.arange(1, 60_001), "mv-salt", spec)
    for s in spec:
        assert float((v == s.name).mean()) == pytest.approx(s.weight, abs=0.006)


def test_invalid_assignment_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="split"):
        assign.assign(1, "s", split=1.5)
    with pytest.raises(ValueError, match="sum to 1"):
        assign.assign_multivariate(
            np.arange(10), "s", [assign.VariantSpec("a", 0.4), assign.VariantSpec("b", 0.4)]
        )


# ---------------------------------------------------------------------------
# Frequentist tests vs independent implementations
# ---------------------------------------------------------------------------
def test_two_proportion_z_matches_statsmodels() -> None:
    x_c, n_c, x_t, n_t = 800, 10_000, 900, 10_000
    res = frequentist.two_proportion_z_test(x_c, n_c, x_t, n_t)
    z_sm, p_sm = proportions_ztest([x_t, x_c], [n_t, n_c])  # pooled, as we do
    assert res.statistic == pytest.approx(z_sm, rel=1e-10)
    assert res.p_value == pytest.approx(p_sm, rel=1e-10)


def test_two_proportion_z_squared_equals_the_chi_square_statistic() -> None:
    """z^2 == chi-square without continuity correction -- an independent check."""
    x_c, n_c, x_t, n_t = 640, 8_000, 720, 8_000
    res = frequentist.two_proportion_z_test(x_c, n_c, x_t, n_t)
    table = [[x_c, n_c - x_c], [x_t, n_t - x_t]]
    chi2, p_chi, _, _ = stats.chi2_contingency(table, correction=False)
    assert res.statistic ** 2 == pytest.approx(chi2, rel=1e-9)
    assert res.p_value == pytest.approx(p_chi, rel=1e-9)


def test_welch_t_matches_scipy() -> None:
    rng = np.random.default_rng(5)
    c = rng.normal(10.0, 2.0, 500)
    t = rng.normal(10.4, 3.5, 700)
    res = frequentist.welch_t_test(c, t)
    sp = stats.ttest_ind(t, c, equal_var=False)
    assert res.statistic == pytest.approx(sp.statistic, rel=1e-12)
    assert res.p_value == pytest.approx(sp.pvalue, rel=1e-12)
    assert res.degrees_of_freedom == pytest.approx(sp.df, rel=1e-12)
    lo, hi = sp.confidence_interval(1 - ALPHA)
    assert (res.ci_low, res.ci_high) == (pytest.approx(lo, rel=1e-9), pytest.approx(hi, rel=1e-9))


def test_confidence_interval_and_p_value_agree_at_the_boundary() -> None:
    """A CI that excludes zero must come with p < alpha, and vice versa."""
    rng = np.random.default_rng(19)
    for _ in range(200):
        n = 4_000
        a, b = rng.binomial(n, 0.08), rng.binomial(n, 0.085)
        res = frequentist.two_proportion_z_test(int(a), n, int(b), n, alpha=ALPHA)
        excludes_zero = res.ci_low > 0 or res.ci_high < 0
        # Pooled test vs unpooled interval can disagree in a razor-thin band;
        # require agreement everywhere except within 10% of the alpha boundary.
        if abs(res.p_value - ALPHA) > 0.005:
            assert excludes_zero == res.significant


def test_one_sided_test_is_half_the_two_sided_p_value() -> None:
    res_two = frequentist.two_proportion_z_test(800, 10_000, 900, 10_000)
    res_one = frequentist.two_proportion_z_test(800, 10_000, 900, 10_000, two_sided=False)
    assert res_one.p_value == pytest.approx(res_two.p_value / 2.0, rel=1e-9)


def test_invalid_test_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one unit"):
        frequentist.two_proportion_z_test(0, 0, 1, 10)
    with pytest.raises(ValueError, match="within"):
        frequentist.two_proportion_z_test(11, 10, 1, 10)
    with pytest.raises(ValueError, match="at least two observations"):
        frequentist.welch_t_test(np.array([1.0]), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# CUPED
# ---------------------------------------------------------------------------
def _cuped_sample(rng: np.random.Generator, n: int, lift: float, rho: float):
    """Continuous metric with a pre-covariate at a known correlation."""
    x_c, x_t = rng.normal(size=n), rng.normal(size=n)
    noise = np.sqrt(1.0 - rho ** 2)
    y_c = rho * x_c + noise * rng.normal(size=n)
    y_t = rho * x_t + noise * rng.normal(size=n) + lift
    return y_c, x_c, y_t, x_t


def test_cuped_variance_reduction_matches_rho_squared() -> None:
    rng = np.random.default_rng(101)
    y_c, x_c, y_t, x_t = _cuped_sample(rng, 40_000, lift=0.05, rho=0.7)
    res = frequentist.cuped_two_sample(y_c, x_c, y_t, x_t)
    assert res.correlation == pytest.approx(0.7, abs=0.02)
    assert res.variance_reduction == pytest.approx(0.49, abs=0.02)
    assert res.variance_reduction == pytest.approx(res.theoretical_reduction, abs=0.005)


def test_cuped_narrows_the_interval_and_stays_on_the_truth() -> None:
    """CUPED must buy precision without losing the effect.

    The point estimate is *expected* to move: CUPED corrects for chance
    imbalance in the pre-covariate between arms, which is most of what it is
    for. The correct assertion is that the shift is within sampling noise and
    both intervals still cover the truth -- not that the estimate is unchanged.
    Systematic bias is ruled out separately by the 300-replication test below.
    """
    rng = np.random.default_rng(202)
    true_lift = 0.05
    y_c, x_c, y_t, x_t = _cuped_sample(rng, 40_000, lift=true_lift, rho=0.7)
    res = frequentist.cuped_two_sample(y_c, x_c, y_t, x_t)

    assert res.ci_width_reduction == pytest.approx(0.30, abs=0.03)  # 1 - sqrt(1-rho^2)
    assert res.test.standard_error < res.unadjusted.standard_error
    shift = abs(res.test.absolute_effect - res.unadjusted.absolute_effect)
    assert shift < 1.5 * res.unadjusted.standard_error, "shift must be within noise"
    for r in (res.test, res.unadjusted):
        assert r.ci_low <= true_lift <= r.ci_high


def test_cuped_is_unbiased_across_repeated_experiments() -> None:
    """Mean CUPED estimate over 300 replications must sit on the true lift."""
    rng = np.random.default_rng(303)
    true_lift = 0.05
    estimates = []
    for _ in range(300):
        y_c, x_c, y_t, x_t = _cuped_sample(rng, 3_000, lift=true_lift, rho=0.7)
        estimates.append(frequentist.cuped_two_sample(y_c, x_c, y_t, x_t).test.absolute_effect)
    mean_est = float(np.mean(estimates))
    se = float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))
    assert abs(mean_est - true_lift) < 3 * se, f"{mean_est:.5f} vs {true_lift}"


def test_cuped_is_a_no_op_for_an_uninformative_covariate() -> None:
    rng = np.random.default_rng(404)
    y_c, x_c, y_t, x_t = _cuped_sample(rng, 20_000, lift=0.05, rho=0.0)
    res = frequentist.cuped_two_sample(y_c, x_c, y_t, x_t)
    assert abs(res.theta) < 0.03
    assert abs(res.variance_reduction) < 0.005


def test_cuped_theta_uses_pooled_data() -> None:
    """Pooled theta keeps the effect intact; a per-arm theta would shrink it."""
    rng = np.random.default_rng(505)
    y_c, x_c, y_t, x_t = _cuped_sample(rng, 20_000, lift=0.10, rho=0.7)
    pooled = frequentist.cuped_theta(
        np.concatenate([y_c, y_t]), np.concatenate([x_c, x_t])
    )
    assert pooled == pytest.approx(0.7, abs=0.03)


# ---------------------------------------------------------------------------
# End-to-end on the injected experiment
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def injected():
    """The warehouse's paywall experiment, straight from the DGP."""
    user_ids = np.arange(1, EXPERIMENT_N + 1, dtype=np.int64)
    return dgp.simulate_experiment(EXPERIMENT, user_ids)


EXPERIMENT_N = 60_000


def test_engine_recovers_the_injected_lift(injected) -> None:
    """THE Phase 4 gate: the true injected lift falls inside the 95% CI."""
    variant = injected["variant"]
    y = injected["primary_metric"]
    c, t = y[variant == "control"], y[variant == "treatment"]

    res = frequentist.two_proportion_z_test(
        int(c.sum()), c.size, int(t.sum()), t.size, alpha=ALPHA
    )
    true_lift = EXPERIMENT.true_treatment_lift_abs
    assert res.ci_low <= true_lift <= res.ci_high, (
        f"95% CI [{res.ci_low:.4f}, {res.ci_high:.4f}] must cover the injected "
        f"{true_lift:.4f}"
    )
    assert res.significant, "a +1.2pp lift at n=60,000 should be detected"
    assert res.control_mean == pytest.approx(EXPERIMENT.control_conversion, abs=0.005)


def test_designed_sample_size_is_consistent_with_what_was_run(injected) -> None:
    """Sanity: 60k users is enough to power the injected effect."""
    d = design.sample_size_two_proportions(
        EXPERIMENT.control_conversion, EXPERIMENT.true_treatment_lift_abs, power=0.80
    )
    assert d.n_total < EXPERIMENT_N, (
        f"design wants {d.n_total:,} users; the run has {EXPERIMENT_N:,}"
    )
    achieved = design.power_two_proportions(
        EXPERIMENT_N // 2, EXPERIMENT.control_conversion, EXPERIMENT.true_treatment_lift_abs
    )
    assert achieved > 0.80


def test_cuped_on_the_warehouse_covariate_is_honest_about_its_weakness(injected) -> None:
    """The generator's covariate is weakly correlated; CUPED must say so.

    This asserts the *measured* behaviour, not a hoped-for one: rho is ~0.05,
    so the variance reduction is a fraction of a percent. See the Phase 4 notes
    -- ``ExperimentConfig.cuped_corr`` does not produce the correlation its name
    implies.
    """
    variant = injected["variant"]
    y, x = injected["primary_metric"], injected["pre_covariate"]
    is_c = variant == "control"
    res = frequentist.cuped_two_sample(y[is_c], x[is_c], y[~is_c], x[~is_c])

    assert abs(res.correlation) < 0.10, "covariate is only weakly predictive"
    assert res.variance_reduction < 0.01
    assert res.variance_reduction == pytest.approx(res.theoretical_reduction, abs=1e-3)
    # It must still be unbiased, however little it buys.
    assert res.test.absolute_effect == pytest.approx(
        res.unadjusted.absolute_effect, abs=0.002
    )
