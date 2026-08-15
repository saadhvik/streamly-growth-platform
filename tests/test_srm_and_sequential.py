"""Phase 5 acceptance tests: SRM detection, sequential control under peeking, guardrails.

Two gates:

1. **A seeded SRM break is detected** -- and, just as important, a healthy
   experiment is not flagged. A check that fires on clean data is worse than no
   check, because teams learn to ignore it.
2. **Type-I error is controlled under peeking.** Naive repeated testing is shown
   to inflate the false-positive rate to ~14%, and the alpha-spending
   boundaries are shown to pull it back to the nominal 5% -- both measured here
   by simulation, not asserted from a citation.

Boundary correctness is validated three independent ways: against the exact
analytic value at the first look, against published Lan-DeMets properties, and
by direct Monte Carlo of the whole stopping rule.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from scipy import stats

os.environ.setdefault("STREAMLY_DATA_DIR", tempfile.mkdtemp(prefix="streamly_test_"))

from streamly.config import EXPERIMENT                                     # noqa: E402
from streamly.datagen import dgp                                           # noqa: E402
from streamly.experiment import assign, guardrails, integrity, sequential  # noqa: E402
from streamly.experiment.guardrails import Direction, Verdict              # noqa: E402

LOOKS = (0.2, 0.4, 0.6, 0.8, 1.0)
ALPHA = 0.05


# ---------------------------------------------------------------------------
# SRM
# ---------------------------------------------------------------------------
def test_healthy_split_passes_srm() -> None:
    res = integrity.srm_check({"control": 29_967, "treatment": 30_033})
    assert res.passed, str(res)
    assert res.p_value > 0.05
    assert res.degrees_of_freedom == 1


def test_seeded_srm_break_is_detected() -> None:
    """THE Phase 5 gate: a 4% routing loss on treatment must be caught.

    30,000 vs 28,800 is what a misconfigured feature flag or a dropped event
    batch looks like -- a 1pp shift in realized split, easy to miss on a
    dashboard, fatal to the analysis.
    """
    res = integrity.srm_check({"control": 30_000, "treatment": 28_800})
    assert not res.passed, str(res)
    assert res.p_value < integrity.SRM_ALPHA
    assert res.worst_variant in ("control", "treatment")
    assert abs(res.worst_delta_units) == pytest.approx(600.0, abs=1.0)


@pytest.mark.parametrize("loss", [0.03, 0.05, 0.10])
def test_srm_sensitivity_scales_with_the_break_size(loss: float) -> None:
    """Losses at or above the documented floor are all caught."""
    arm = 30_000
    treatment = int(arm * (1.0 - loss))
    res = integrity.srm_check({"control": arm, "treatment": treatment})
    assert not res.passed, f"a {loss:.0%} arm loss should be caught"


def test_srm_sensitivity_floor_is_documented_not_hidden() -> None:
    """The cost of alpha=0.001: small breaks are genuinely invisible.

    A 1% loss at this sample size does NOT fire, and pretending otherwise would
    make a passing SRM look like a guarantee. The helper states the threshold so
    the limitation is quotable rather than discovered later.
    """
    floor = integrity.srm_minimum_detectable_loss(30_000)
    assert 0.02 < floor < 0.03, floor

    below = integrity.srm_check({"control": 30_000, "treatment": 29_700})   # 1% loss
    assert below.passed, "a 1% loss is below the detection floor at n=60k"

    just_above = integrity.srm_check(
        {"control": 30_000, "treatment": int(30_000 * (1 - floor * 1.2))}
    )
    assert not just_above.passed

    # Sensitivity improves with scale, as it must.
    assert integrity.srm_minimum_detectable_loss(3_000_000) < floor / 5


def test_srm_false_positive_rate_matches_its_alpha() -> None:
    """A clean 50/50 must almost never fire -- that is what makes it credible."""
    rng = np.random.default_rng(7)
    sims = 20_000
    control = rng.binomial(60_000, 0.5, sims)
    fires = sum(
        not integrity.srm_check({"control": int(c), "treatment": 60_000 - int(c)}).passed
        for c in control
    )
    rate = fires / sims
    assert rate < 0.004, f"false-positive rate {rate:.4f} too high for alpha=0.001"


def test_srm_handles_intended_unequal_splits() -> None:
    """A 90/10 ramp is not an SRM; treating it as one would block every ramp."""
    ok = integrity.srm_check(
        {"control": 54_000, "treatment": 6_000},
        expected_split={"control": 0.9, "treatment": 0.1},
    )
    assert ok.passed, str(ok)
    # The same counts judged against a 50/50 expectation are a catastrophic SRM.
    assert not integrity.srm_check({"control": 54_000, "treatment": 6_000}).passed


def test_srm_supports_more_than_two_arms() -> None:
    res = integrity.srm_check({"control": 20_000, "variant_a": 20_100, "variant_b": 19_900})
    assert res.degrees_of_freedom == 2
    assert res.passed


@pytest.mark.parametrize(
    ("observed", "kwargs", "match"),
    [
        ({"control": 100}, {}, "at least two variants"),
        ({"control": 0, "treatment": 0}, {}, "no units"),
        ({"control": -1, "treatment": 10}, {}, "non-negative"),
        ({"control": 1, "treatment": 1}, {"expected_split": {"control": 1.0}}, "missing variants"),
    ],
)
def test_srm_rejects_invalid_inputs(observed: dict, kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        integrity.srm_check(observed, **kwargs)


# ---------------------------------------------------------------------------
# Assignment integrity beyond counts
# ---------------------------------------------------------------------------
def test_assignment_reproducibility_catches_drift() -> None:
    ids = np.arange(1, 20_001)
    recorded = assign.assign_many(ids, EXPERIMENT.salt)
    ok, rate, bad = integrity.check_assignment_reproducibility(ids, recorded, EXPERIMENT.salt)
    assert ok and rate == 0.0 and bad.size == 0

    # Flip 50 users, as a drifted assignment service would.
    drifted = recorded.copy()
    flip = slice(0, 50)
    drifted[flip] = np.where(drifted[flip] == "control", "treatment", "control")
    ok, rate, bad = integrity.check_assignment_reproducibility(ids, drifted, EXPERIMENT.salt)
    assert not ok and bad.size == 50
    assert rate == pytest.approx(50 / 20_000)


def test_covariate_balance_passes_when_randomized_and_fails_when_not() -> None:
    rng = np.random.default_rng(3)
    balanced = integrity.covariate_balance(rng.normal(size=20_000), rng.normal(size=20_000))
    assert balanced.passed, str(balanced)

    skewed = integrity.covariate_balance(
        rng.normal(0.0, 1.0, 20_000), rng.normal(0.25, 1.0, 20_000)
    )
    assert not skewed.passed
    assert skewed.standardized_difference == pytest.approx(0.25, abs=0.03)


def test_duplicate_units_are_surfaced() -> None:
    dupes = integrity.duplicate_units(np.array([1, 2, 3, 2, 4, 4, 4]))
    assert sorted(dupes.tolist()) == [2, 4]
    assert integrity.duplicate_units(np.arange(100)).size == 0


# ---------------------------------------------------------------------------
# Sequential boundaries
# ---------------------------------------------------------------------------
def test_first_look_boundary_matches_the_analytic_value() -> None:
    """With no prior looks the boundary is exactly the spending function's quantile."""
    plan = sequential.compute_boundaries(LOOKS, ALPHA, "obrien_fleming")
    expected = float(stats.norm.ppf(1 - ALPHA / 2) / np.sqrt(LOOKS[0]))
    assert plan.z_boundaries[0] == pytest.approx(expected, abs=1e-3)


def test_obrien_fleming_boundaries_are_strict_early_and_relax() -> None:
    plan = sequential.compute_boundaries(LOOKS, ALPHA, "obrien_fleming")
    b = plan.z_boundaries
    assert all(x > y for x, y in zip(b, b[1:])), "OBF boundaries must decrease"
    assert b[0] > 4.0, "an early stop must demand overwhelming evidence"
    assert b[-1] == pytest.approx(2.06, abs=0.05), "final look near the nominal 1.96"
    assert b[-1] > stats.norm.ppf(1 - ALPHA / 2), "final boundary still above nominal"


def test_pocock_boundaries_are_nearly_constant() -> None:
    plan = sequential.compute_boundaries(LOOKS, ALPHA, "pocock")
    b = np.array(plan.z_boundaries)
    assert b.std() < 0.03, "Pocock spends evenly, so boundaries barely move"
    assert b.mean() == pytest.approx(2.41, abs=0.05)


def test_pocock_stops_earlier_but_pays_at_the_end() -> None:
    obf = sequential.compute_boundaries(LOOKS, ALPHA, "obrien_fleming")
    poc = sequential.compute_boundaries(LOOKS, ALPHA, "pocock")
    assert poc.z_boundaries[0] < obf.z_boundaries[0]     # easier to stop early
    assert poc.z_boundaries[-1] > obf.z_boundaries[-1]   # harder to win at the end


def test_alpha_spending_is_monotone_and_exhausts_the_budget() -> None:
    for name in sequential.SPENDING_FUNCTIONS:
        plan = sequential.compute_boundaries(LOOKS, ALPHA, name)
        spent = plan.cumulative_alpha_spent
        assert all(a <= b + 1e-12 for a, b in zip(spent, spent[1:])), name
        assert spent[-1] == pytest.approx(ALPHA, rel=1e-9), name
        assert sum(plan.incremental_alpha) == pytest.approx(ALPHA, rel=1e-9), name


def test_a_single_look_reduces_to_the_fixed_sample_test() -> None:
    plan = sequential.compute_boundaries((1.0,), ALPHA)
    assert plan.z_boundaries[0] == pytest.approx(stats.norm.ppf(1 - ALPHA / 2), abs=1e-3)


@pytest.mark.parametrize(
    ("looks", "kwargs", "match"),
    [
        ((), {}, "at least one look"),
        ((0.5, 0.5), {}, "strictly increasing"),
        ((0.5, 1.5), {}, "must lie in"),
        ((0.5, 1.0), {"spending": "nope"}, "unknown spending function"),
        ((0.5, 1.0), {"alpha": 1.5}, "alpha must be"),
    ],
)
def test_boundary_computation_rejects_invalid_plans(
    looks: tuple, kwargs: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        sequential.compute_boundaries(looks, **kwargs)


# ---------------------------------------------------------------------------
# The peeking gate
# ---------------------------------------------------------------------------
def _simulate_looks(sims: int, looks: tuple[float, ...], drift: float, seed: int) -> np.ndarray:
    """Z-statistics across looks for ``sims`` experiments.

    Brownian motion: ``B(t)`` has independent increments with variance equal to
    the information increment, and ``Z(t) = B(t)/sqrt(t)``. ``drift`` is the
    effect size in units of the final-analysis z-statistic.
    """
    rng = np.random.default_rng(seed)
    t = np.array(looks)
    increments = rng.normal(0.0, np.sqrt(np.diff(np.concatenate([[0.0], t]))), (sims, t.size))
    b = np.cumsum(increments, axis=1) + drift * t
    return b / np.sqrt(t)


def test_naive_peeking_inflates_type_one_error() -> None:
    """The problem, measured: five looks at a fixed 0.05 gives ~14%, not 5%."""
    rate = sequential.naive_peeking_type_one_error(len(LOOKS), ALPHA, simulations=40_000, seed=2)
    assert rate == pytest.approx(0.142, abs=0.01), rate
    assert rate > 2.5 * ALPHA

    # And it keeps getting worse the more you look.
    more = sequential.naive_peeking_type_one_error(20, ALPHA, simulations=40_000, seed=2)
    assert more > rate


@pytest.mark.parametrize("spending", ["obrien_fleming", "pocock", "linear"])
def test_alpha_spending_holds_type_one_error_at_alpha_under_peeking(spending: str) -> None:
    """THE Phase 5 gate: peek at every look and still land on nominal alpha."""
    plan = sequential.compute_boundaries(LOOKS, ALPHA, spending)
    z = _simulate_looks(sims=100_000, looks=LOOKS, drift=0.0, seed=13)
    crossed = (np.abs(z) >= np.array(plan.z_boundaries)).any(axis=1)
    rate = float(crossed.mean())
    # SE over 100k sims at p=0.05 is 0.0007; 0.004 is well outside noise.
    assert rate == pytest.approx(ALPHA, abs=0.004), f"{spending}: {rate:.4f}"


def test_sequential_testing_retains_power_against_a_real_effect() -> None:
    """The insurance is close to free: OBF keeps ~the fixed-sample power."""
    drift = float(stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(0.80))  # 80%-power effect
    z = _simulate_looks(sims=40_000, looks=LOOKS, drift=drift, seed=17)

    plan = sequential.compute_boundaries(LOOKS, ALPHA, "obrien_fleming")
    seq_power = float((np.abs(z) >= np.array(plan.z_boundaries)).any(axis=1).mean())
    fixed_power = float((np.abs(z[:, -1]) >= stats.norm.ppf(1 - ALPHA / 2)).mean())

    assert fixed_power == pytest.approx(0.80, abs=0.02)
    assert seq_power > fixed_power - 0.03, "OBF should cost almost no power"


def test_obrien_fleming_stops_early_on_a_large_effect() -> None:
    """The payoff: a big effect is caught before the full sample is spent."""
    z = _simulate_looks(sims=20_000, looks=LOOKS, drift=6.0, seed=23)
    plan = sequential.compute_boundaries(LOOKS, ALPHA, "obrien_fleming")
    crossed = np.abs(z) >= np.array(plan.z_boundaries)
    first = np.argmax(crossed, axis=1)[crossed.any(axis=1)]
    assert float((first < len(LOOKS) - 1).mean()) > 0.9, "most runs should stop early"


def test_evaluate_look_and_sequence_agree() -> None:
    plan = sequential.compute_boundaries(LOOKS, ALPHA, "obrien_fleming")
    assert not sequential.evaluate_look(plan, 1, 3.0).crossed     # 3.0 < 4.38 boundary
    assert sequential.evaluate_look(plan, 1, 5.0).crossed
    assert sequential.evaluate_look(plan, 5, 2.5).crossed

    # 2.1 sits under the look-4 boundary (2.254); 2.2 clears the look-5 one (2.064).
    stopped = sequential.evaluate_sequence(plan, [1.0, 1.5, 2.0, 2.1, 2.2])
    assert stopped.look == 5 and stopped.crossed
    early = sequential.evaluate_sequence(plan, [5.0, 1.0])
    assert early.look == 1 and early.crossed and early.stop

    continuing = sequential.evaluate_sequence(plan, [1.0, 1.2])
    assert not continuing.crossed and not continuing.stop
    with pytest.raises(ValueError, match="look must be in"):
        sequential.evaluate_look(plan, 9, 1.0)


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
def test_within_tolerance_drift_passes() -> None:
    """6.0% -> 6.3% refunds against a 1pp tolerance, at n large enough to tell."""
    res = guardrails.guardrail_proportion(
        "refund_rate", 3_000, 50_000, 3_150, 50_000, margin=0.01
    )
    assert res.verdict is Verdict.PASS, str(res)
    assert not res.blocks_launch
    assert res.harm == pytest.approx(0.003, abs=1e-9)


def test_seeded_regression_beyond_the_margin_fails() -> None:
    """6.0% -> 8.5% refunds blows through a 1pp tolerance."""
    res = guardrails.guardrail_proportion(
        "refund_rate", 3_000, 50_000, 4_250, 50_000, margin=0.01
    )
    assert res.verdict is Verdict.FAIL, str(res)
    assert res.blocks_launch
    assert res.harm_ci_low > res.margin


def test_underpowered_guardrail_is_inconclusive_not_a_pass() -> None:
    """The failure mode this design exists to prevent: absence of evidence.

    At n=500 per arm a 1pp margin is far inside the noise. A significance test
    would report 'not significant' and ship; the non-inferiority framing
    correctly refuses to conclude.
    """
    res = guardrails.guardrail_proportion("refund_rate", 30, 500, 33, 500, margin=0.01)
    assert res.verdict is Verdict.INCONCLUSIVE, str(res)
    assert res.blocks_launch, "inconclusive must block, not pass"
    assert res.test.p_value > 0.05, "a naive significance test would have passed this"


def test_higher_is_better_guardrail_flips_the_harm_sign() -> None:
    """Retention dropping is harm even though the effect is negative."""
    res = guardrails.guardrail_proportion(
        "d7_retention", 20_000, 50_000, 18_500, 50_000,
        margin=0.01, direction=Direction.HIGHER_IS_BETTER,
    )
    assert res.harm == pytest.approx(0.03, abs=1e-9), "a 3pp drop is +3pp of harm"
    assert res.verdict is Verdict.FAIL
    assert res.harm_ci_low < res.harm_ci_high


def test_an_improving_guardrail_passes_comfortably() -> None:
    res = guardrails.guardrail_proportion(
        "refund_rate", 3_000, 50_000, 2_500, 50_000, margin=0.01
    )
    assert res.verdict is Verdict.PASS
    assert res.harm < 0, "an improvement is negative harm"


def test_latency_guardrail_on_a_continuous_metric() -> None:
    rng = np.random.default_rng(41)
    control = rng.normal(220.0, 30.0, 30_000)
    ok = rng.normal(224.0, 30.0, 30_000)        # +4ms, inside a 10ms tolerance
    bad = rng.normal(245.0, 30.0, 30_000)       # +25ms, well beyond it

    passing = guardrails.guardrail_mean("latency_ms", control, ok, margin=10.0)
    failing = guardrails.guardrail_mean("latency_ms", control, bad, margin=10.0)
    assert passing.verdict is Verdict.PASS, str(passing)
    assert failing.verdict is Verdict.FAIL, str(failing)
    assert failing.harm == pytest.approx(25.0, abs=1.0)


def test_guardrail_report_blocks_when_any_guardrail_blocks() -> None:
    good = guardrails.guardrail_proportion("refunds", 3_000, 50_000, 3_050, 50_000, margin=0.01)
    bad = guardrails.guardrail_proportion("crashes", 500, 50_000, 1_500, 50_000, margin=0.005)
    report = guardrails.evaluate_guardrails([good, bad])
    assert not report.passed
    assert [r.name for r in report.blocking] == ["crashes"]
    assert guardrails.evaluate_guardrails([good]).passed


def test_margin_must_be_a_positive_tolerance() -> None:
    with pytest.raises(ValueError, match="margin must be positive"):
        guardrails.guardrail_proportion("x", 1, 10, 1, 10, margin=0.0)
    with pytest.raises(ValueError, match="margin must be positive"):
        guardrails.guardrail_mean("x", np.zeros(10), np.ones(10), margin=-1.0)


# ---------------------------------------------------------------------------
# End-to-end on the injected experiment
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def injected():
    return dgp.simulate_experiment(EXPERIMENT, np.arange(1, 60_001, dtype=np.int64))


def test_the_real_experiment_passes_integrity_checks(injected) -> None:
    variant = injected["variant"]
    counts = {v: int((variant == v).sum()) for v in ("control", "treatment")}
    srm = integrity.srm_check(counts, {"control": EXPERIMENT.true_traffic_split,
                                       "treatment": 1 - EXPERIMENT.true_traffic_split})
    assert srm.passed, str(srm)

    x = injected["pre_covariate"]
    is_c = variant == "control"
    balance = integrity.covariate_balance(x[is_c], x[~is_c])
    assert balance.passed, str(balance)


def test_the_real_guardrails_are_within_tolerance(injected) -> None:
    """Refund drift of 6.0% -> 6.6% against a 1pp tolerance, on converters."""
    variant = injected["variant"]
    refund, converted = injected["guardrail_refund"], injected["primary_metric"] > 0
    is_c = variant == "control"

    res = guardrails.guardrail_proportion(
        "refund_rate",
        int(refund[is_c & converted].sum()), int((is_c & converted).sum()),
        int(refund[~is_c & converted].sum()), int((~is_c & converted).sum()),
        margin=0.05,
    )
    assert res.verdict is not Verdict.FAIL, str(res)

    latency = injected["guardrail_latency_ms"]
    lat = guardrails.guardrail_mean("latency_ms", latency[is_c], latency[~is_c], margin=10.0)
    assert lat.verdict is Verdict.PASS, str(lat)


# ---------------------------------------------------------------------------
# Guardrail power -- answering the margin question at intake, not at readout
# ---------------------------------------------------------------------------
def test_resolvable_margin_and_required_n_are_inverses() -> None:
    base, n = 0.005, 30_000
    margin = guardrails.minimum_resolvable_margin(base, n, n)
    assert guardrails.required_n_for_margin(base, margin) == pytest.approx(n, rel=0.01)


def test_a_margin_below_the_resolvable_floor_can_never_pass() -> None:
    """The design defect this calculator exists to catch.

    With a margin under the floor, even a treatment with *zero* true harm
    returns INCONCLUSIVE -- the guardrail blocks the launch while telling you
    nothing about the treatment.
    """
    n, rate = 30_000, 0.005
    floor = guardrails.minimum_resolvable_margin(rate, n, n)
    events = int(n * rate)

    doomed = guardrails.guardrail_proportion(
        "refunds", events, n, events, n, margin=floor * 0.5
    )
    assert doomed.verdict is Verdict.INCONCLUSIVE
    assert not doomed.adequately_powered

    workable = guardrails.guardrail_proportion(
        "refunds", events, n, events, n, margin=floor * 1.5
    )
    assert workable.verdict is Verdict.PASS
    assert workable.adequately_powered


def test_conditioning_on_a_subpopulation_destroys_guardrail_power() -> None:
    """Refunds among converters vs among all assigned users, same underlying rate."""
    all_users = guardrails.minimum_resolvable_margin(0.005, 30_000, 30_000)
    converters_only = guardrails.minimum_resolvable_margin(0.06, 2_400, 2_400)
    assert converters_only > 5 * all_users, (
        "conditioning on conversion should cost roughly an order of magnitude"
    )


def test_required_n_scales_quadratically_as_the_margin_tightens() -> None:
    loose = guardrails.required_n_for_margin(0.005, 0.0025)
    tight = guardrails.required_n_for_margin(0.005, 0.00125)
    assert tight / loose == pytest.approx(4.0, rel=0.02)


def test_report_surfaces_underpowered_guardrails_separately() -> None:
    good = guardrails.guardrail_proportion("refunds", 150, 30_000, 160, 30_000, margin=0.0025)
    doomed = guardrails.guardrail_proportion("crashes", 30, 500, 32, 500, margin=0.001)
    report = guardrails.evaluate_guardrails([good, doomed])
    assert [r.name for r in report.underpowered] == ["crashes"]
    assert good.adequately_powered


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"base_rate": 0.0, "n_control": 10, "n_treatment": 10}, "base_rate must be in"),
        ({"base_rate": 0.05, "n_control": 0, "n_treatment": 10}, "at least one unit"),
    ],
)
def test_resolvable_margin_rejects_invalid_inputs(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        guardrails.minimum_resolvable_margin(**kwargs)


def test_required_n_rejects_an_unreachable_margin() -> None:
    with pytest.raises(ValueError, match="must exceed the assumed harm"):
        guardrails.required_n_for_margin(0.005, 0.001, assumed_harm=0.002)
