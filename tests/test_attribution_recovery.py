"""Phase 3 acceptance tests: Markov, Shapley, ROI, and recovery vs ground truth.

The gate is :func:`test_ddm_recovers_truth_better_than_last_touch` -- the claim
the whole project rests on. Around it sit axiom tests (Shapley efficiency,
symmetry, null player) and closed-form Markov cases, so a regression tells you
*which* property broke rather than just that a number moved.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("STREAMLY_DATA_DIR", tempfile.mkdtemp(prefix="streamly_test_"))

from streamly.attribution import roi, validate  # noqa: E402
from streamly.attribution.markov import markov_attribution, transition_matrix  # noqa: E402
from streamly.attribution.sessionize import (  # noqa: E402
    JourneySet,
    build_journeys,
    journeys_from_frame,
)
from streamly.attribution.shapley import efficiency_residual, shapley_attribution  # noqa: E402
from streamly.config import CHANNELS  # noqa: E402
from streamly.datagen import generator  # noqa: E402


def _frame(rows: list[tuple[int, str, str, str | None]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["user_id", "channel", "ts", "convert_ts"])
    df["ts"] = pd.to_datetime(df["ts"])
    df["convert_ts"] = pd.to_datetime(df["convert_ts"])
    df["touch_id"] = np.arange(1, len(df) + 1)
    df["cost_attributed"] = 1.0
    df["revenue"] = np.where(df["convert_ts"].notna(), 96.0, 0.0)
    df["is_refunded"] = False
    df["converted"] = df["convert_ts"].notna()
    return df


def _repeat(spec: list[tuple[list[str], bool]], times: int) -> JourneySet:
    """Build a JourneySet from ``(path, converted)`` specs repeated ``times``."""
    rows: list[tuple[int, str, str, str | None]] = []
    uid = 0
    for _ in range(times):
        for path, converted in spec:
            uid += 1
            for day, ch in enumerate(path, start=1):
                rows.append((uid, ch, f"2026-06-{day:02d}", "2026-06-20" if converted else None))
    return journeys_from_frame(_frame(rows))


# ---------------------------------------------------------------------------
# Markov: closed-form cases
# ---------------------------------------------------------------------------
def test_transition_matrix_rows_are_stochastic() -> None:
    js = _repeat([(["google", "meta"], True), (["tiktok"], False)], times=5)
    trans, states = transition_matrix(js)
    np.testing.assert_allclose(trans.sum(axis=1), 1.0, atol=1e-12)
    assert states[0] == "(start)" and states[-2:] == ("(conversion)", "(null)")


def test_removal_effect_of_the_only_converting_channel_is_total() -> None:
    """google converts, meta never does: removing google must zero the chain."""
    js = _repeat([(["google"], True), (["meta"], False)], times=50)
    res = markov_attribution(js)
    assert res.base_conversion_prob == pytest.approx(0.5)
    assert res.removal_effect["google"] == pytest.approx(1.0)
    assert res.removal_effect["meta"] == pytest.approx(0.0)
    assert res.attribution["google"] == pytest.approx(1.0)


def test_markov_attribution_is_a_valid_distribution() -> None:
    js = _repeat([(["google", "email"], True), (["meta", "tiktok"], False),
                  (["referral"], True)], times=20)
    res = markov_attribution(js)
    shares = np.array(list(res.attribution.values()))
    assert shares.sum() == pytest.approx(1.0)
    assert (shares >= 0).all()


# ---------------------------------------------------------------------------
# Shapley: the axioms it is chosen for
# ---------------------------------------------------------------------------
def test_shapley_satisfies_efficiency() -> None:
    js = _repeat([(["google", "email"], True), (["meta"], False),
                  (["google"], True), (["email", "meta"], False)], times=25)
    assert efficiency_residual(shapley_attribution(js)) < 1e-12


def test_shapley_is_symmetric_for_exchangeable_channels() -> None:
    """Two channels appearing in identical contexts must earn identical credit."""
    js = _repeat([
        (["google"], True), (["meta"], True),
        (["google"], False), (["meta"], False),
        (["google", "email"], True), (["meta", "email"], True),
    ], times=20)
    res = shapley_attribution(js)
    assert res.shapley_value["google"] == pytest.approx(res.shapley_value["meta"], abs=1e-12)


def test_shapley_gives_an_absent_channel_zero_credit() -> None:
    """Null-player axiom: a channel with no touches contributes nothing."""
    js = _repeat([(["google"], True), (["meta"], False)], times=30)
    res = shapley_attribution(js)
    for absent in ("tiktok", "email", "referral"):
        assert res.shapley_value[absent] == pytest.approx(0.0, abs=1e-12)
        assert res.attribution[absent] == pytest.approx(0.0, abs=1e-12)


def test_shapley_ranks_a_high_lift_low_volume_channel_above_a_flood() -> None:
    """The core scenario in miniature: email is rare but decisive, meta is a flood.

    meta appears in 40 of 50 journeys but converts at 20% alone; email appears
    in 20 and converts at 50% alone and 80% alongside meta. Last-touch-style
    volume logic favours meta; Shapley must favour email.

    Note both channels are given solo journeys. Without them, a channel that
    only ever co-occurs has an unobservable solo coalition value of zero and is
    structurally under-credited -- the baseline bias documented in
    :mod:`streamly.attribution.shapley`, not a property of the channel.
    """
    spec: list[tuple[list[str], bool]] = (
        [(["meta"], True)] * 6 + [(["meta"], False)] * 24          # meta solo: 20%
        + [(["email"], True)] * 5 + [(["email"], False)] * 5       # email solo: 50%
        + [(["meta", "email"], True)] * 8                          # together: 80%
        + [(["meta", "email"], False)] * 2
    )
    res = shapley_attribution(_repeat(spec, times=1))
    assert res.attribution["email"] > res.attribution["meta"]
    assert res.attribution["email"] > 0.5


# ---------------------------------------------------------------------------
# The Phase 3 gate, on the generated warehouse
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def warehouse_journeys() -> JourneySet:
    generator.generate()
    return build_journeys()


def test_ddm_recovers_truth_better_than_last_touch(warehouse_journeys: JourneySet) -> None:
    """THE Phase 3 gate: data-driven attribution beats the incumbent on MAE."""
    table = validate.attribution_matrix(warehouse_journeys)
    scores = validate.recovery_scores(table)

    last_touch_mae = scores.loc["last_touch", "mae"]
    assert scores.loc["shapley", "mae"] < last_touch_mae * 0.6, (
        f"Shapley MAE {scores.loc['shapley', 'mae']:.4f} must be materially below "
        f"last-touch {last_touch_mae:.4f}"
    )
    assert scores.loc["markov", "mae"] < last_touch_mae, "Markov must at least beat last-touch"
    assert scores.index[0] == "shapley", "Shapley should rank best on this DGP"


def test_shapley_corrects_the_meta_over_credit(warehouse_journeys: JourneySet) -> None:
    """The finance-facing claim: meta's inflated share shrinks toward truth."""
    table = validate.attribution_matrix(warehouse_journeys)
    truth = validate.load_ground_truth()
    for ch in ("meta", "email"):
        err_lt = abs(table.loc[ch, "last_touch"] - truth[ch])
        err_sh = abs(table.loc[ch, "shapley"] - truth[ch])
        assert err_sh < err_lt, f"Shapley should be closer to truth than last-touch on {ch}"
    assert table.loc["meta", "shapley"] < table.loc["meta", "last_touch"]
    assert table.loc["email", "shapley"] > table.loc["email", "last_touch"]


def test_every_model_share_column_is_a_distribution(warehouse_journeys: JourneySet) -> None:
    table = validate.attribution_matrix(warehouse_journeys)
    assert list(table.index) == list(CHANNELS)
    np.testing.assert_allclose(table.sum(axis=0).to_numpy(), 1.0, atol=1e-9)
    assert (table.values >= 0).all()


# ---------------------------------------------------------------------------
# ROI and reallocation
# ---------------------------------------------------------------------------
def test_credited_value_is_net_of_refunds(warehouse_journeys: JourneySet) -> None:
    js = warehouse_journeys
    _conv, rev = roi.credited_value(js, "last_touch")
    expected = float(js.revenue[js.converted & ~js.is_refunded].sum())
    assert rev.sum() == pytest.approx(expected, rel=1e-9)
    assert expected < float(js.revenue[js.converted].sum()), "refunds must reduce credited revenue"


def test_roi_table_reconciles_to_the_monthly_brief(warehouse_journeys: JourneySet) -> None:
    t = roi.roi_table(warehouse_journeys, "shapley")
    assert 450_000 < t["monthly_spend"].sum() < 550_000
    assert t["value_share"].sum() == pytest.approx(1.0)
    assert (t["cac"] > 0).all()


def test_reallocation_is_budget_neutral_and_respects_the_cap(
    warehouse_journeys: JourneySet,
) -> None:
    plan = roi.reallocation_plan(warehouse_journeys, "shapley", max_shift=0.30)
    t = plan.table
    assert t["delta"].sum() == pytest.approx(0.0, abs=1e-6), "plan must be budget-neutral"
    assert t["proposed_spend"].sum() == pytest.approx(t["current_spend"].sum(), rel=1e-9)
    assert (t["delta_pct"].abs() <= 0.30 + 1e-9).all(), "no channel may move beyond the cap"
    assert (t["proposed_spend"] > 0).all()


def test_reallocation_moves_budget_from_meta_toward_email(
    warehouse_journeys: JourneySet,
) -> None:
    plan = roi.reallocation_plan(warehouse_journeys, "shapley")
    assert plan.table.loc["meta", "delta"] < 0
    assert plan.table.loc["email", "delta"] > 0
    assert plan.expected_incremental_conversions > 0


def test_misallocation_versus_incumbent_is_material(warehouse_journeys: JourneySet) -> None:
    per_channel, total = roi.misallocation_vs_incumbent(warehouse_journeys)
    assert total > 50_000, "last-touch vs Shapley should steer a material share of budget"
    assert per_channel["difference"].sum() == pytest.approx(0.0, abs=1e-6)
