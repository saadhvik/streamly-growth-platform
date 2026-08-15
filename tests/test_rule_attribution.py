"""Phase 2 acceptance tests: journey construction + rule-based attribution.

The headline gate is conservation -- every rule must distribute exactly 100% of
credit per converting journey, so credited totals equal the conversion count and
credited revenue equals converting revenue. Everything downstream (ROI,
reallocation) is meaningless if credit leaks here.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# Route the warehouse to a locally-deletable dir (mirrors test_datagen).
os.environ.setdefault("STREAMLY_DATA_DIR", tempfile.mkdtemp(prefix="streamly_test_"))

from streamly.attribution import rules  # noqa: E402
from streamly.attribution.sessionize import (  # noqa: E402
    JourneySet,
    build_journeys,
    journeys_from_frame,
)
from streamly.config import ATTRIBUTION, CHANNELS  # noqa: E402
from streamly.datagen import generator  # noqa: E402

_CFG = ATTRIBUTION


# ---------------------------------------------------------------------------
# Hand-built fixtures: exact, known-answer cases
# ---------------------------------------------------------------------------
def _frame(rows: list[tuple[int, str, str, str | None]]) -> pd.DataFrame:
    """Build a touch frame from ``(user_id, channel, ts, convert_ts)`` tuples."""
    df = pd.DataFrame(rows, columns=["user_id", "channel", "ts", "convert_ts"])
    df["ts"] = pd.to_datetime(df["ts"])
    df["convert_ts"] = pd.to_datetime(df["convert_ts"])
    df["touch_id"] = np.arange(1, len(df) + 1)
    df["cost_attributed"] = 1.0
    df["revenue"] = np.where(df["convert_ts"].notna(), 96.0, 0.0)
    df["is_refunded"] = False
    df["converted"] = df["convert_ts"].notna()
    return df


@pytest.fixture(scope="module")
def toy() -> JourneySet:
    """Two converters (4-touch and 1-touch) and one non-converter."""
    return journeys_from_frame(_frame([
        (1, "google", "2026-06-01", "2026-06-11"),   # 10 days before conversion
        (1, "meta",   "2026-06-04", "2026-06-11"),   #  7 days
        (1, "tiktok", "2026-06-08", "2026-06-11"),   #  3 days
        (1, "email",  "2026-06-11", "2026-06-11"),   #  0 days
        (2, "meta",   "2026-06-02", "2026-06-05"),
        (3, "tiktok", "2026-06-02", None),           # non-converter
        (3, "meta",   "2026-06-03", None),
    ]))


def test_journeys_are_ordered_and_typed(toy: JourneySet) -> None:
    assert len(toy) == 3
    assert toy.n_conversions == 2
    assert toy.path(0) == ["google", "meta", "tiktok", "email"]
    assert toy.path(2) == ["tiktok", "meta"]
    # Non-converters have no conversion anchor, so no decay clock.
    assert toy.days_to_conversion(2).size == 0
    np.testing.assert_allclose(toy.days_to_conversion(0), [10.0, 7.0, 3.0, 0.0])


def test_lookback_window_and_post_conversion_touches_are_dropped() -> None:
    js = journeys_from_frame(_frame([
        (1, "tiktok", "2026-01-01", "2026-06-11"),   # far outside 45-day window
        (1, "google", "2026-06-01", "2026-06-11"),   # inside
        (1, "email",  "2026-06-20", "2026-06-11"),   # logged AFTER conversion
    ]))
    assert js.path(0) == ["google"], "window must exclude stale and post-conversion touches"


# ---------------------------------------------------------------------------
# Known-answer weights, per rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("first_touch", [1.0, 0.0, 0.0, 0.0]),
        ("last_touch", [0.0, 0.0, 0.0, 1.0]),
        ("linear", [0.25, 0.25, 0.25, 0.25]),
        ("position_based", [0.40, 0.10, 0.10, 0.40]),
    ],
)
def test_rule_weights_match_closed_form(toy: JourneySet, model: str, expected: list[float]) -> None:
    w = rules.journey_weights(model, toy.path(0), toy.days_to_conversion(0), _CFG)
    np.testing.assert_allclose(w, expected, atol=1e-12)


def test_time_decay_halves_at_the_half_life(toy: JourneySet) -> None:
    """Touches at 10 / 7 / 3 / 0 days with a 7-day half-life."""
    days = toy.days_to_conversion(0)
    w = rules.journey_weights("time_decay", toy.path(0), days, _CFG)
    raw = np.power(0.5, days / _CFG.half_life_days)
    np.testing.assert_allclose(w, raw / raw.sum(), rtol=1e-12)
    # The 7-day-old touch must carry exactly half the weight of the same-day one.
    assert w[1] == pytest.approx(w[3] * 0.5)


@pytest.mark.parametrize("n", [1, 2, 3, 8])
def test_position_based_degenerate_paths_still_total_one(n: int) -> None:
    path = ["google"] * n
    w = rules.journey_weights("position_based", path, np.zeros(n), _CFG)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    if n == 2:
        np.testing.assert_allclose(w, [0.5, 0.5])


def test_unknown_model_is_rejected(toy: JourneySet) -> None:
    with pytest.raises(ValueError, match="unknown rule model"):
        rules.journey_weights("markov", toy.path(0), toy.days_to_conversion(0), _CFG)


def test_repeated_channel_accumulates_credit() -> None:
    js = journeys_from_frame(_frame([
        (1, "meta",   "2026-06-01", "2026-06-05"),
        (1, "meta",   "2026-06-02", "2026-06-05"),
        (1, "google", "2026-06-03", "2026-06-05"),
    ]))
    conv, _ = rules.credit_by_channel(js, "linear", _CFG)
    assert conv["meta"] == pytest.approx(2 / 3)
    assert conv["google"] == pytest.approx(1 / 3)


def test_non_converters_receive_no_credit(toy: JourneySet) -> None:
    conv, _ = rules.credit_by_channel(toy, "last_touch", _CFG)
    # Only journeys 1 (email) and 2 (meta) convert; tiktok closes only the
    # non-converting journey and must earn nothing.
    assert conv["tiktok"] == pytest.approx(0.0)
    assert conv["email"] == pytest.approx(1.0)
    assert conv["meta"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Acceptance gate, on the full generated warehouse
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def warehouse_journeys() -> JourneySet:
    generator.generate()
    return build_journeys()


def test_credit_conserves_total_conversions(warehouse_journeys: JourneySet) -> None:
    """THE Phase 2 gate: every model credits exactly 100% of conversions."""
    js = warehouse_journeys
    for model in _CFG.rule_models:
        conv, rev = rules.credit_by_channel(js, model, _CFG)
        assert sum(conv.values()) == pytest.approx(js.n_conversions, rel=1e-9), model
        expected_rev = float(js.revenue[js.converted].sum())
        assert sum(rev.values()) == pytest.approx(expected_rev, rel=1e-9), model


def test_credit_table_shares_are_valid_distributions(warehouse_journeys: JourneySet) -> None:
    table = rules.credit_table(warehouse_journeys, _CFG)
    assert list(table.index) == list(CHANNELS)
    assert set(table.columns) == set(_CFG.rule_models)
    assert (table.values >= 0).all()
    np.testing.assert_allclose(table.sum(axis=0).to_numpy(), 1.0, atol=1e-9)


def test_each_rule_is_biased_toward_the_funnel_position_it_favours(
    warehouse_journeys: JourneySet,
) -> None:
    """No heuristic escapes bias -- but they fail in *different directions*.

    Each rule rewards whichever channel sits where the rule happens to look:

    * last-touch hands the bulk of the credit to meta, the late-funnel channel
    * first-touch does the same for tiktok, the awareness channel
    * neither channel earned it -- meta's true importance is 14%, tiktok's 10%

    That divergence is the useful finding. It means "pick a fairer rule" is not
    a fix: swapping last-touch for first-touch does not reduce the error, it
    just moves the over-credit to a different channel. Only a method that
    observes a counterfactual escapes it.
    """
    import json

    from streamly.config import GROUND_TRUTH_DIR

    truth = json.load(open(GROUND_TRUTH_DIR / "ground_truth.json", encoding="utf-8"))["channel_importance_true"]
    table = rules.credit_table(warehouse_journeys, _CFG)

    # Position-driven over-crediting, in opposite directions.
    assert table.loc["meta", "last_touch"] > 3 * truth["meta"], "last-touch inflates the closer"
    assert table.loc["tiktok", "first_touch"] > 3 * truth["tiktok"], "first-touch inflates the opener"
    # And each rule starves the channel at the far end of the funnel from it.
    assert table.loc["meta", "first_touch"] < truth["meta"]
    assert table.loc["tiktok", "last_touch"] < truth["tiktok"]

    # Email is the channel every rule misses: real lift, never at either edge.
    for model in _CFG.rule_models:
        assert table.loc["email", model] < truth["email"] - 0.05, f"{model} under-credits email"
