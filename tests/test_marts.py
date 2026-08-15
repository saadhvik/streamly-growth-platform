"""Tests for the analytical mart layer.

The headline test is :func:`test_sql_last_touch_matches_the_python_engine`: the
same quantity is computed twice, once in SQL over the raw tables and once by the
Python attribution engine, and the two must agree exactly. Two independent
implementations of the same definition is the strongest check available that
neither has drifted -- in particular that the SQL applies the same attribution
window (45-day lookback, no post-conversion touches) as the journey loader.
"""
from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault("STREAMLY_DATA_DIR", tempfile.mkdtemp(prefix="streamly_test_"))

from streamly import marts, warehouse                            # noqa: E402
from streamly.attribution import rules                           # noqa: E402
from streamly.attribution.sessionize import build_journeys       # noqa: E402
from streamly.config import CHANNELS, DATAGEN                    # noqa: E402
from streamly.datagen import generator                           # noqa: E402


@pytest.fixture(scope="module")
def built() -> None:
    generator.generate()


def _query(sql: str):
    con = warehouse.connect(read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def test_generation_builds_every_mart(built: None) -> None:
    """Marts are views, so they can never be stale relative to the raw tables."""
    names = {r[0] for r in _query(
        "SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'"
    )}
    for view in marts.MART_VIEWS:
        assert view in names, f"{view} was not created by generate()"


def test_user_journeys_covers_every_user_exactly_once(built: None) -> None:
    rows = _query("SELECT COUNT(*), COUNT(DISTINCT user_id) FROM mart_user_journeys")
    total, distinct = rows[0]
    assert total == distinct, "one row per user"
    assert total == DATAGEN.n_users


def test_journey_path_agrees_with_its_own_first_and_last_columns(built: None) -> None:
    """The denormalized edge columns must match the ordered path string."""
    bad = _query("""
        SELECT COUNT(*) FROM mart_user_journeys
        WHERE first_channel <> SPLIT_PART(path, ' > ', 1)
           OR last_channel  <> SPLIT_PART(path, ' > ', touch_count)
    """)[0][0]
    assert bad == 0


def test_touch_count_matches_the_path_length(built: None) -> None:
    """Separator count + 1 must equal the touch count.

    The parentheses matter: without them the division binds tighter than the
    subtraction and the check silently compares nonsense.
    """
    bad = _query("""
        SELECT COUNT(*) FROM mart_user_journeys
        WHERE touch_count
              <> ((LENGTH(path) - LENGTH(REPLACE(path, ' > ', ''))) / 3) + 1
    """)[0][0]
    assert bad == 0


def test_net_revenue_zeroes_refunded_conversions(built: None) -> None:
    leaked = _query(
        "SELECT COUNT(*) FROM mart_user_journeys WHERE is_refunded AND net_revenue <> 0"
    )[0][0]
    assert leaked == 0
    gross, net = _query(
        "SELECT SUM(revenue), SUM(net_revenue) FROM mart_user_journeys"
    )[0]
    assert net < gross, "refunds must reduce net revenue"


def test_sql_last_touch_matches_the_python_engine(built: None) -> None:
    """THE mart gate: two independent implementations, one definition.

    If the SQL window drifts from the Python loader's -- a different lookback,
    or forgetting to exclude post-conversion touches -- these diverge.
    """
    sql = dict(_query("SELECT channel, last_touch_conversions FROM mart_channel_roi"))
    python_credit, _rev = rules.credit_by_channel(build_journeys(), "last_touch")

    assert set(sql) == set(CHANNELS)
    for channel in CHANNELS:
        assert sql[channel] == pytest.approx(python_credit[channel], abs=1e-6), channel


def test_sql_linear_credit_also_matches_python(built: None) -> None:
    """Linear splits fractionally, so this catches window *and* arithmetic drift."""
    sql = dict(_query("SELECT channel, linear_conversions FROM mart_channel_roi"))
    python_credit, _rev = rules.credit_by_channel(build_journeys(), "linear")
    for channel in CHANNELS:
        assert sql[channel] == pytest.approx(python_credit[channel], abs=1e-6), channel


def test_channel_roi_conserves_total_conversions(built: None) -> None:
    """Every rule credits exactly the conversion count -- the Phase 2 invariant."""
    total = _query("SELECT COUNT(*) FROM conversions")[0][0]
    last, first, linear = _query(
        "SELECT SUM(last_touch_conversions), SUM(first_touch_conversions), "
        "SUM(linear_conversions) FROM mart_channel_roi"
    )[0]
    assert last == total
    assert first == total
    assert linear == pytest.approx(total, abs=1e-6)


def test_channel_roi_spend_reconciles_to_the_brief(built: None) -> None:
    monthly = _query("SELECT SUM(monthly_spend) FROM mart_channel_roi")[0][0]
    assert 450_000 < monthly < 550_000


def test_experiment_metrics_matches_the_raw_table(built: None) -> None:
    rows = _query("""
        SELECT variant, units, conversions, conversion_rate
        FROM mart_experiment_metrics ORDER BY variant
    """)
    assert [r[0] for r in rows] == ["control", "treatment"]
    assert sum(r[1] for r in rows) == DATAGEN.n_users

    raw = dict(_query("""
        SELECT variant, AVG(primary_metric) FROM experiment_assignment
        GROUP BY variant
    """))
    for variant, _units, _conv, rate in rows:
        assert rate == pytest.approx(raw[variant], rel=1e-12)


def test_experiment_refund_rate_uses_all_assigned_users(built: None) -> None:
    """The denominator fix, enforced in SQL as well as in Python."""
    rows = dict(_query(
        "SELECT variant, refund_rate FROM mart_experiment_metrics"
    ))
    manual = dict(_query("""
        SELECT variant, SUM(CASE WHEN guardrail_refund THEN 1.0 ELSE 0 END) / COUNT(*)
        FROM experiment_assignment GROUP BY variant
    """))
    for variant, rate in rows.items():
        assert rate == pytest.approx(manual[variant], rel=1e-12)
    # Sanity: measured over everyone, the rate is far below the converter-only rate.
    assert all(r < 0.02 for r in rows.values())


def test_marts_are_idempotent(built: None) -> None:
    """CREATE OR REPLACE, so rebuilding is safe and changes nothing."""
    before = _query("SELECT COUNT(*) FROM mart_user_journeys")[0][0]
    con = warehouse.connect()
    try:
        marts.build_marts(con)
        marts.build_marts(con)
    finally:
        con.close()
    assert _query("SELECT COUNT(*) FROM mart_user_journeys")[0][0] == before


def test_mart_sql_avoids_engine_specific_shortcuts() -> None:
    """The portability claim in the module docstring, made checkable.

    DuckDB-only constructs would silently break the "lift to BigQuery is a
    connection change" story.
    """
    sql = marts.mart_ddl().lower()
    for banned in ("arg_min(", "arg_max(", "list(", "::"):
        assert banned not in sql, f"non-portable construct: {banned}"
