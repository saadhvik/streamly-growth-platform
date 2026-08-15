"""Analytical marts over the raw warehouse tables.

The five raw tables are shaped for the generator, not for a person with a SQL
prompt. These three views are the modelling layer that makes the warehouse
usable directly:

* ``mart_user_journeys``      -- one row per user, with the ordered channel path
* ``mart_channel_roi``        -- one row per channel, with spend and rule credit
* ``mart_experiment_metrics`` -- one row per experiment x variant

Why views rather than dbt
-------------------------
The original design left this as "dbt or plain views". Views win here: dbt
would add a dependency, a profiles file and a second execution model to a
project whose transformations are three SELECTs, and CI would have to install
and run it to prove the marts still build. As views they are created by the
same ``generate()`` call that loads the data and are covered by the existing
test run. If the transformation layer ever grows past a handful of models --
incremental builds, snapshots, cross-project lineage -- dbt earns its place and
these SELECTs port over unchanged.

Portability
-----------
Deliberately ANSI: window functions and ``STRING_AGG`` rather than DuckDB's
``arg_min``/``list`` shortcuts, so the claim that lifting to BigQuery is a
connection change rather than a rewrite is actually true of this SQL.

The journey window matches :mod:`streamly.attribution.sessionize` exactly --
same 45-day lookback, same exclusion of post-conversion touches. That is not
cosmetic: the test suite asserts the last-touch share computed here in SQL
equals the one computed by the Python engine, so a drift in either
implementation fails the build.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from streamly import warehouse
from streamly.config import ATTRIBUTION, WAREHOUSE_PATH

MART_VIEWS: tuple[str, ...] = (
    "mart_user_journeys",
    "mart_channel_roi",
    "mart_experiment_metrics",
)


def _journey_sql(lookback_days: int) -> str:
    """Touches inside the attribution window, joined to their conversion."""
    return f"""
    WITH conv AS (
        SELECT user_id,
               MIN(convert_ts)      AS convert_ts,
               SUM(revenue)         AS revenue,
               BOOL_OR(is_refunded) AS is_refunded
        FROM conversions
        GROUP BY user_id
    ),
    windowed AS (
        SELECT t.user_id,
               t.touch_id,
               t.channel,
               t.ts,
               c.convert_ts,
               COALESCE(c.revenue, 0.0)       AS revenue,
               COALESCE(c.is_refunded, FALSE) AS is_refunded,
               (c.user_id IS NOT NULL)        AS converted
        FROM touchpoints t
        LEFT JOIN conv c ON c.user_id = t.user_id
        -- Same window as the Python loader: a touch after the conversion cannot
        -- have caused it, and anything older than the lookback is out of scope.
        WHERE c.convert_ts IS NULL
           OR (t.ts <= c.convert_ts
               AND DATE_DIFF('day', t.ts, c.convert_ts) <= {lookback_days})
    )
    """


def mart_ddl(lookback_days: int = ATTRIBUTION.lookback_days) -> str:
    """DDL for all three marts."""
    journeys = _journey_sql(lookback_days)
    return f"""
CREATE OR REPLACE VIEW mart_user_journeys AS
{journeys}
, ranked AS (
    SELECT user_id, channel, ts, convert_ts, revenue, is_refunded, converted,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ts, touch_id) AS pos_asc,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ts DESC, touch_id DESC) AS pos_desc
    FROM windowed
)
SELECT
    user_id,
    STRING_AGG(channel, ' > ' ORDER BY pos_asc)                    AS path,
    COUNT(*)                                                       AS touch_count,
    COUNT(DISTINCT channel)                                        AS distinct_channels,
    MAX(CASE WHEN pos_asc  = 1 THEN channel END)                   AS first_channel,
    MAX(CASE WHEN pos_desc = 1 THEN channel END)                   AS last_channel,
    MIN(ts)                                                        AS first_touch_ts,
    MAX(ts)                                                        AS last_touch_ts,
    ANY_VALUE(converted)                                           AS converted,
    ANY_VALUE(convert_ts)                                          AS convert_ts,
    ANY_VALUE(revenue)                                             AS revenue,
    ANY_VALUE(is_refunded)                                         AS is_refunded,
    -- Net of refunds, matching the ROI convention in streamly.attribution.roi.
    CASE WHEN ANY_VALUE(is_refunded) THEN 0.0 ELSE ANY_VALUE(revenue) END AS net_revenue
FROM ranked
GROUP BY user_id;

CREATE OR REPLACE VIEW mart_channel_roi AS
WITH spend_by_channel AS (
    SELECT channel,
           SUM(spend)              AS total_spend,
           COUNT(DISTINCT date)    AS days,
           SUM(impressions)        AS impressions,
           SUM(clicks)             AS clicks
    FROM spend
    GROUP BY channel
),
credit AS (
    SELECT
        channel,
        COUNT(*)                                                    AS touches,
        SUM(CASE WHEN converted AND is_last  THEN 1 ELSE 0 END)     AS last_touch_conversions,
        SUM(CASE WHEN converted AND is_first THEN 1 ELSE 0 END)     AS first_touch_conversions,
        SUM(CASE WHEN converted THEN 1.0 / touches_in_journey ELSE 0 END) AS linear_conversions
    FROM (
        SELECT w.channel, w.converted,
               ROW_NUMBER() OVER (PARTITION BY w.user_id ORDER BY w.ts DESC, w.touch_id DESC) = 1 AS is_last,
               ROW_NUMBER() OVER (PARTITION BY w.user_id ORDER BY w.ts, w.touch_id) = 1          AS is_first,
               COUNT(*) OVER (PARTITION BY w.user_id)                                            AS touches_in_journey
        FROM ({journeys} SELECT * FROM windowed) w
    ) x
    GROUP BY channel
)
SELECT
    s.channel,
    s.total_spend,
    s.total_spend / (s.days / 30.0)                       AS monthly_spend,
    s.impressions,
    s.clicks,
    c.touches,
    c.last_touch_conversions,
    c.first_touch_conversions,
    c.linear_conversions,
    -- CAC under each rule. The data-driven credit (Markov, Shapley) is not
    -- expressible as a single SQL aggregate and lives in the Python engine.
    CASE WHEN c.last_touch_conversions > 0
         THEN s.total_spend / c.last_touch_conversions END AS cac_last_touch,
    CASE WHEN c.linear_conversions > 0
         THEN s.total_spend / c.linear_conversions END     AS cac_linear
FROM spend_by_channel s
JOIN credit c ON c.channel = s.channel;

CREATE OR REPLACE VIEW mart_experiment_metrics AS
SELECT
    experiment_id,
    variant,
    COUNT(*)                                             AS units,
    SUM(primary_metric)                                  AS conversions,
    AVG(primary_metric)                                  AS conversion_rate,
    -- Guardrails are measured over ALL assigned users, not just converters --
    -- see docs/metric_definitions.md on why the denominator matters.
    AVG(CASE WHEN guardrail_refund THEN 1.0 ELSE 0.0 END) AS refund_rate,
    AVG(guardrail_latency_ms)                            AS mean_latency_ms,
    AVG(pre_covariate)                                   AS mean_pre_covariate,
    MIN(assign_ts)                                       AS first_assigned,
    MAX(assign_ts)                                       AS last_assigned
FROM experiment_assignment
GROUP BY experiment_id, variant;
"""


def build_marts(
    con: duckdb.DuckDBPyConnection, lookback_days: int = ATTRIBUTION.lookback_days
) -> None:
    """Create or replace every mart view."""
    con.execute(mart_ddl(lookback_days))


def build(path: Path | str = WAREHOUSE_PATH) -> dict[str, int]:
    """Build the marts against the warehouse and return each view's row count."""
    con = warehouse.connect(path)
    try:
        build_marts(con)
        counts = {}
        for view in MART_VIEWS:
            row = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()
            counts[view] = int(row[0]) if row else 0
    finally:
        con.close()
    return counts


def _main() -> None:
    counts = build()
    print("Mart views built:")
    for name, n in counts.items():
        print(f"  {name:26s} {n:>8,} rows")

    con = warehouse.connect(read_only=True)
    try:
        print("\nTop converting paths (mart_user_journeys):")
        rows = con.execute(
            "SELECT path, COUNT(*) n FROM mart_user_journeys "
            "WHERE converted GROUP BY path ORDER BY n DESC LIMIT 5"
        ).fetchall()
        for path, n in rows:
            print(f"  {n:>5,}  {path}")

        print("\nChannel ROI (mart_channel_roi):")
        for r in con.execute(
            "SELECT channel, ROUND(monthly_spend) spend, last_touch_conversions, "
            "ROUND(cac_last_touch, 2) cac FROM mart_channel_roi ORDER BY spend DESC"
        ).fetchall():
            print(f"  {r[0]:10s} spend={r[1]:>9,.0f}  last-touch conv={r[2]:>6,}  CAC={r[3]}")
    finally:
        con.close()


if __name__ == "__main__":
    _main()
