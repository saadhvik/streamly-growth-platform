"""DuckDB warehouse: connection helper and schema DDL.

Single source of truth for the physical schema described in Phase 0. All
generators and marts target these tables.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from streamly.config import WAREHOUSE_PATH

# Raw schema -- five core tables. Kept ANSI-portable so a lift to BigQuery is a
# connection change, not a rewrite.
SCHEMA_DDL: str = """
CREATE TABLE IF NOT EXISTS users (
    user_id             BIGINT PRIMARY KEY,
    signup_ts           TIMESTAMP,
    platform            VARCHAR,          -- ios | android | web
    country             VARCHAR,
    acquisition_cohort  VARCHAR           -- YYYY-MM
);

CREATE TABLE IF NOT EXISTS touchpoints (
    touch_id            BIGINT PRIMARY KEY,
    user_id             BIGINT,
    channel             VARCHAR,          -- google|meta|tiktok|email|referral
    campaign            VARCHAR,
    ts                  TIMESTAMP,
    cost_attributed     DOUBLE,           -- per-touch cost allocation
    position_in_journey INTEGER
);

CREATE TABLE IF NOT EXISTS conversions (
    conversion_id       BIGINT PRIMARY KEY,
    user_id             BIGINT,
    convert_ts          TIMESTAMP,
    plan_type           VARCHAR,          -- monthly | annual
    revenue             DOUBLE,
    is_refunded         BOOLEAN,
    refund_ts           TIMESTAMP
);

CREATE TABLE IF NOT EXISTS spend (
    date                DATE,
    channel             VARCHAR,
    campaign            VARCHAR,
    spend               DOUBLE,
    impressions         BIGINT,
    clicks              BIGINT
);

CREATE TABLE IF NOT EXISTS experiment_assignment (
    user_id             BIGINT,
    experiment_id       VARCHAR,
    variant             VARCHAR,          -- control | treatment
    assign_ts           TIMESTAMP,
    bucketing_hash      DOUBLE,           -- deterministic [0,1) hash
    pre_covariate       DOUBLE,           -- pre-period value (CUPED)
    primary_metric      DOUBLE,           -- LTV proxy / converted flag
    guardrail_refund    BOOLEAN,
    guardrail_latency_ms DOUBLE
);
"""

RAW_TABLES: tuple[str, ...] = (
    "users",
    "touchpoints",
    "conversions",
    "spend",
    "experiment_assignment",
)


def connect(path: Path | str = WAREHOUSE_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open (and create parent dir for) the DuckDB warehouse."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(p), read_only=read_only)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create all raw tables if absent."""
    con.execute(SCHEMA_DDL)


def reset_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Drop and recreate raw tables -- used for clean regeneration."""
    for t in RAW_TABLES:
        con.execute(f"DROP TABLE IF EXISTS {t}")
    con.execute(SCHEMA_DDL)
