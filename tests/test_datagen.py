"""Phase 1 acceptance tests: schema integrity + ground-truth recording."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

# Route the warehouse to a locally-deletable dir for the test session.
os.environ.setdefault("STREAMLY_DATA_DIR", tempfile.mkdtemp(prefix="streamly_test_"))

from streamly import warehouse  # noqa: E402
from streamly.config import CHANNELS, GROUND_TRUTH_DIR  # noqa: E402
from streamly.datagen import generator  # noqa: E402


@pytest.fixture(scope="module")
def counts() -> dict[str, int]:
    return generator.generate()


def test_all_tables_populated(counts: dict[str, int]) -> None:
    for table in warehouse.RAW_TABLES:
        assert counts[table] > 0, f"{table} is empty"
    assert counts["users"] == 60_000


def test_spend_reconciles_to_brief(counts: dict[str, int]) -> None:
    con = warehouse.connect(read_only=True)
    total = con.execute("SELECT SUM(spend) FROM spend").fetchone()[0]
    con.close()
    monthly = total / 3.0  # 90 days of history
    assert 450_000 < monthly < 550_000, f"monthly spend {monthly:,.0f} off-brief"


def test_conversion_rate_is_realistic(counts: dict[str, int]) -> None:
    con = warehouse.connect(read_only=True)
    cr = con.execute(
        "SELECT COUNT(*)*1.0/(SELECT COUNT(*) FROM users) FROM conversions"
    ).fetchone()[0]
    con.close()
    assert 0.03 < cr < 0.20, f"conversion rate {cr:.1%} unrealistic for freemium->paid"


def test_ground_truth_recorded(counts: dict[str, int]) -> None:
    gt = json.load(open(GROUND_TRUTH_DIR / "ground_truth.json", encoding="utf-8"))
    imp = gt["channel_importance_true"]
    assert set(imp) == set(CHANNELS)
    assert abs(sum(imp.values()) - 1.0) < 1e-9, "true importance must be normalized"


def test_last_touch_is_biased_vs_truth(counts: dict[str, int]) -> None:
    """The whole validation premise: last-touch must MISattribute vs ground truth.

    Meta (high exposure, low true importance) should be over-credited by
    last-touch relative to its true weight. If this fails, Phase 3 has nothing
    to prove.
    """
    con = warehouse.connect(read_only=True)
    rows = con.execute(
        """
        WITH lt AS (
          SELECT t.user_id, t.channel,
                 ROW_NUMBER() OVER (PARTITION BY t.user_id ORDER BY t.ts DESC) rn
          FROM touchpoints t JOIN conversions c USING(user_id)
        )
        SELECT channel, COUNT(*)*1.0/SUM(COUNT(*)) OVER () AS share
        FROM lt WHERE rn = 1 GROUP BY channel
        """
    ).fetchall()
    con.close()
    lt_share = {ch: s for ch, s in rows}
    gt = json.load(open(GROUND_TRUTH_DIR / "ground_truth.json", encoding="utf-8"))
    true_imp = gt["channel_importance_true"]
    assert lt_share["meta"] > true_imp["meta"] + 0.05, "Meta should be over-credited"
    assert lt_share["email"] < true_imp["email"] - 0.05, "Email should be under-credited"
