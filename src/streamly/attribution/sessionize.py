"""Touchpoint log -> ordered user journeys.

This is the single entry point every attribution model reads from: rule-based
(Phase 2), Markov removal-effect and Shapley (Phase 3) all consume the same
:class:`JourneySet`, so any difference in their output is a difference in
*method*, never in data preparation.

Design notes
------------
* **Non-converters are kept.** Rule models only need converting paths, but
  Markov's removal effect and Shapley's coalition values are both defined
  against the full population -- dropping null paths inflates every channel's
  measured contribution. Building them once here avoids two divergent loaders.
* **The lookback window is applied at build time** so every downstream model
  sees an identical, auditable universe of touches.
* **Ordering is by timestamp, ascending**, with ``touch_id`` as a deterministic
  tie-breaker so runs are byte-reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from streamly import warehouse
from streamly.config import ATTRIBUTION, WAREHOUSE_PATH, AttributionConfig

# One row per touch, already joined to the (at most one) conversion per user.
# The conversion side is aggregated defensively so a duplicate conversion row
# can never fan out the touch log.
_JOURNEY_SQL = """
WITH conv AS (
    SELECT user_id,
           MIN(convert_ts)                    AS convert_ts,
           SUM(revenue)                       AS revenue,
           BOOL_OR(is_refunded)               AS is_refunded
    FROM conversions
    GROUP BY user_id
)
SELECT t.user_id,
       t.touch_id,
       t.channel,
       t.ts,
       t.cost_attributed,
       c.convert_ts,
       COALESCE(c.revenue, 0.0)               AS revenue,
       COALESCE(c.is_refunded, FALSE)         AS is_refunded,
       (c.user_id IS NOT NULL)                AS converted
FROM touchpoints t
LEFT JOIN conv c USING (user_id)
ORDER BY t.user_id, t.ts, t.touch_id
"""


@dataclass(frozen=True)
class JourneySet:
    """Ordered marketing journeys for the whole user population.

    Stored column-wise (parallel arrays + an offset index) rather than as a list
    of per-user objects: at 60k users / 270k touches the flat layout keeps the
    Shapley and Markov passes vectorizable and cheap to slice.

    Attributes
    ----------
    channels:
        Flat array of channel names for every retained touch, journey-ordered.
    ts:
        Flat array of touch timestamps, parallel to ``channels``.
    offsets:
        Length ``n_users + 1``; journey ``i`` occupies ``channels[offsets[i]:
        offsets[i + 1]]``.
    user_ids, converted, convert_ts, revenue, is_refunded:
        Per-journey arrays of length ``n_users``.
    """

    user_ids: np.ndarray
    channels: np.ndarray
    ts: np.ndarray
    offsets: np.ndarray
    converted: np.ndarray
    convert_ts: np.ndarray
    revenue: np.ndarray
    is_refunded: np.ndarray

    def __len__(self) -> int:
        return int(self.user_ids.shape[0])

    @property
    def n_conversions(self) -> int:
        return int(self.converted.sum())

    def path(self, i: int) -> list[str]:
        """Ordered channel path for journey ``i``."""
        return list(self.channels[self.offsets[i]:self.offsets[i + 1]])

    def days_to_conversion(self, i: int) -> np.ndarray:
        """Days between each touch in journey ``i`` and that user's conversion.

        Returns an empty array for non-converters (time-decay is undefined
        without a conversion anchor).
        """
        if not self.converted[i]:
            return np.empty(0, dtype=float)
        touches = self.ts[self.offsets[i]:self.offsets[i + 1]]
        delta = self.convert_ts[i] - touches
        return delta / np.timedelta64(1, "D")

    def paths(self, converters_only: bool = False) -> list[list[str]]:
        """Materialize journeys as a list of channel paths."""
        idx = range(len(self))
        if converters_only:
            idx = (i for i in idx if self.converted[i])  # type: ignore[assignment]
        return [self.path(i) for i in idx]


def build_journeys(
    cfg: AttributionConfig = ATTRIBUTION,
    path: Path | str = WAREHOUSE_PATH,
) -> JourneySet:
    """Read the warehouse and assemble journeys under the lookback window.

    Touches more than ``cfg.lookback_days`` before a user's conversion are
    dropped. Non-converters keep their full path (there is no anchor to measure
    the window from), which is the standard convention.
    """
    con = warehouse.connect(path, read_only=True)
    try:
        df = con.execute(_JOURNEY_SQL).fetch_df()
    finally:
        con.close()
    return journeys_from_frame(df, cfg)


def journeys_from_frame(df: pd.DataFrame, cfg: AttributionConfig = ATTRIBUTION) -> JourneySet:
    """Build a :class:`JourneySet` from an already-loaded touch frame.

    Split out from :func:`build_journeys` so tests can drive the exact same
    assembly logic from a hand-written frame without a warehouse.
    """
    df = df.sort_values(["user_id", "ts", "touch_id"], kind="mergesort")

    # Apply the lookback window (converters only -- see docstring).
    age_days = (df["convert_ts"] - df["ts"]) / pd.Timedelta(days=1)
    in_window = age_days.isna() | (age_days <= cfg.lookback_days)
    # A touch logged *after* the conversion cannot have caused it.
    in_window &= age_days.isna() | (age_days >= 0)
    df = df.loc[in_window]

    # Per-journey (user-level) attributes, taken from the first row of each user.
    first = df.groupby("user_id", sort=True).first()
    counts = df.groupby("user_id", sort=True).size().to_numpy()

    offsets = np.zeros(counts.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    return JourneySet(
        user_ids=first.index.to_numpy(dtype=np.int64),
        channels=df["channel"].to_numpy(dtype=object),
        ts=df["ts"].to_numpy(dtype="datetime64[ns]"),
        offsets=offsets,
        converted=first["converted"].to_numpy(dtype=bool),
        convert_ts=first["convert_ts"].to_numpy(dtype="datetime64[ns]"),
        revenue=first["revenue"].to_numpy(dtype=float),
        is_refunded=first["is_refunded"].to_numpy(dtype=bool),
    )


def path_summary(js: JourneySet, top_n: int = 10) -> pd.DataFrame:
    """Most common converting paths -- a sanity read on journey construction."""
    rows: dict[str, list[float]] = {}
    for i in range(len(js)):
        if not js.converted[i]:
            continue
        key = " > ".join(js.path(i))
        rows.setdefault(key, [0.0, 0.0])
        rows[key][0] += 1
        rows[key][1] += js.revenue[i]
    out = pd.DataFrame(
        [(k, int(v[0]), v[1]) for k, v in rows.items()],
        columns=["path", "conversions", "revenue"],
    )
    out["path_length"] = out["path"].str.count(">") + 1
    return out.sort_values("conversions", ascending=False).head(top_n).reset_index(drop=True)
