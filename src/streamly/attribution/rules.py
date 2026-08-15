"""Rule-based (heuristic) attribution models.

Five industry-standard rules, each expressed as the same primitive: given one
converting journey, return a weight per touch that sums to 1.0. Everything else
-- per-channel aggregation, revenue credit, the comparison table -- is shared,
so the models differ only in the line that matters.

    first_touch      100% to the opening touch          (demand creation view)
    last_touch       100% to the closing touch          (the incumbent; the $500K problem)
    linear           equal split across all touches     (naive fairness)
    time_decay       exponential half-life to conversion(recency-weighted)
    position_based   40 / 20 / 40 U-shape               (first + last matter most)

**Conservation invariant.** Every model distributes exactly 1.0 credit per
converting journey, so total credit always equals the conversion count and
revenue credit always equals total converting revenue. That is the Phase 2
acceptance gate and it is asserted in :func:`credit_table` itself, not only in
tests -- a rule that silently leaks credit would corrupt every ROI number
downstream.

These are all *heuristics*: none observes the counterfactual, so none can be
right except by luck. Phase 3 measures exactly how wrong they are against the
recorded ground truth.

Run:  PYTHONPATH=src python -m streamly.attribution.rules
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from streamly.attribution.sessionize import JourneySet, build_journeys
from streamly.config import ATTRIBUTION, CHANNELS, AttributionConfig

# A weighting rule: (path, days_before_conversion, cfg) -> weights summing to 1.
WeightFn = Callable[[list[str], np.ndarray, AttributionConfig], np.ndarray]


def _first_touch(path: list[str], days: np.ndarray, cfg: AttributionConfig) -> np.ndarray:
    w = np.zeros(len(path))
    w[0] = 1.0
    return w


def _last_touch(path: list[str], days: np.ndarray, cfg: AttributionConfig) -> np.ndarray:
    w = np.zeros(len(path))
    w[-1] = 1.0
    return w


def _linear(path: list[str], days: np.ndarray, cfg: AttributionConfig) -> np.ndarray:
    return np.full(len(path), 1.0 / len(path))


def _time_decay(path: list[str], days: np.ndarray, cfg: AttributionConfig) -> np.ndarray:
    """Exponential decay: a touch ``half_life_days`` older gets half the credit.

    Falls back to a linear split if the conversion anchor is missing (which
    should not happen for converters, but keeps the rule total-preserving
    rather than raising mid-aggregation).
    """
    if days.shape[0] != len(path):
        return _linear(path, days, cfg)
    w = np.power(0.5, np.maximum(days, 0.0) / cfg.half_life_days)
    total = w.sum()
    if total <= 0:  # every touch decayed to underflow -- degrade to linear
        return _linear(path, days, cfg)
    return w / total


def _position_based(path: list[str], days: np.ndarray, cfg: AttributionConfig) -> np.ndarray:
    """U-shaped 40/20/40.

    Degenerate paths renormalize rather than drop credit: a single touch takes
    100%, and a two-touch path splits the first/last weights in proportion
    (50/50 at the default 40/40), keeping the sum at exactly 1.0.
    """
    n = len(path)
    if n == 1:
        return np.array([1.0])
    if n == 2:
        edge = cfg.position_first + cfg.position_last
        return np.array([cfg.position_first / edge, cfg.position_last / edge])
    w = np.full(n, cfg.position_middle / (n - 2))
    w[0] = cfg.position_first
    w[-1] = cfg.position_last
    return w


RULES: dict[str, WeightFn] = {
    "first_touch": _first_touch,
    "last_touch": _last_touch,
    "linear": _linear,
    "time_decay": _time_decay,
    "position_based": _position_based,
}


def journey_weights(
    model: str,
    path: list[str],
    days: np.ndarray,
    cfg: AttributionConfig = ATTRIBUTION,
) -> np.ndarray:
    """Per-touch credit weights for one journey under ``model``."""
    try:
        fn = RULES[model]
    except KeyError:
        raise ValueError(f"unknown rule model {model!r}; expected one of {sorted(RULES)}") from None
    if not path:
        raise ValueError("cannot attribute an empty journey")
    return fn(path, days, cfg)


def credit_by_channel(
    js: JourneySet,
    model: str,
    cfg: AttributionConfig = ATTRIBUTION,
    channels: tuple[str, ...] = CHANNELS,
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate one model's credit over all converting journeys.

    Returns ``(conversion_credit, revenue_credit)`` keyed by channel. Repeated
    touches on the same channel within a journey accumulate, which is the
    intended behaviour: a channel that appears three times in a linear journey
    earns three shares.
    """
    idx = {c: k for k, c in enumerate(channels)}
    conv = np.zeros(len(channels))
    rev = np.zeros(len(channels))

    for i in range(len(js)):
        if not js.converted[i]:
            continue  # heuristic rules credit converting paths only
        path = js.path(i)
        if not path:
            continue
        w = journey_weights(model, path, js.days_to_conversion(i), cfg)
        revenue = float(js.revenue[i])
        for ch, wt in zip(path, w):
            k = idx[ch]
            conv[k] += wt
            rev[k] += wt * revenue

    return (
        {c: float(v) for c, v in zip(channels, conv)},
        {c: float(v) for c, v in zip(channels, rev)},
    )


def credit_table(
    js: JourneySet,
    cfg: AttributionConfig = ATTRIBUTION,
    models: tuple[str, ...] | None = None,
    channels: tuple[str, ...] = CHANNELS,
) -> pd.DataFrame:
    """Comparison table: credited conversion **share** per channel per model.

    Shares (not raw counts) are the comparable unit -- every model credits the
    same total, so only the split differs, and shares line up directly against
    the ground-truth importance vector in Phase 3.

    Raises
    ------
    AssertionError
        If any model fails the conservation invariant (credit must total the
        conversion count). This is the Phase 2 acceptance gate, enforced at the
        point of computation.
    """
    models = models or cfg.rule_models
    total_conversions = float(
        sum(1 for i in range(len(js)) if js.converted[i] and js.offsets[i + 1] > js.offsets[i])
    )

    out = pd.DataFrame(index=pd.Index(list(channels), name="channel"))
    for m in models:
        conv, _rev = credit_by_channel(js, m, cfg, channels)
        credited = sum(conv.values())
        assert abs(credited - total_conversions) < 1e-6, (
            f"{m} leaked credit: distributed {credited:,.4f} over "
            f"{total_conversions:,.0f} conversions"
        )
        out[m] = [conv[c] / total_conversions for c in channels]
    return out


def revenue_credit_table(
    js: JourneySet,
    cfg: AttributionConfig = ATTRIBUTION,
    models: tuple[str, ...] | None = None,
    channels: tuple[str, ...] = CHANNELS,
) -> pd.DataFrame:
    """Credited revenue (absolute dollars) per channel per model.

    Feeds the Phase 3 ROI and reallocation work, where the dollar figure -- not
    the share -- is what finance argues about.
    """
    models = models or cfg.rule_models
    out = pd.DataFrame(index=pd.Index(list(channels), name="channel"))
    for m in models:
        _conv, rev = credit_by_channel(js, m, cfg, channels)
        out[m] = [rev[c] for c in channels]
    return out


def _main() -> None:
    js = build_journeys()
    table = credit_table(js)

    print(f"journeys: {len(js):,}   conversions: {js.n_conversions:,}")
    print("\nCredited conversion share by rule model")
    print((table * 100).round(1).to_string(float_format=lambda v: f"{v:5.1f}%"))
    print("\nColumn sums (conservation check):")
    print(table.sum().round(6).to_string())

    print("\nMost common converting paths")
    from streamly.attribution.sessionize import path_summary

    print(path_summary(js, top_n=5).to_string(index=False))


if __name__ == "__main__":
    _main()
