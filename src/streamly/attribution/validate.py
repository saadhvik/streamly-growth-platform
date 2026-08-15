"""Score every attribution model against the recorded ground truth.

This is the module that makes the whole platform falsifiable. Ground truth is
read from ``data/ground_truth/ground_truth.json`` -- written by the generator
and never touched by any model -- so the recovery error here is an honest
out-of-model measurement, not a fit statistic.

Headline gate (Phase 3): data-driven attribution must recover the true channel
importance vector with materially lower error than last-touch.

Run:  PYTHONPATH=src python -m streamly.attribution.validate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from streamly.attribution import rules
from streamly.attribution.markov import markov_attribution
from streamly.attribution.sessionize import JourneySet, build_journeys
from streamly.attribution.shapley import shapley_attribution
from streamly.config import ATTRIBUTION, CHANNELS, GROUND_TRUTH_DIR, AttributionConfig

BASELINE_MODEL = "last_touch"   # the incumbent every method is scored against


def load_ground_truth(path: Path | None = None) -> dict[str, float]:
    """Read the locked true channel-importance vector."""
    p = path or (GROUND_TRUTH_DIR / "ground_truth.json")
    with open(p, encoding="utf-8") as f:
        return dict(json.load(f)["channel_importance_true"])


def attribution_matrix(
    js: JourneySet,
    cfg: AttributionConfig = ATTRIBUTION,
    channels: tuple[str, ...] = CHANNELS,
) -> pd.DataFrame:
    """All models' attribution shares, one column per model.

    Rule models plus the two data-driven methods, on a common scale (each
    column sums to 1) so they are directly comparable to ground truth.
    """
    table = rules.credit_table(js, cfg, channels=channels)
    table["markov"] = pd.Series(markov_attribution(js, channels).attribution)
    table["shapley"] = pd.Series(shapley_attribution(js, channels).attribution)
    return table


def recovery_scores(
    table: pd.DataFrame,
    truth: dict[str, float] | None = None,
    baseline: str = BASELINE_MODEL,
) -> pd.DataFrame:
    """Per-model recovery error against ground truth, ranked best-first.

    Columns
    -------
    mae / rmse / max_abs_error
        Error between recovered and true shares, in share points.
    error_reduction_vs_baseline
        Fractional MAE reduction relative to ``baseline`` -- the number the
        Phase 3 gate is written against.
    """
    truth = truth or load_ground_truth()
    t = np.array([truth[c] for c in table.index], dtype=float)

    rows = []
    for model in table.columns:
        err = table[model].to_numpy(dtype=float) - t
        rows.append({
            "model": model,
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "max_abs_error": float(np.abs(err).max()),
        })
    out = pd.DataFrame(rows).set_index("model")
    # .loc returns a broad scalar union under pandas-stubs; narrow it once.
    base_mae = float(out["mae"].loc[baseline])
    out["error_reduction_vs_baseline"] = 1.0 - out["mae"] / base_mae
    return out.sort_values("mae")


def comparison_report(
    js: JourneySet,
    cfg: AttributionConfig = ATTRIBUTION,
    channels: tuple[str, ...] = CHANNELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(shares_vs_truth, scores)`` -- the two tables the memo and app render."""
    table = attribution_matrix(js, cfg, channels)
    truth = load_ground_truth()
    shares = table.copy()
    shares.insert(0, "TRUE", [truth[c] for c in shares.index])
    return shares, recovery_scores(table, truth)


def _main() -> None:
    js = build_journeys()
    shares, scores = comparison_report(js)

    print(f"journeys: {len(js):,}   conversions: {js.n_conversions:,}\n")
    print("Attribution share by model vs GROUND TRUTH")
    print((shares * 100).round(1).to_string(float_format=lambda v: f"{v:6.1f}"))

    print("\nRecovery error vs ground truth (share points, lower is better)")
    disp = scores.copy()
    disp[["mae", "rmse", "max_abs_error"]] *= 100
    print(disp.round(2).to_string())

    # Values are pulled out and narrowed to float first: indexing a DataFrame
    # with .loc[row, col] is typed as a broad scalar union under pandas-stubs,
    # so arithmetic and format specifiers on it cannot be checked.
    best = str(scores.index[0])
    best_mae = float(scores["mae"].loc[best]) * 100
    baseline_mae = float(scores["mae"].loc[BASELINE_MODEL]) * 100
    reduction = float(scores["error_reduction_vs_baseline"].loc[best])
    print(f"\nBest recovery: {best} (MAE {best_mae:.2f}pp vs "
          f"last-touch {baseline_mae:.2f}pp, {reduction:.0%} lower)")


if __name__ == "__main__":
    _main()
