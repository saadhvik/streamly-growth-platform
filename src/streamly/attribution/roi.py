"""Channel ROI and budget reallocation from credited value.

Turns attribution shares into the numbers a CFO argues about: cost per
acquisition, return on ad spend, and a concrete monthly budget move with an
expected return and a stated confidence.

Conventions
-----------
* **Net revenue.** Refunded conversions contribute zero revenue. Crediting
  gross revenue would reward channels that acquire users who churn back out.
* **Monthly normalization.** The warehouse holds 90 days; every dollar figure
  is divided to a monthly run rate so it reconciles against the $500K/month
  brief.
* **Spend is actual, credit is modelled.** Spend comes from the ``spend``
  table; only the *allocation of value* varies by model. Two models therefore
  differ in ROI purely because they disagree about which channel earned the
  conversion.

The reallocation rule
---------------------
Move budget toward value share, capped. Target spend share is the model's
credited value share; each channel's move is capped at ``max_shift`` of its
current budget so the plan is executable in one cycle rather than a paper
optimum that would strand campaign commitments.

Expected gain is a **first-order** estimate: it prices moved dollars at each
channel's *current* credited conversions per dollar. That assumes locally
constant returns, which overstates gains for large moves into small channels --
so the plan caps moves, reports the assumption inline, and treats the number as
a directional prior to be confirmed by a geo holdout, not a promise.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from streamly import warehouse
from streamly.attribution import rules
from streamly.attribution.markov import markov_attribution
from streamly.attribution.sessionize import JourneySet
from streamly.attribution.shapley import shapley_attribution
from streamly.config import ATTRIBUTION, CHANNELS, WAREHOUSE_PATH, AttributionConfig

DAYS_PER_MONTH = 30.0

# Models whose shares come from a coalition/chain solve rather than per-journey
# weights, and so are applied to conversion and revenue totals post hoc.
_DDM_MODELS = ("markov", "shapley")


@dataclass(frozen=True)
class ReallocationPlan:
    """A concrete, capped budget move with its expected first-order return."""

    model: str
    table: pd.DataFrame            # per-channel current/proposed spend and deltas
    dollars_moved: float           # total monthly dollars shifted
    expected_incremental_conversions: float
    expected_incremental_revenue: float
    max_shift: float


def channel_spend(
    path: Path | str = WAREHOUSE_PATH, channels: tuple[str, ...] = CHANNELS
) -> pd.Series:
    """Monthly spend per channel, from the actual spend log."""
    con = warehouse.connect(path, read_only=True)
    try:
        df = con.execute(
            "SELECT channel, SUM(spend) AS spend, COUNT(DISTINCT date) AS days "
            "FROM spend GROUP BY channel"
        ).fetch_df()
    finally:
        con.close()
    monthly = df.set_index("channel")["spend"] / (df.set_index("channel")["days"] / DAYS_PER_MONTH)
    return monthly.reindex(list(channels)).fillna(0.0)


def net_revenue_journeys(js: JourneySet) -> JourneySet:
    """Copy of ``js`` with refunded conversions' revenue zeroed."""
    return replace(js, revenue=np.where(js.is_refunded, 0.0, js.revenue))


def credited_value(
    js: JourneySet,
    model: str,
    cfg: AttributionConfig = ATTRIBUTION,
    channels: tuple[str, ...] = CHANNELS,
) -> tuple[pd.Series, pd.Series]:
    """``(credited_conversions, credited_net_revenue)`` per channel for one model.

    Rule models distribute per journey; the data-driven models produce a single
    share vector, which is applied to the population totals. Both routes credit
    exactly the same totals, so ROI stays comparable across models.
    """
    net = net_revenue_journeys(js)
    if model in _DDM_MODELS:
        shares = (markov_attribution if model == "markov" else shapley_attribution)(
            net, channels
        ).attribution
        s = pd.Series(shares).reindex(list(channels))
        total_conv = float(net.n_conversions)
        total_rev = float(net.revenue[net.converted].sum())
        return s * total_conv, s * total_rev

    conv, rev = rules.credit_by_channel(net, model, cfg, channels)
    return (
        pd.Series(conv).reindex(list(channels)),
        pd.Series(rev).reindex(list(channels)),
    )


def roi_table(
    js: JourneySet,
    model: str,
    cfg: AttributionConfig = ATTRIBUTION,
    channels: tuple[str, ...] = CHANNELS,
    path: Path | str = WAREHOUSE_PATH,
    months: float | None = None,
) -> pd.DataFrame:
    """Per-channel spend, credited value, CAC and ROAS under one model.

    ``roas`` is **first-payment** return, not lifetime: the warehouse records
    only the initial subscription payment, so the levels read far below 1.0 and
    should be quoted as a payback input, never as profitability. The
    reallocation decision is unaffected -- multiplying every channel's revenue
    by a common LTV factor rescales all ROAS values identically and leaves the
    ranking, the value shares, and the proposed budget move unchanged.
    """
    spend = channel_spend(path, channels)
    conv, rev = credited_value(js, model, cfg, channels)

    # Credited value covers the full history; put it on the same monthly basis.
    if months is None:
        con = warehouse.connect(path, read_only=True)
        try:
            row = con.execute("SELECT COUNT(DISTINCT date) FROM spend").fetchone()
        finally:
            con.close()
        if not row or not row[0]:
            raise ValueError("spend table is empty; cannot put credited value on a monthly basis")
        months = float(row[0]) / DAYS_PER_MONTH
    conv_m, rev_m = conv / months, rev / months

    out = pd.DataFrame({
        "monthly_spend": spend,
        "credited_conversions": conv_m,
        "credited_revenue": rev_m,
    })
    out["spend_share"] = out["monthly_spend"] / out["monthly_spend"].sum()
    out["value_share"] = out["credited_conversions"] / out["credited_conversions"].sum()
    # Guard against a zero-credit channel producing an infinite CAC.
    out["cac"] = np.where(out["credited_conversions"] > 0,
                          out["monthly_spend"] / out["credited_conversions"].replace(0, np.nan),
                          np.nan)
    out["roas"] = out["credited_revenue"] / out["monthly_spend"]
    out.index.name = "channel"
    return out


def reallocation_plan(
    js: JourneySet,
    model: str = "shapley",
    max_shift: float = 0.30,
    cfg: AttributionConfig = ATTRIBUTION,
    channels: tuple[str, ...] = CHANNELS,
    path: Path | str = WAREHOUSE_PATH,
) -> ReallocationPlan:
    """Propose a budget-neutral monthly reallocation toward credited value share.

    ``max_shift`` caps each channel's move at that fraction of its current
    budget. After capping, the residual is rebalanced pro rata so the plan stays
    budget-neutral to the cent.
    """
    roi = roi_table(js, model, cfg, channels, path)
    budget = float(roi["monthly_spend"].sum())

    target = roi["value_share"] * budget
    lower = roi["monthly_spend"] * (1.0 - max_shift)
    upper = roi["monthly_spend"] * (1.0 + max_shift)
    proposed = target.clip(lower=lower, upper=upper)

    # Capping breaks budget neutrality; return the residual to the uncapped
    # channels in proportion to their headroom.
    residual = budget - float(proposed.sum())
    if abs(residual) > 1e-9:
        headroom = (upper - proposed) if residual > 0 else (proposed - lower)
        total_headroom = float(headroom.sum())
        if total_headroom > 1e-9:
            proposed = proposed + np.sign(residual) * headroom * (
                abs(residual) / total_headroom
            )

    delta = proposed - roi["monthly_spend"]
    conv_per_dollar = roi["credited_conversions"] / roi["monthly_spend"]
    rev_per_dollar = roi["credited_revenue"] / roi["monthly_spend"]

    table = pd.DataFrame({
        "current_spend": roi["monthly_spend"],
        "spend_share": roi["spend_share"],
        "value_share": roi["value_share"],
        "proposed_spend": proposed,
        "delta": delta,
        "delta_pct": delta / roi["monthly_spend"],
        "cac": roi["cac"],
        "roas": roi["roas"],
    })
    table.index.name = "channel"

    return ReallocationPlan(
        model=model,
        table=table,
        dollars_moved=float(delta[delta > 0].sum()),
        expected_incremental_conversions=float((delta * conv_per_dollar).sum()),
        expected_incremental_revenue=float((delta * rev_per_dollar).sum()),
        max_shift=max_shift,
    )


def misallocation_vs_incumbent(
    js: JourneySet,
    incumbent: str = "last_touch",
    challenger: str = "shapley",
    cfg: AttributionConfig = ATTRIBUTION,
    channels: tuple[str, ...] = CHANNELS,
    path: Path | str = WAREHOUSE_PATH,
) -> tuple[pd.DataFrame, float]:
    """Monthly dollars steered differently by two models' value shares.

    The headline finance number: how much of the budget the incumbent model
    points at the wrong channel. Returned as ``(per_channel, total_absolute)``
    where the total counts each misplaced dollar once.
    """
    budget = float(channel_spend(path, channels).sum())
    a = roi_table(js, incumbent, cfg, channels, path)["value_share"] * budget
    b = roi_table(js, challenger, cfg, channels, path)["value_share"] * budget
    per_channel = pd.DataFrame({
        f"implied_by_{incumbent}": a,
        f"implied_by_{challenger}": b,
        "difference": b - a,
    })
    per_channel.index.name = "channel"
    return per_channel, float((b - a).abs().sum() / 2.0)


def _main() -> None:
    from streamly.attribution.sessionize import build_journeys

    js = build_journeys()

    for model in ("last_touch", "shapley"):
        print(f"\n=== ROI under {model} ===")
        t = roi_table(js, model)
        print(t.assign(
            monthly_spend=lambda d: d["monthly_spend"].round(0),
            credited_conversions=lambda d: d["credited_conversions"].round(1),
            credited_revenue=lambda d: d["credited_revenue"].round(0),
            spend_share=lambda d: (d["spend_share"] * 100).round(1),
            value_share=lambda d: (d["value_share"] * 100).round(1),
            cac=lambda d: d["cac"].round(2),
            roas=lambda d: d["roas"].round(3),
        ).to_string())

    diff, total = misallocation_vs_incumbent(js)
    print("\n=== Budget steered differently: last_touch vs shapley ===")
    print(diff.round(0).to_string())
    print(f"\nMisallocated monthly budget: ${total:,.0f}")

    plan = reallocation_plan(js, "shapley", max_shift=0.30)
    print("\n=== Proposed reallocation (Shapley, capped at +/-30%) ===")
    print(plan.table.assign(
        current_spend=lambda d: d["current_spend"].round(0),
        proposed_spend=lambda d: d["proposed_spend"].round(0),
        delta=lambda d: d["delta"].round(0),
        spend_share=lambda d: (d["spend_share"] * 100).round(1),
        value_share=lambda d: (d["value_share"] * 100).round(1),
        delta_pct=lambda d: (d["delta_pct"] * 100).round(1),
        cac=lambda d: d["cac"].round(2),
        roas=lambda d: d["roas"].round(3),
    ).to_string())
    print(f"\ndollars moved:        ${plan.dollars_moved:,.0f}/mo")
    print(f"expected +conversions: {plan.expected_incremental_conversions:,.0f}/mo "
          f"(first-order, constant-returns assumption)")
    print(f"expected +revenue:     ${plan.expected_incremental_revenue:,.0f}/mo")


if __name__ == "__main__":
    _main()
