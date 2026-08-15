"""Chart builders for the Streamly app.

Kept out of ``app/streamlit_app.py`` on purpose: chart *specifications* are
logic, and logic belongs somewhere importable and testable. The app composes
these; the test suite validates every spec compiles.

Design rules applied here (and why)
-----------------------------------
* **Categorical hues are assigned in a fixed order and never cycled.** Colour
  follows the entity, so filtering a series out never repaints the survivors.
  Only the first three categorical slots are used -- that trio is validated for
  colour-vision separation across all pairs, which the full eight-slot set is
  not.
* **Diverging blue/red with a neutral midpoint** for the budget deltas, because
  the data has a true zero with opposite-signed meaning. A sequential ramp would
  imply "more of the same thing" across a sign change.
* **One axis, never two.** Metrics on different scales get separate charts.
* **Identity is never carried by colour alone**: every multi-series chart ships a
  legend, and the app pairs each chart with a table view.
* **Recessive chrome.** Hairline gridlines, muted axis ink, thin marks -- the
  data is the only thing at full contrast.

Palettes are defined for both light and dark surfaces; the dark column is the
same hues re-stepped for the dark surface rather than an automatic inversion.
"""
from __future__ import annotations

from dataclasses import dataclass

import altair as alt
import pandas as pd


@dataclass(frozen=True)
class Palette:
    """Colour roles for one surface mode."""

    surface: str
    text_primary: str
    text_secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, str, str]     # categorical slots 1-3
    positive: str                    # diverging warm/cool poles
    negative: str
    neutral: str
    good: str
    warning: str
    critical: str


LIGHT = Palette(
    surface="#fcfcfb", text_primary="#0b0b0b", text_secondary="#52514e",
    muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    positive="#2a78d6", negative="#d03b3b", neutral="#f0efec",
    good="#0ca30c", warning="#fab219", critical="#d03b3b",
)

DARK = Palette(
    surface="#1a1a19", text_primary="#ffffff", text_secondary="#c3c2b7",
    muted="#898781", grid="#2c2c2a", axis="#383835",
    series=("#3987e5", "#d95926", "#199e70"),
    positive="#3987e5", negative="#d03b3b", neutral="#383835",
    good="#0ca30c", warning="#fab219", critical="#d03b3b",
)

# Matches the app stylesheet (app/theme.py) so charts and page text are one
# typeface. Falls back to the system stack if the webfont has not loaded.
FONT = '"Fira Sans", system-ui, -apple-system, "Segoe UI", sans-serif'

# Layered charts are LayerChart, single-mark charts are Chart; both accept the
# same top-level configuration, so the helpers are written against the union.
ChartLike = alt.Chart | alt.LayerChart


ROW_HEIGHT = 34           # px per categorical band; below ~30 the labels collide
INTERVAL_ROW_HEIGHT = 72  # interval rows carry a rule plus a point marker
CHART_CHROME = 130        # title + axis title + tick labels
LEGEND_CHROME = 40        # a top-oriented legend takes another row


def _band_height(n_rows: int, legend: bool = False) -> int:
    """Total chart height needed for ``n_rows`` categorical bands.

    Streamlit renders these charts with Vega's autosize **fit**, which makes
    ``height`` the height of the whole container rather than of the plot area --
    so the title, legend and axis chrome are subtracted from it, not added to
    it. Sizing as if ``height`` were the plot area is what collapsed the
    reallocation chart: at a nominal 214px the ~170px of chrome left roughly 9px
    per band and five channels rendered on top of each other.

    Measured empirically against a live browser rather than assumed: at
    ``height=520`` the same five-row chart produced 72px bands, which is what
    pins the chrome constants below.
    """
    return n_rows * ROW_HEIGHT + CHART_CHROME + (LEGEND_CHROME if legend else 0)


def _style(chart: ChartLike, p: Palette) -> ChartLike:
    """Apply recessive chrome consistently to every chart."""
    return (
        chart.configure_view(strokeWidth=0, fill=p.surface)
        .configure_axis(
            labelColor=p.muted, titleColor=p.text_secondary, labelFont=FONT,
            titleFont=FONT, labelFontSize=12, titleFontSize=12,
            gridColor=p.grid, gridWidth=1, domainColor=p.axis, tickColor=p.axis,
            titleFontWeight="normal",
        )
        .configure_legend(
            labelColor=p.text_secondary, titleColor=p.text_secondary,
            labelFont=FONT, titleFont=FONT, labelFontSize=12, titleFontSize=12,
            symbolType="square", symbolSize=110, orient="top", direction="horizontal",
            title=None,
        )
        .configure_title(color=p.text_primary, font=FONT, fontSize=15, anchor="start")
        .configure_text(font=FONT)
    )


def recovery_error_chart(scores: pd.DataFrame, p: Palette = LIGHT) -> ChartLike:
    """Recovery error by attribution model -- the headline validation chart.

    A single series: the quantity is one magnitude measured across models, so
    colouring by model would encode identity that the axis already carries.
    Bars are labelled directly, which also satisfies the relief rule for the
    lower-contrast surface.
    """
    df = scores.reset_index().rename(columns={"index": "model"})
    df = df.assign(mae_pp=df["mae"] * 100).sort_values("mae_pp")

    base = alt.Chart(df).encode(
        y=alt.Y("model:N", sort="x", title=None,
                axis=alt.Axis(labelFontSize=13, labelColor=p.text_secondary)),
        x=alt.X("mae_pp:Q", title="Mean absolute error vs ground truth (share points)",
                scale=alt.Scale(nice=True)),
    )
    bars = base.mark_bar(
        color=p.series[0], height=18, cornerRadiusEnd=4,
    )
    labels = base.mark_text(
        align="left", dx=6, color=p.text_secondary, fontSize=12, font=FONT,
    ).encode(text=alt.Text("mae_pp:Q", format=".2f"))

    # Height scales with the row count: a fixed height collapses the bands as
    # models are added, overlapping the bars and colliding the axis labels.
    return _style(
        (bars + labels).properties(height=_band_height(len(df)), title="Lower is better"), p
    )


def channel_share_chart(shares: pd.DataFrame, p: Palette = LIGHT) -> ChartLike:
    """Credited share per channel: ground truth against two models.

    Exactly three series -- the validated all-pairs limit for the categorical
    slots. Ground truth leads the fixed hue order because it is the reference
    every other bar is read against.
    """
    keep = [c for c in ("TRUE", "last_touch", "shapley") if c in shares.columns]
    df = (
        shares[keep].reset_index()
        .melt(id_vars="channel", var_name="series", value_name="share")
    )
    df["share"] = df["share"] * 100
    order = [c for c in ("TRUE", "last_touch", "shapley") if c in keep]

    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("channel:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
        xOffset=alt.XOffset("series:N", sort=order),
        y=alt.Y("share:Q", title="Credited share (%)"),
        color=alt.Color(
            "series:N", sort=order,
            scale=alt.Scale(domain=order, range=list(p.series[:len(order)])),
            legend=alt.Legend(title=None),
        ),
        tooltip=[
            alt.Tooltip("channel:N", title="Channel"),
            alt.Tooltip("series:N", title="Model"),
            alt.Tooltip("share:Q", title="Share", format=".1f"),
        ],
    )
    return _style(
        bars.properties(height=300, title="Attribution share vs ground truth"), p
    )


def reallocation_chart(plan_table: pd.DataFrame, p: Palette = LIGHT) -> ChartLike:
    """Proposed monthly budget change per channel.

    Diverging encoding: the data has a real zero and the two signs mean opposite
    things (fund vs defund). Colour is redundant with bar direction and the
    signed labels, so it never carries the meaning alone.
    """
    df = plan_table.reset_index()[["channel", "delta"]].copy()
    df["delta_k"] = df["delta"] / 1_000.0
    df["direction"] = df["delta"].apply(lambda v: "increase" if v >= 0 else "decrease")

    # Sort in pandas and pass the resulting order explicitly. An
    # ``EncodingSortField`` cannot be resolved here: the zero-line layer carries
    # a different dataset with no ``delta`` column, and the unresolved sort
    # collapses the shared band scale -- five channels render onto two rows with
    # the bars drawn on top of each other.
    df = df.sort_values("delta", ascending=False).reset_index(drop=True)
    channel_order = df["channel"].tolist()

    base = alt.Chart(df).encode(
        y=alt.Y("channel:N", sort=channel_order, title=None,
                axis=alt.Axis(labelFontSize=13, labelColor=p.text_secondary)),
        x=alt.X("delta_k:Q", title="Monthly budget change ($K)"),
    )
    bars = base.mark_bar(height=18, cornerRadius=4).encode(
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(domain=["increase", "decrease"], range=[p.positive, p.negative]),
            legend=alt.Legend(title=None),
        ),
        tooltip=[
            alt.Tooltip("channel:N", title="Channel"),
            alt.Tooltip("delta:Q", title="Change", format="+,.0f"),
        ],
    )
    # Labels sit just beyond the bar end, on the side the bar points, so they
    # never overprint the mark they annotate.
    label_mark = dict(fontSize=12, color=p.text_secondary, font=FONT, baseline="middle")
    labels = (
        base.transform_filter(alt.datum.delta_k >= 0)
        .mark_text(align="left", dx=6, **label_mark)
        .encode(text=alt.Text("delta_k:Q", format="+.1f"))
        + base.transform_filter(alt.datum.delta_k < 0)
        .mark_text(align="right", dx=-6, **label_mark)
        .encode(text=alt.Text("delta_k:Q", format="+.1f"))
    )
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        color=p.axis, strokeWidth=1,
    ).encode(x="x:Q")

    return _style(
        (bars + zero + labels).properties(
            height=_band_height(len(df), legend=True),
            title="Budget reallocation (Shapley, capped +/-30%)"
        ),
        p,
    )


def effect_interval_chart(
    label_to_result: dict[str, tuple[float, float, float]],
    practical_threshold: float,
    p: Palette = LIGHT,
) -> ChartLike:
    """Point estimate with confidence interval, against the ship threshold.

    The chart the decision is actually made from: the ship rule compares the
    *interval* to the practical threshold, so the threshold is drawn as a rule
    and the reader can see which side the whole interval falls on.
    """
    rows = [
        {"label": k, "effect": v[0] * 100, "low": v[1] * 100, "high": v[2] * 100}
        for k, v in label_to_result.items()
    ]
    df = pd.DataFrame(rows)

    base = alt.Chart(df).encode(
        y=alt.Y("label:N", title=None,
                axis=alt.Axis(labelFontSize=13, labelColor=p.text_secondary)),
    )
    interval = base.mark_rule(strokeWidth=2, color=p.series[0]).encode(
        x=alt.X("low:Q", title="Absolute effect (percentage points)"),
        x2="high:Q",
    )
    point = base.mark_point(
        filled=True, size=110, color=p.series[0], stroke=p.surface, strokeWidth=2,
    ).encode(
        x="effect:Q",
        tooltip=[
            alt.Tooltip("label:N", title="Estimator"),
            alt.Tooltip("effect:Q", title="Effect (pp)", format="+.3f"),
            alt.Tooltip("low:Q", title="CI low", format="+.3f"),
            alt.Tooltip("high:Q", title="CI high", format="+.3f"),
        ],
    )
    threshold = alt.Chart(
        pd.DataFrame({"x": [practical_threshold * 100]})
    ).mark_rule(color=p.warning, strokeWidth=2, strokeDash=[5, 3]).encode(x="x:Q")
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        color=p.axis, strokeWidth=1
    ).encode(x="x:Q")

    return _style(
        (zero + threshold + interval + point).properties(
            # Streamlit renders with autosize 'fit', so the title and x-axis
            # eat into the declared height rather than adding to it. The plot
            # area needs a generous allowance or the rows collapse together.
            height=CHART_CHROME + len(df) * INTERVAL_ROW_HEIGHT,
            title="Effect and 95% CI vs the practical significance threshold (dashed)",
        ),
        p,
    )


def sequential_boundary_chart(
    information_fractions: tuple[float, ...],
    boundaries: tuple[float, ...],
    observed_look: int | None = None,
    observed_z: float | None = None,
    p: Palette = LIGHT,
) -> ChartLike:
    """Efficacy boundary by look, with the observed statistic overlaid."""
    df = pd.DataFrame({
        "look": range(1, len(boundaries) + 1),
        "information": information_fractions,
        "boundary": boundaries,
    })
    line = alt.Chart(df).mark_line(
        strokeWidth=2, color=p.series[0], point=alt.OverlayMarkDef(
            size=80, filled=True, fill=p.series[0], stroke=p.surface, strokeWidth=2
        ),
    ).encode(
        x=alt.X("look:O", title="Look", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("boundary:Q", title="|Z| required to stop"),
        tooltip=[
            alt.Tooltip("look:O", title="Look"),
            alt.Tooltip("information:Q", title="Information fraction", format=".2f"),
            alt.Tooltip("boundary:Q", title="Boundary", format=".3f"),
        ],
    )
    layers: list[ChartLike] = [line]

    if observed_look is not None and observed_z is not None:
        obs = pd.DataFrame({"look": [observed_look], "z": [abs(observed_z)]})
        layers.append(
            alt.Chart(obs).mark_point(
                filled=True, size=160, color=p.series[1], stroke=p.surface, strokeWidth=2,
                shape="diamond",
            ).encode(
                x=alt.X("look:O"), y=alt.Y("z:Q"),
                tooltip=[alt.Tooltip("z:Q", title="Observed |Z|", format=".3f")],
            )
        )

    return _style(
        alt.layer(*layers).properties(
            height=280, title="O'Brien-Fleming efficacy boundaries"
        ),
        p,
    )
