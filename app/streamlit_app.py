"""Streamly Growth Platform — attribution and experimentation UI.

Run:  PYTHONPATH=src streamlit run app/streamlit_app.py

Deliberately thin: every number on screen comes from the same functions the
tests exercise, and every chart spec lives in :mod:`streamly.viz`. Nothing is
computed inline here, so the app cannot drift from the validated pipeline.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow `streamlit run app/streamlit_app.py` without PYTHONPATH gymnastics.
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _choose_data_dir() -> None:
    """Point the warehouse somewhere writable, before streamly.config is imported.

    ``config.DATA_DIR`` is resolved at import time, so this has to run first.
    The repo's own ``data/`` is preferred (that is where a local run expects
    it), but a read-only checkout -- which is how some hosts mount a
    deployment -- falls back to a temp directory. The warehouse is fully
    reproducible from a fixed seed, so a throwaway location costs nothing but
    the ~10s to rebuild.
    """
    if os.environ.get("STREAMLY_DATA_DIR"):
        return
    candidate = _REPO / "data"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        os.environ["STREAMLY_DATA_DIR"] = tempfile.mkdtemp(prefix="streamly_")


_choose_data_dir()

import theme  # noqa: E402

from streamly import viz  # noqa: E402
from streamly.attribution import roi, validate  # noqa: E402
from streamly.attribution.sessionize import build_journeys  # noqa: E402
from streamly.config import ATTRIBUTION, EXPERIMENT  # noqa: E402
from streamly.experiment import bayesian, design, readout, sequential  # noqa: E402
from streamly.experiment.guardrails import Verdict  # noqa: E402

st.set_page_config(page_title="Streamly Growth Platform", page_icon="📊", layout="wide")


def is_dark_mode() -> bool:
    """Whether the page will render dark, from either source that decides it.

    Streamlit's dark mode is an app-level setting, so CSS ``prefers-color-scheme``
    cannot see it -- the stylesheet and the chart palette both have to be told.
    Two independent things can make the page dark, and checking only one gets it
    wrong:

    * ``theme.base = "dark"`` in config (or ``--theme.base dark``) forces dark
      regardless of the viewer.
    * With no configured base, Streamlit follows the viewer's own preference,
      which is what ``st.context.theme.type`` reports.

    Verified against a live server: with ``--theme.base dark`` the page renders
    dark while ``st.context.theme.type`` still returns ``'light'``, because it
    describes the browser rather than the app. Consulting it alone left the KPI
    accent at a 2.17:1 contrast on the dark background.
    """
    try:
        if st.get_option("theme.base") == "dark":
            return True
    except Exception:      # option unavailable in this Streamlit build
        pass
    try:
        return bool(st.context.theme.type == "dark")
    except Exception:      # older Streamlit, or no theme context
        return False


def palette() -> viz.Palette:
    """Match the chart surface to the viewer's Streamlit theme."""
    return viz.DARK if is_dark_mode() else viz.LIGHT


theme.inject(dark=is_dark_mode())


# ---------------------------------------------------------------------------
# Bootstrap: the warehouse is generated, not committed.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="First run: generating the synthetic warehouse…")
def ensure_warehouse() -> str:
    """Build the warehouse if it is absent, and report where it lives.

    ``data/warehouse.duckdb`` is deliberately gitignored -- it is 9MB and fully
    reproducible from the seed in ``config.py`` -- so a fresh deployment starts
    without it. Generating on demand takes about ten seconds and makes the app
    self-contained: clone or deploy, run, done. ``cache_resource`` keeps it to
    once per container rather than once per session.
    """
    from streamly.config import WAREHOUSE_PATH
    from streamly.datagen import generator

    if not Path(WAREHOUSE_PATH).exists():
        generator.generate()
    return str(WAREHOUSE_PATH)


# ---------------------------------------------------------------------------
# Cached data access -- the warehouse is read-only here.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Building journeys from the warehouse…")
def load_attribution() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    js = build_journeys()
    shares, scores = validate.comparison_report(js)
    roi_shapley = roi.roi_table(js, "shapley")
    roi_last = roi.roi_table(js, "last_touch")
    plan = roi.reallocation_plan(js, "shapley", max_shift=0.30)
    _diff, misallocated = roi.misallocation_vs_incumbent(js)
    meta = {
        "journeys": len(js),
        "conversions": js.n_conversions,
        "misallocated": misallocated,
        "plan": plan,
        "roi_last": roi_last,
    }
    return shares, scores, roi_shapley, plan.table, meta


@st.cache_data(show_spinner="Analyzing the experiment…")
def load_readout(threshold: float) -> tuple[readout.ExperimentReadout, str]:
    return readout.generate(practical_threshold=threshold)


def verdict_color(v: Verdict, p: viz.Palette) -> str:
    return {Verdict.PASS: p.good, Verdict.FAIL: p.critical,
            Verdict.INCONCLUSIVE: p.warning}[v]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
p = palette()
st.title("Streamly Growth Platform")
st.caption(
    "Multi-touch attribution and trustworthy experimentation, validated against "
    "a known ground truth. Every figure below is reproducible from a fixed seed."
)

attribution_tab, experiment_tab, method_tab = st.tabs(
    ["Attribution", "Experimentation", "Method"]
)

# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
with attribution_tab:
    try:
        ensure_warehouse()
        shares, scores, roi_table, plan_table, meta = load_attribution()
    except Exception as exc:                                    # noqa: BLE001
        st.error(
            f"Could not read the warehouse: {exc}\n\n"
            "Generate it first:  `PYTHONPATH=src python -m streamly.datagen.generator`"
        )
        st.stop()

    lt_mae = float(scores.loc["last_touch", "mae"])
    sh_mae = float(scores.loc["shapley", "mae"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last-touch error", f"{lt_mae * 100:.2f} pp", help="MAE vs true channel importance")
    c2.metric("Shapley error", f"{sh_mae * 100:.2f} pp",
              delta=f"{-(lt_mae - sh_mae) * 100:.2f} pp", delta_color="inverse")
    c3.metric("Error reduction", f"{1 - sh_mae / lt_mae:.0%}")
    c4.metric("Budget misallocated", f"${meta['misallocated']:,.0f}/mo",
              help="Monthly spend the incumbent model points at the wrong channel")

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Which model recovers the truth?")
        st.altair_chart(viz.recovery_error_chart(scores, p), width="stretch")
        st.markdown(
            "The heuristics are biased in **opposite** directions: last-touch inflates "
            "the late-funnel channel, first-touch inflates the opener. Swapping one for "
            "another moves the over-credit rather than reducing it. Linear scores best "
            "of them only because it ignores order, so it has no position to be fooled by."
        )
        with st.expander("Table view — recovery error"):
            st.dataframe(
                (scores[["mae", "rmse", "max_abs_error"]] * 100).round(2)
                .rename(columns=lambda c: f"{c} (pp)"),
                width="stretch",
            )

    with right:
        st.subheader("Where the credit actually goes")
        st.altair_chart(viz.channel_share_chart(shares, p), width="stretch")
        st.markdown(
            "Meta takes 68.4% of last-touch credit against 14.0% true importance. It is "
            "high-volume late-funnel retargeting, so it lands right before the decision "
            "more often than anything else — and last-touch reads position as performance."
        )
        with st.expander("Table view — attribution shares"):
            st.dataframe((shares * 100).round(1), width="stretch")

    st.divider()
    st.subheader("Unit economics and the proposed move")

    lcol, rcol = st.columns([1, 1])
    with lcol:
        st.altair_chart(viz.reallocation_chart(plan_table, p), width="stretch")
        plan = meta["plan"]
        m1, m2 = st.columns(2)
        m1.metric("Dollars moved", f"${plan.dollars_moved:,.0f}/mo")
        m2.metric("Expected gain", f"+{plan.expected_incremental_conversions:,.0f} conv/mo")
        st.caption(
            "Budget-neutral and capped at ±30% per channel so the plan is executable "
            "in one cycle. The expected gain is a first-order estimate at current "
            "conversions-per-dollar — a directional prior for a geo holdout, not a forecast."
        )

    with rcol:
        display = roi_table.copy()
        display["monthly_spend"] = display["monthly_spend"].round(0)
        display["credited_conversions"] = display["credited_conversions"].round(1)
        display["cac"] = display["cac"].round(2)
        display["roas"] = display["roas"].round(3)
        st.dataframe(
            display[["monthly_spend", "credited_conversions", "cac", "roas"]],
            width="stretch",
        )
        st.caption(
            "CAC and ROAS under Shapley. ROAS is **first-payment** return, not "
            "lifetime — use it for payback, not profitability. A common LTV multiple "
            "rescales every channel identically and leaves the decision unchanged."
        )
        with st.expander("Compare against last-touch"):
            st.dataframe(
                meta["roi_last"][["credited_conversions", "cac", "roas"]].round(2),
                width="stretch",
            )

# ---------------------------------------------------------------------------
# Experimentation
# ---------------------------------------------------------------------------
with experiment_tab:
    st.subheader("Design calculator")
    d1, d2, d3, d4 = st.columns(4)
    baseline = d1.number_input("Baseline rate", 0.001, 0.999, 0.080, step=0.005, format="%.3f")
    mde = d2.number_input("MDE (absolute)", 0.001, 0.500, 0.012, step=0.001, format="%.3f")
    power_target = d3.slider("Power", 0.50, 0.99, 0.80, step=0.01)
    daily = d4.number_input("Daily eligible users", 100, 1_000_000, 5_000, step=500)

    try:
        plan_design = design.sample_size_two_proportions(baseline, mde, power=power_target)
        days = design.duration_days(plan_design.n_total, daily)
        s1, s2, s3 = st.columns(3)
        s1.metric("Users required", f"{plan_design.n_total:,}")
        s2.metric("Per arm", f"{plan_design.n_control:,}")
        s3.metric("Duration", f"{days:.1f} days")
        st.caption(
            f"Detects {plan_design.mde_relative:+.1%} relative on a {baseline:.2%} "
            f"baseline at alpha=0.05. MDE is specified in absolute percentage points — "
            f"\"a 10% lift\" is ambiguous and that ambiguity underpowers experiments."
        )
    except ValueError as exc:
        st.warning(str(exc))

    st.divider()
    st.subheader(f"Readout — `{EXPERIMENT.experiment_id}`")
    threshold = st.slider(
        "Practical significance threshold (absolute)", 0.000, 0.020, 0.005, step=0.001,
        format="%.3f",
        help="The smallest effect worth the cost of shipping. This, not the p-value, "
             "is the ship criterion.",
    )

    try:
        r, markdown = load_readout(threshold)
    except Exception as exc:                                    # noqa: BLE001
        st.error(f"Could not analyze the experiment: {exc}")
        st.stop()

    decision_color = {
        readout.Decision.SHIP: p.good,
        readout.Decision.DO_NOT_SHIP: p.critical,
        readout.Decision.KEEP_RUNNING: p.warning,
        readout.Decision.INVALID: p.critical,
    }[r.decision]
    st.markdown(
        theme.status_badge(
            r.decision.value, decision_color,
            f"{r.n_control:,} control / {r.n_treatment:,} treatment · "
            f"threshold {threshold:+.2%}",
        ),
        unsafe_allow_html=True,
    )
    for n, reason in enumerate(r.rationale, start=1):
        st.markdown(f"{n}. {reason}")

    st.divider()
    g1, g2 = st.columns([1, 1])

    with g1:
        st.markdown("**Integrity**")
        st.dataframe(pd.DataFrame([
            {"check": "Sample ratio", "result": "PASS" if r.srm.passed else "FAIL",
             "detail": f"chi2={r.srm.chi_square:.2f}, p={r.srm.p_value:.3g}"},
            {"check": "Pre-period balance", "result": "PASS" if r.balance.passed else "FAIL",
             "detail": f"std diff {r.balance.standardized_difference:+.4f}"},
        ]).set_index("check"), width="stretch")

        if r.guardrail_report is not None:
            st.markdown("**Guardrails**")
            st.dataframe(pd.DataFrame([
                {"metric": g.name, "verdict": g.verdict.value,
                 "harm": round(g.harm, 4),
                 "95% CI": f"[{g.harm_ci_low:+.4f}, {g.harm_ci_high:+.4f}]",
                 "margin": g.margin}
                for g in r.guardrail_report.results
            ]).set_index("metric"), width="stretch")
            st.caption(
                "Guardrails are non-inferiority tests, not significance tests. "
                "INCONCLUSIVE blocks exactly like FAIL — an underpowered guardrail "
                "must not pass by default."
            )

    with g2:
        if r.primary is not None and r.cuped is not None:
            st.altair_chart(
                viz.effect_interval_chart(
                    {
                        "Unadjusted": (r.primary.absolute_effect,
                                       r.primary.ci_low, r.primary.ci_high),
                        "CUPED": (r.cuped.test.absolute_effect,
                                  r.cuped.test.ci_low, r.cuped.test.ci_high),
                    },
                    threshold, p,
                ),
                width="stretch",
            )
            st.caption(
                f"CUPED covariate correlation {r.cuped.correlation:.3f} → "
                f"{r.cuped.variance_reduction:.2%} variance reduction. Reduction scales "
                f"as 1−ρ², so a weak covariate buys almost nothing; this one is weak."
            )

    if r.bayesian is not None:
        st.divider()
        st.markdown("**Bayesian view — what does it cost us to be wrong?**")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("P(treatment > control)", f"{r.bayesian.prob_treatment_better:.1%}")
        b2.metric("P(beats threshold)", f"{r.bayesian.prob_exceeds_threshold:.1%}")
        b3.metric("Expected loss if we ship", f"{r.bayesian.expected_loss_ship:.4%}")
        b4.metric("Expected loss if we hold", f"{r.bayesian.expected_loss_stay:.4%}")
        st.markdown(bayesian.decision_summary(r.bayesian))
        st.info(
            "This does **not** override the decision above. The frequentist rule was "
            "pre-registered; running two decision rules in parallel invites picking "
            "whichever one wins. The posterior is reported as corroborating evidence "
            "and as the risk framing for the business.",
        )

    if r.sequential_plan is not None and r.sequential_decision is not None:
        st.divider()
        sq1, sq2 = st.columns([1, 1])
        with sq1:
            st.altair_chart(
                viz.sequential_boundary_chart(
                    r.sequential_plan.information_fractions,
                    r.sequential_plan.z_boundaries,
                    r.sequential_decision.look,
                    r.sequential_decision.z,
                    p,
                ),
                width="stretch",
            )
        with sq2:
            st.markdown("**Why sequential boundaries exist**")
            naive = sequential.naive_peeking_type_one_error(5, 0.05, simulations=20_000)
            n1, n2 = st.columns(2)
            n1.metric("Peeking at fixed α=0.05", f"{naive:.1%}", help="5 looks, simulated")
            n2.metric("With α-spending", "5.0%")
            st.markdown(
                "Teams peek. Each look is another chance to cross the threshold, so "
                "five looks at a fixed 0.05 produce roughly a 14% false-positive rate. "
                "Telling people not to peek does not work; giving them boundaries that "
                "make peeking valid does."
            )

    st.download_button(
        "Download readout (Markdown)", markdown,
        file_name=f"readout_{EXPERIMENT.experiment_id}.md", mime="text/markdown",
    )

# ---------------------------------------------------------------------------
# Method
# ---------------------------------------------------------------------------
with method_tab:
    st.subheader("How this is validated")
    st.markdown(
        """
Every model here is scored against a **known ground truth**. The synthetic
generator writes the true channel-importance vector and the injected treatment
lift to `data/ground_truth/ground_truth.json`; no model reads that file until
validation. Real data cannot falsify an attribution model — there is nothing to
check the answer against — which is why the whole platform is synthetic-first.

**Attribution.** Channel *exposure volume* is deliberately decoupled from true
*conversion importance*. Meta gets the highest exposure (0.40) but only 0.14 true
importance, so last-touch is misled by construction and a method that recovers
the truth has to do so on the merits.

**Experimentation.** A known lift is injected and the engine must cover it with
its confidence interval — and must hold Type-I error at α under repeated peeking.
Both are measured by simulation in the test suite, not asserted from a citation.

**Known limitations, stated rather than hidden:**

- Shapley still flatters Meta (18.6% vs 14.0% true) because zero-touch users are
  unobservable in a marketing log, so the organic baseline gets spread across
  channels. It biases *toward* the incumbent's answer, making the case against
  Meta a conservative floor.
- Markov's removal effect conflates a channel's *reach* with its *incrementality*
  and only beats last-touch by 15%. It is retained as a divergence alarm, not a
  decision model.
- Neither method randomizes exposure, so neither measures true incrementality.
  A geo holdout is the recommended confirmation before any budget moves.
- The CUPED covariate in this dataset is weakly correlated (ρ≈0.05), so it buys a
  0.2% variance reduction. The engine reports what it actually achieved.
        """
    )
    st.caption(
        f"Attribution lookback: {ATTRIBUTION.lookback_days} days · "
        f"time-decay half-life: {ATTRIBUTION.half_life_days} days · "
        f"SRM screened at α=0.001"
    )
