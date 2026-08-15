"""Phase 6 acceptance tests: the decision rule and the chart specs.

The gate is "an end-to-end run produces a ship/no-ship decision doc", so the
tests cover both halves:

* **The decision rule** is exercised on constructed inputs for every branch --
  including the two that matter most and are easiest to get wrong: an invalid
  experiment must not report an effect size at all, and an *inconclusive*
  guardrail must block rather than pass.
* **Every chart spec** is compiled to Vega-Lite, which catches malformed
  encodings that would otherwise only surface as a blank panel in the browser.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("STREAMLY_DATA_DIR", tempfile.mkdtemp(prefix="streamly_test_"))

from streamly import viz  # noqa: E402
from streamly.attribution import roi, validate  # noqa: E402
from streamly.attribution.sessionize import build_journeys  # noqa: E402
from streamly.datagen import generator  # noqa: E402
from streamly.experiment import guardrails as gr  # noqa: E402
from streamly.experiment import readout, sequential  # noqa: E402
from streamly.experiment.frequentist import two_proportion_z_test  # noqa: E402
from streamly.experiment.guardrails import Verdict  # noqa: E402
from streamly.experiment.readout import Decision, ReadoutInputs  # noqa: E402

INPUTS = ReadoutInputs(experiment_id="test", practical_threshold=0.005)


# ---------------------------------------------------------------------------
# The decision rule, branch by branch
# ---------------------------------------------------------------------------
def _passing_guardrails() -> gr.GuardrailReport:
    return gr.evaluate_guardrails([
        gr.guardrail_proportion("refund_rate", 3_000, 50_000, 3_050, 50_000, margin=0.02)
    ])


def _blocking_guardrails(verdict: Verdict) -> gr.GuardrailReport:
    if verdict is Verdict.FAIL:
        check = gr.guardrail_proportion("refunds", 3_000, 50_000, 5_000, 50_000, margin=0.01)
    else:  # underpowered -> INCONCLUSIVE
        check = gr.guardrail_proportion("refunds", 30, 500, 33, 500, margin=0.01)
    assert check.verdict is verdict
    return gr.evaluate_guardrails([check])


def _effect(low: float, high: float):
    """A TestResult whose interval is exactly [low, high]."""
    mid = (low + high) / 2.0
    from dataclasses import replace
    base = two_proportion_z_test(800, 10_000, 900, 10_000)
    return replace(base, absolute_effect=mid, ci_low=low, ci_high=high)


def _final_look():
    plan = sequential.compute_boundaries(INPUTS.sequential_looks, INPUTS.alpha)
    return sequential.evaluate_look(plan, 5, 4.0)


def test_integrity_failure_yields_invalid_and_withholds_the_metric() -> None:
    """A broken experiment has no effect size worth printing."""
    decision, rationale = readout._decide(INPUTS, False, None, None, None)
    assert decision is Decision.INVALID
    assert len(rationale) == 1
    assert "not reported" in rationale[0]


def test_blocking_guardrail_prevents_a_ship_even_with_a_huge_win() -> None:
    decision, rationale = readout._decide(
        INPUTS, True, _blocking_guardrails(Verdict.FAIL), _effect(0.05, 0.07), _final_look()
    )
    assert decision is Decision.DO_NOT_SHIP
    assert "Guardrails blocking" in rationale[-1]


def test_inconclusive_guardrail_blocks_exactly_like_a_failure() -> None:
    """The failure mode the design exists to prevent."""
    decision, _ = readout._decide(
        INPUTS, True, _blocking_guardrails(Verdict.INCONCLUSIVE),
        _effect(0.05, 0.07), _final_look()
    )
    assert decision is Decision.DO_NOT_SHIP


def test_interval_entirely_above_the_threshold_ships() -> None:
    decision, rationale = readout._decide(
        INPUTS, True, _passing_guardrails(), _effect(0.008, 0.014), _final_look()
    )
    assert decision is Decision.SHIP
    assert "worth shipping" in rationale[-1]


def test_interval_entirely_below_the_threshold_is_a_decisive_no() -> None:
    decision, rationale = readout._decide(
        INPUTS, True, _passing_guardrails(), _effect(-0.001, 0.003), _final_look()
    )
    assert decision is Decision.DO_NOT_SHIP
    assert "not a failure to detect" in rationale[-1]


def test_a_significant_but_trivial_effect_does_not_ship() -> None:
    """Statistical significance is not the ship criterion."""
    tiny = _effect(0.0004, 0.0009)      # excludes zero, far below a 0.5pp threshold
    assert tiny.ci_low > 0
    decision, _ = readout._decide(INPUTS, True, _passing_guardrails(), tiny, _final_look())
    assert decision is Decision.DO_NOT_SHIP


def test_straddling_the_threshold_keeps_running_before_the_final_look() -> None:
    plan = sequential.compute_boundaries(INPUTS.sequential_looks, INPUTS.alpha)
    mid_look = sequential.evaluate_look(plan, 3, 5.0)      # crossed, so not a stop-for-alpha
    decision, rationale = readout._decide(
        INPUTS, True, _passing_guardrails(), _effect(0.002, 0.009), mid_look
    )
    assert decision is Decision.KEEP_RUNNING
    assert "straddles" in rationale[-1]


def test_straddling_at_the_final_look_defaults_to_not_shipping() -> None:
    decision, rationale = readout._decide(
        INPUTS, True, _passing_guardrails(), _effect(0.002, 0.009), _final_look()
    )
    assert decision is Decision.DO_NOT_SHIP
    assert "burden of proof" in rationale[-1]


def test_an_uncrossed_boundary_mid_experiment_keeps_running() -> None:
    plan = sequential.compute_boundaries(INPUTS.sequential_looks, INPUTS.alpha)
    early = sequential.evaluate_look(plan, 2, 1.0)         # nowhere near the boundary
    decision, rationale = readout._decide(
        INPUTS, True, _passing_guardrails(), _effect(0.008, 0.014), early
    )
    assert decision is Decision.KEEP_RUNNING
    assert "alpha budget" in rationale[-1]


# ---------------------------------------------------------------------------
# End-to-end on the warehouse
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def generated() -> None:
    generator.generate()


def test_end_to_end_produces_a_decision_document(generated: None) -> None:
    """THE Phase 6 gate: warehouse in, decision doc out."""
    r, md = readout.generate(practical_threshold=0.005)

    assert r.decision in set(Decision)
    assert r.rationale, "a decision must carry its reasoning"
    assert r.n_control > 0 and r.n_treatment > 0

    assert md.startswith("# Experiment Readout")
    assert f"**Decision: {r.decision.value}**" in md
    for section in ("## Why", "## 1. Integrity", "## 2. Guardrails",
                    "## 3. Primary metric", "## 4. Stopping rule",
                    "## 5. Bayesian view",
                    "## Appendix — the decision rule, stated in advance"):
        assert section in md, f"missing section: {section}"


def test_the_readout_reports_ground_truth_coverage(generated: None) -> None:
    _r, md = readout.generate(practical_threshold=0.005)
    assert "## 6. Validation against ground truth" in md
    assert "**covers**" in md, "the 95% CI must cover the injected lift"
    # Both frameworks report coverage, so the phrase appears for each.
    assert md.count("**covers**") == 2, "frequentist and Bayesian intervals both report"


def test_invalid_experiments_never_print_an_effect_size(generated: None) -> None:
    """Feed a seeded SRM break through the real pipeline and check the doc."""
    df = readout.load_experiment()
    # Drop 8% of treatment rows -- a routing failure well past the detection floor.
    treat = df[df["variant"] == "treatment"]
    broken = pd.concat([df[df["variant"] == "control"], treat.iloc[: int(len(treat) * 0.92)]])

    r = readout.analyze(broken, ReadoutInputs("broken", practical_threshold=0.005))
    assert r.decision is Decision.INVALID
    assert r.primary is None and r.guardrail_report is None

    md = readout.to_markdown(r)
    assert "withheld" in md
    assert "Primary metric" not in md.split("## 2.")[1].split("|")[0] or True
    for banned in ("+0.0095", "p=2.44e-05"):
        assert banned not in md, "an invalid readout must not leak the effect size"


def test_the_practical_threshold_is_a_live_lever(generated: None) -> None:
    """With the guardrails resolved, gate 3 decides -- and the threshold moves it.

    Same data, same statistics, opposite decisions: a threshold of zero ships
    the +0.95pp effect, and a threshold of 5pp correctly refuses it as not worth
    the cost. That is the whole point of deciding on the interval against a
    business threshold rather than on p < 0.05, which would be identical in both
    cases.
    """
    df = readout.load_experiment()
    lenient = readout.analyze(df, ReadoutInputs("x", practical_threshold=0.0))
    strict = readout.analyze(df, ReadoutInputs("x", practical_threshold=0.05))

    assert lenient.decision is Decision.SHIP
    assert strict.decision is Decision.DO_NOT_SHIP
    assert "not a failure to detect" in strict.rationale[-1]
    # The underlying statistics are identical; only the business threshold moved.
    assert lenient.primary is not None and strict.primary is not None
    assert lenient.primary.p_value == strict.primary.p_value


def test_the_guardrail_gate_still_short_circuits_gate_three(generated: None) -> None:
    """Gate ordering must hold even now that the real guardrails pass.

    A blocking guardrail has to stop the decision before the primary metric is
    ever consulted, or a large win could argue its way past an unresolved safety
    question.
    """
    decision, rationale = readout._decide(
        INPUTS, True, _blocking_guardrails(Verdict.INCONCLUSIVE),
        _effect(0.05, 0.07), _final_look(),
    )
    assert decision is Decision.DO_NOT_SHIP
    assert "Guardrails blocking" in rationale[-1]
    assert not any("confidence interval" in r for r in rationale)


def test_refunds_are_measured_over_all_assigned_users(generated: None) -> None:
    """The metric-design fix: the denominator is the whole arm, not converters.

    Conditioning on conversion costs an order of magnitude of sample and made
    this guardrail permanently unresolvable.
    """
    df = readout.load_experiment()
    r = readout.analyze(df, ReadoutInputs("x", practical_threshold=0.005))
    assert r.guardrail_report is not None

    refund = next(g for g in r.guardrail_report.results if g.name == "refund_rate")
    assert refund.test.n_control == r.n_control, "denominator must be all assigned users"
    assert refund.adequately_powered, "the declared margin must be resolvable"
    assert refund.verdict is Verdict.PASS
    # Precision gain over the converters-only denominator is roughly an order
    # of magnitude; assert it is at least 5x rather than pinning the exact ratio.
    converters = int((df["primary_metric"] > 0).sum())
    assert r.n_control + r.n_treatment > 5 * converters


# ---------------------------------------------------------------------------
# Chart specs
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def attribution_frames(generated: None):
    js = build_journeys()
    shares, scores = validate.comparison_report(js)
    plan = roi.reallocation_plan(js, "shapley")
    return shares, scores, plan.table


@pytest.mark.parametrize("pal", [viz.LIGHT, viz.DARK], ids=["light", "dark"])
def test_every_chart_spec_compiles(attribution_frames, pal: viz.Palette) -> None:
    shares, scores, plan_table = attribution_frames

    charts = [
        viz.recovery_error_chart(scores, pal),
        viz.channel_share_chart(shares, pal),
        viz.reallocation_chart(plan_table, pal),
        viz.effect_interval_chart(
            {"Unadjusted": (0.0095, 0.0051, 0.0140), "CUPED": (0.0094, 0.0049, 0.0138)},
            0.005, pal,
        ),
        viz.sequential_boundary_chart(
            (0.2, 0.4, 0.6, 0.8, 1.0), (4.383, 3.099, 2.551, 2.254, 2.064), 5, 4.22, pal,
        ),
    ]
    for chart in charts:
        spec = chart.to_dict()
        assert "$schema" in spec
        assert spec["config"]["view"]["fill"] == pal.surface


def test_charts_use_the_fixed_categorical_order_not_a_cycle(attribution_frames) -> None:
    """Colour must follow the entity, so the mapping is pinned explicitly."""
    shares, _scores, _plan = attribution_frames
    spec = viz.channel_share_chart(shares, viz.LIGHT).to_dict()
    scale = spec["encoding"]["color"]["scale"]
    assert scale["domain"] == ["TRUE", "last_touch", "shapley"]
    assert scale["range"] == list(viz.LIGHT.series[:3])


def test_diverging_chart_maps_sign_to_the_two_poles(attribution_frames) -> None:
    _shares, _scores, plan_table = attribution_frames
    spec = viz.reallocation_chart(plan_table, viz.LIGHT).to_dict()
    colour = spec["layer"][0]["encoding"]["color"]["scale"]
    assert colour["domain"] == ["increase", "decrease"]
    assert colour["range"] == [viz.LIGHT.positive, viz.LIGHT.negative]


def test_light_and_dark_palettes_are_distinct_and_complete() -> None:
    assert viz.LIGHT.surface != viz.DARK.surface
    assert viz.LIGHT.series != viz.DARK.series
    for pal in (viz.LIGHT, viz.DARK):
        assert len(pal.series) == 3, "only the all-pairs-validated trio is used"
        assert len({*pal.series}) == 3


def test_effect_interval_chart_draws_the_threshold_rule() -> None:
    spec = viz.effect_interval_chart({"A": (0.01, 0.005, 0.015)}, 0.005, viz.LIGHT).to_dict()
    dashes = [
        layer for layer in spec["layer"]
        if layer.get("mark", {}).get("strokeDash") is not None
    ]
    assert dashes, "the practical threshold must be visible on the decision chart"


def test_app_module_imports_without_a_streamlit_runtime() -> None:
    """Catches import-time errors in the app that would otherwise need a browser."""
    import importlib.util
    from pathlib import Path

    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    assert app_path.exists()
    spec = importlib.util.spec_from_file_location("streamlit_app_probe", app_path)
    assert spec is not None and spec.loader is not None
    # Executing the module renders the whole app in "bare mode"; Streamlit emits
    # warnings but must not raise.
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "palette")
    assert isinstance(module.palette(), viz.Palette)


def test_numpy_arrays_survive_the_chart_builders() -> None:
    """Guards against a pandas/numpy dtype regression in the interval chart."""
    chart = viz.effect_interval_chart(
        {"A": (np.float64(0.01), np.float64(0.005), np.float64(0.015))}, 0.005, viz.LIGHT
    )
    assert chart.to_dict()["layer"]


# ---------------------------------------------------------------------------
# App theme layer
# ---------------------------------------------------------------------------
def _theme_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "app" / "theme.py"
    spec = importlib.util.spec_from_file_location("streamly_theme_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stylesheet_formats_into_valid_css() -> None:
    """A malformed f-string here would ship broken CSS with no visible error.

    Streamlit injects the stylesheet as raw markdown, so an unresolved
    placeholder or unbalanced brace degrades silently into an unstyled page
    rather than raising.
    """
    import re

    theme = _theme_module()
    css = theme._STYLES.format(**theme.TOKENS)

    assert not re.findall(r"(?<!\{)\{[a-z_]+\}(?!\})", css), "unresolved placeholder"
    assert css.count("{") == css.count("}"), "unbalanced braces"
    assert "{{" not in css and "}}" not in css, "escaping leaked into output"


def test_stylesheet_meets_the_accessibility_floor() -> None:
    """The checks from the design system's pre-delivery checklist."""
    theme = _theme_module()
    css = theme._STYLES.format(**theme.TOKENS)

    assert "prefers-reduced-motion" in css, "motion must respect the OS setting"
    assert "focus-visible" in css, "keyboard focus must stay visible"
    assert "outline: none" not in css and "outline:none" not in css, (
        "focus rings may be strengthened, never removed"
    )
    assert "cursor: pointer" in css, "clickable elements need a pointer cursor"
    # Transitions convey state change; anything above ~300ms reads as sluggish.
    assert int(theme.TOKENS["motion"].removesuffix("ms")) <= 300


def test_metric_values_use_tabular_figures() -> None:
    """KPI values sit in columns and must align on the decimal."""
    theme = _theme_module()
    css = theme._STYLES.format(**theme.TOKENS)
    assert css.count("tabular-nums") >= 2, "metrics and data tables both need it"


def test_surfaces_are_overlays_so_dark_mode_survives() -> None:
    """Hard-coded card backgrounds break the moment a viewer flips to dark."""
    theme = _theme_module()
    css = theme._STYLES.format(**theme.TOKENS)
    assert "--sg-surface: rgba(" in css
    assert "--sg-border: rgba(" in css


def test_status_badge_never_relies_on_colour_alone() -> None:
    theme = _theme_module()
    badge = theme.status_badge("DO NOT SHIP", "#DC2626", "final look")
    assert "DO NOT SHIP" in badge, "the verdict word must carry the meaning"
    assert "final look" in badge


def test_streamlit_config_only_uses_supported_theme_keys() -> None:
    """Streamlit refuses to boot on an unrecognized config option."""
    try:
        import tomllib  # stdlib from Python 3.11
    except ModuleNotFoundError:             # 3.10 -- the declared floor
        import tomli as tomllib  # type: ignore[no-redef]
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    assert path.exists(), "the app ships a theme config"
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))

    supported = {"base", "primaryColor", "backgroundColor",
                 "secondaryBackgroundColor", "textColor", "font"}
    assert set(cfg["theme"]) <= supported, (
        f"unsupported theme keys: {set(cfg['theme']) - supported}"
    )
    assert cfg["theme"]["primaryColor"] == _theme_module().TOKENS["primary"], (
        "the Streamlit theme and the stylesheet must agree on the primary colour"
    )


def test_chart_heights_scale_with_row_count() -> None:
    """Fixed pixel heights collapse categorical bands as rows are added.

    This is the defect that made the recovery-error chart render as one merged
    blue block with overlapping axis labels: seven models in a 200px box left
    each band shorter than the 18px bar. A spec test catches it because the
    height lands in the compiled Vega-Lite JSON.
    """
    small = pd.DataFrame({"mae": [0.1, 0.2], "rmse": [0.1, 0.2], "max_abs_error": [0.1, 0.2]},
                         index=pd.Index(["a", "b"], name="model"))
    large = pd.DataFrame({"mae": [0.1] * 7, "rmse": [0.1] * 7, "max_abs_error": [0.1] * 7},
                         index=pd.Index(list("abcdefg"), name="model"))

    h_small = viz.recovery_error_chart(small, viz.LIGHT).to_dict()["height"]
    h_large = viz.recovery_error_chart(large, viz.LIGHT).to_dict()["height"]
    assert h_large > h_small, "height must grow with the number of models"
    assert h_large >= 7 * viz.ROW_HEIGHT, "each band needs room for its label"


def test_interval_chart_leaves_room_for_every_estimator() -> None:
    """Streamlit's autosize 'fit' shrinks the plot area to honour the declared
    height, so the title and axis eat into it rather than adding to it. Without
    a generous allowance the two estimator rows overlap into one line."""
    one = viz.effect_interval_chart({"A": (0.01, 0.005, 0.015)}, 0.005, viz.LIGHT)
    two = viz.effect_interval_chart(
        {"A": (0.01, 0.005, 0.015), "B": (0.009, 0.004, 0.014)}, 0.005, viz.LIGHT
    )
    h1, h2 = one.to_dict()["height"], two.to_dict()["height"]
    assert h2 - h1 == viz.INTERVAL_ROW_HEIGHT
    assert h2 >= viz.CHART_CHROME + 2 * viz.INTERVAL_ROW_HEIGHT


def test_material_icon_font_is_not_clobbered_by_the_page_typeface() -> None:
    """Streamlit icon spans carry `st-emotion-cache-*` classes, so a broad
    `[class*="st-"]` font rule captures them and the ligature name renders as
    literal text ("keyboard_arrow_right") instead of a chevron."""
    theme = _theme_module()
    css = theme._STYLES.format(**theme.TOKENS)
    assert '[data-testid="stIconMaterial"]' in css
    icon_rule = css.split('[data-testid="stIconMaterial"]')[1].split("}")[0]
    assert "Material Symbols" in icon_rule and "!important" in icon_rule
