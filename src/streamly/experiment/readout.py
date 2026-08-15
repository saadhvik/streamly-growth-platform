"""Automated experiment readout: from warehouse rows to a ship/no-ship decision.

This module encodes the decision *procedure*, not just the statistics. The
order of the gates is the whole point:

    1. INTEGRITY   -- is this experiment valid at all?
    2. GUARDRAILS  -- did it break anything?
    3. PRIMARY     -- did it win by enough to be worth shipping?

Two properties of that ordering are deliberate and worth defending:

**The primary metric is withheld when integrity fails.** A readout that prints
"SRM detected" next to "+2.4% lift, p=0.001" will get shipped anyway; somebody
always argues the mismatch was probably harmless. Once a number is on the page
it anchors the decision. So when the integrity gate fails, this module reports
the failure and refuses to compute the headline -- the only honest readout of
an invalid experiment is that it is invalid.

**Statistical significance is not the ship criterion.** The rule is the
*practical* significance threshold: the confidence interval must clear the
smallest effect worth the cost of shipping. A significant +0.02pp is a true
finding and a bad decision. Comparing the interval against the threshold --
rather than the p-value against 0.05 -- also produces an honest CONTINUE
verdict when the data simply cannot yet tell, instead of forcing a coin-flip
into a binary.

Run:  PYTHONPATH=src python -m streamly.experiment.readout
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd

from streamly import warehouse
from streamly.config import EXPERIMENT, WAREHOUSE_PATH, ExperimentConfig
from streamly.experiment import bayesian as bayes
from streamly.experiment import guardrails as gr
from streamly.experiment import integrity, sequential
from streamly.experiment.frequentist import CupedResult, TestResult, cuped_two_sample
from streamly.experiment.frequentist import two_proportion_z_test


class Decision(str, Enum):
    """The four outcomes an experiment can have. There is no fifth."""

    SHIP = "SHIP"
    DO_NOT_SHIP = "DO NOT SHIP"
    KEEP_RUNNING = "KEEP RUNNING"
    INVALID = "INVALID — DO NOT INTERPRET"


@dataclass(frozen=True)
class ReadoutInputs:
    """Everything the decision depends on, in one auditable place."""

    experiment_id: str
    practical_threshold: float       # smallest effect worth shipping (absolute)
    alpha: float = 0.05
    guardrail_margins: dict[str, float] = field(
        default_factory=lambda: {"latency_ms": 25.0}
    )
    # Refund tolerance is declared *relative* to the control rate rather than as
    # an absolute point value. An absolute margin silently changes meaning when
    # the denominator changes: 2pp is a quadrupling of a 0.5% base rate but a
    # rounding error on a 6% one. 50% relative is the business tolerance -- a
    # half-again increase in refunds is unambiguously material -- and it is
    # stated here, before the data, for the same reason every other margin is.
    refund_margin_relative: float = 0.50
    sequential_looks: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    current_look: int = 5


@dataclass(frozen=True)
class ExperimentReadout:
    """A complete, self-justifying decision document."""

    inputs: ReadoutInputs
    generated_at: str
    n_control: int
    n_treatment: int

    srm: integrity.SrmResult
    balance: integrity.BalanceResult
    integrity_passed: bool

    guardrail_report: gr.GuardrailReport | None
    primary: TestResult | None
    cuped: CupedResult | None
    bayesian: bayes.BayesianResult | None
    sequential_plan: sequential.SequentialPlan | None
    sequential_decision: sequential.SequentialDecision | None

    decision: Decision
    rationale: tuple[str, ...]

    @property
    def shipped(self) -> bool:
        return self.decision is Decision.SHIP


def _decide(
    inputs: ReadoutInputs,
    integrity_passed: bool,
    guardrail_report: gr.GuardrailReport | None,
    primary: TestResult | None,
    seq: sequential.SequentialDecision | None,
) -> tuple[Decision, tuple[str, ...]]:
    """Apply the gates in order and record why each mattered.

    Every branch returns a rationale, because a decision doc whose reasoning
    cannot be reconstructed six months later is not a decision doc.
    """
    reasons: list[str] = []

    # Gate 1 -- validity.
    if not integrity_passed:
        reasons.append(
            "Integrity checks failed, so the assignment mechanism cannot be trusted. "
            "The primary metric is deliberately not reported: an invalid experiment "
            "has no effect size to interpret, and showing one anchors the decision."
        )
        return Decision.INVALID, tuple(reasons)
    reasons.append("Integrity checks passed: arm sizes and pre-period balance are as designed.")

    # Gate 2 -- harm.
    if guardrail_report is not None and not guardrail_report.passed:
        blocking = ", ".join(
            f"{r.name} ({r.verdict.value})" for r in guardrail_report.blocking
        )
        reasons.append(
            f"Guardrails blocking: {blocking}. A guardrail that is INCONCLUSIVE blocks "
            f"exactly like one that FAILS -- absence of evidence of harm is not "
            f"evidence of absence."
        )
        return Decision.DO_NOT_SHIP, tuple(reasons)
    reasons.append("All guardrails cleared their tolerance margins.")

    if primary is None:
        return Decision.KEEP_RUNNING, tuple(reasons + ["No primary metric available."])

    # Gate 3 -- is the win big enough to be worth shipping?
    threshold = inputs.practical_threshold
    if seq is not None and not seq.crossed and seq.look < len(inputs.sequential_looks):
        reasons.append(
            f"Sequential boundary not crossed at look {seq.look} "
            f"(|Z|={abs(seq.z):.2f} < {seq.boundary:.2f}). Stopping now would spend "
            f"more than this look's alpha budget allows."
        )
        return Decision.KEEP_RUNNING, tuple(reasons)

    if primary.ci_low > threshold:
        reasons.append(
            f"The {1 - inputs.alpha:.0%} confidence interval "
            f"[{primary.ci_low:+.4f}, {primary.ci_high:+.4f}] lies entirely above the "
            f"practical threshold of {threshold:+.4f}. The effect is not just real, "
            f"it is large enough to be worth shipping."
        )
        return Decision.SHIP, tuple(reasons)

    if primary.ci_high < threshold:
        reasons.append(
            f"The confidence interval [{primary.ci_low:+.4f}, {primary.ci_high:+.4f}] "
            f"lies entirely below the practical threshold of {threshold:+.4f}. Even the "
            f"optimistic end of the estimate is not worth the cost of shipping. This is "
            f"a decisive negative result, not a failure to detect."
        )
        return Decision.DO_NOT_SHIP, tuple(reasons)

    reasons.append(
        f"The confidence interval [{primary.ci_low:+.4f}, {primary.ci_high:+.4f}] "
        f"straddles the practical threshold of {threshold:+.4f}, so the data cannot yet "
        f"separate 'worth shipping' from 'not worth shipping'."
    )
    if seq is not None and seq.look >= len(inputs.sequential_looks):
        reasons.append(
            "The final look has been reached, so more data is not coming. Defaulting to "
            "DO NOT SHIP: the burden of proof sits with the change, not with the status quo."
        )
        return Decision.DO_NOT_SHIP, tuple(reasons)
    return Decision.KEEP_RUNNING, tuple(reasons)


def load_experiment(
    experiment_id: str = EXPERIMENT.experiment_id, path: Path | str = WAREHOUSE_PATH
) -> pd.DataFrame:
    """Read one experiment's assignment rows from the warehouse."""
    con = warehouse.connect(path, read_only=True)
    try:
        df = con.execute(
            "SELECT user_id, variant, primary_metric, pre_covariate, "
            "guardrail_refund, guardrail_latency_ms "
            "FROM experiment_assignment WHERE experiment_id = ?",
            [experiment_id],
        ).fetch_df()
    finally:
        con.close()
    if df.empty:
        raise ValueError(f"no rows found for experiment_id={experiment_id!r}")
    return df


def analyze(
    df: pd.DataFrame,
    inputs: ReadoutInputs,
    ecfg: ExperimentConfig = EXPERIMENT,
) -> ExperimentReadout:
    """Run the full gate sequence and produce a decision."""
    control = df[df["variant"] == "control"]
    treatment = df[df["variant"] == "treatment"]
    n_c, n_t = len(control), len(treatment)

    # --- Gate 1: integrity ---
    srm = integrity.srm_check(
        {"control": n_c, "treatment": n_t},
        {"control": ecfg.true_traffic_split, "treatment": 1.0 - ecfg.true_traffic_split},
    )
    balance = integrity.covariate_balance(
        control["pre_covariate"].to_numpy(), treatment["pre_covariate"].to_numpy()
    )
    integrity_passed = srm.passed and balance.passed

    if not integrity_passed:
        decision, rationale = _decide(inputs, False, None, None, None)
        return ExperimentReadout(
            inputs=inputs, generated_at=_now(), n_control=n_c, n_treatment=n_t,
            srm=srm, balance=balance, integrity_passed=False,
            guardrail_report=None, primary=None, cuped=None, bayesian=None,
            sequential_plan=None, sequential_decision=None,
            decision=decision, rationale=rationale,
        )

    # --- Gate 2: guardrails ---
    # Refunds are measured over ALL assigned users, not just converters.
    # Conditioning on conversion shrinks the denominator by an order of
    # magnitude and costs ~11.6x in precision, which is the difference between a
    # resolvable guardrail and a permanent INCONCLUSIVE. The metric still means
    # what it should -- "how often does a user assigned to this arm end up
    # refunding" -- and it is now answerable.
    refunds_c = int(control["guardrail_refund"].sum())
    refunds_t = int(treatment["guardrail_refund"].sum())
    refund_margin = inputs.refund_margin_relative * (refunds_c / n_c if n_c else 0.0)
    checks = [
        gr.guardrail_proportion(
            "refund_rate", refunds_c, n_c, refunds_t, n_t, margin=refund_margin,
        ),
        gr.guardrail_mean(
            "latency_ms",
            control["guardrail_latency_ms"].to_numpy(),
            treatment["guardrail_latency_ms"].to_numpy(),
            margin=inputs.guardrail_margins["latency_ms"],
        ),
    ]
    report = gr.evaluate_guardrails(checks)

    # --- Gate 3: primary metric ---
    primary = two_proportion_z_test(
        int(control["primary_metric"].sum()), n_c,
        int(treatment["primary_metric"].sum()), n_t,
        alpha=inputs.alpha,
    )
    cuped = cuped_two_sample(
        control["primary_metric"].to_numpy(), control["pre_covariate"].to_numpy(),
        treatment["primary_metric"].to_numpy(), treatment["pre_covariate"].to_numpy(),
        alpha=inputs.alpha,
    )

    posterior = bayes.beta_binomial_test(
        int(control["primary_metric"].sum()), n_c,
        int(treatment["primary_metric"].sum()), n_t,
        threshold=inputs.practical_threshold,
        credible_level=1.0 - inputs.alpha,
    )

    plan = sequential.compute_boundaries(inputs.sequential_looks, inputs.alpha)
    seq = sequential.evaluate_look(plan, inputs.current_look, primary.statistic)

    decision, rationale = _decide(inputs, True, report, primary, seq)
    return ExperimentReadout(
        inputs=inputs, generated_at=_now(), n_control=n_c, n_treatment=n_t,
        srm=srm, balance=balance, integrity_passed=True,
        guardrail_report=report, primary=primary, cuped=cuped, bayesian=posterior,
        sequential_plan=plan, sequential_decision=seq,
        decision=decision, rationale=rationale,
    )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def to_markdown(r: ExperimentReadout, true_lift: float | None = None) -> str:
    """Render the readout as the decision document a review meeting reads."""
    i = r.inputs
    lines = [
        f"# Experiment Readout — `{i.experiment_id}`",
        "",
        f"**Decision: {r.decision.value}**",
        "",
        f"*Generated {r.generated_at} · alpha={i.alpha} · "
        f"practical threshold {i.practical_threshold:+.2%}*",
        "",
        "## Why",
        "",
    ]
    lines += [f"{n}. {reason}" for n, reason in enumerate(r.rationale, start=1)]
    lines += [
        "",
        "## 1. Integrity — is this experiment valid?",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
        f"| Sample ratio | {'PASS' if r.srm.passed else 'FAIL'} | "
        f"{r.n_control:,} control / {r.n_treatment:,} treatment; "
        f"chi2={r.srm.chi_square:.2f}, p={r.srm.p_value:.3g} vs alpha={r.srm.alpha} |",
        f"| Pre-period balance | {'PASS' if r.balance.passed else 'FAIL'} | "
        f"std diff {r.balance.standardized_difference:+.4f}, p={r.balance.p_value:.3g} |",
        "",
        f"SRM is screened at alpha={r.srm.alpha} rather than 0.05 because it runs on every "
        "experiment; at 0.05 one healthy experiment in twenty would be flagged and the "
        "alarm would stop being believed. At this sample size the check cannot detect an "
        f"arm loss below "
        f"{integrity.srm_minimum_detectable_loss(max(r.n_control, 1)):.2%} — a pass is "
        "evidence, not a guarantee.",
        "",
    ]

    if not r.integrity_passed:
        lines += [
            "## 2. Primary metric — withheld",
            "",
            "The integrity gate failed, so no effect size is reported. This is "
            "deliberate: a number printed next to a failed validity check gets used "
            "anyway. Fix the assignment pipeline and rerun.",
            "",
        ]
        return "\n".join(lines)

    assert r.guardrail_report is not None and r.primary is not None
    assert r.cuped is not None and r.sequential_decision is not None

    lines += [
        "## 2. Guardrails — did it break anything?",
        "",
        "| Metric | Verdict | Control | Treatment | Harm | 95% CI | Margin | "
        "Smallest resolvable margin |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for g in r.guardrail_report.results:
        power_note = "" if g.adequately_powered else " ⚠"
        lines.append(
            f"| {g.name} | **{g.verdict.value}** | {g.control_value:.4f} | "
            f"{g.treatment_value:.4f} | {g.harm:+.4f} | "
            f"[{g.harm_ci_low:+.4f}, {g.harm_ci_high:+.4f}] | {g.margin:.4f} | "
            f"{g.resolvable_margin:.4f}{power_note} |"
        )

    refund = next((g for g in r.guardrail_report.results if g.name == "refund_rate"), None)
    lines += [
        "",
        "Guardrails are non-inferiority tests against a pre-agreed tolerance, not "
        "significance tests. `harm` is signed so positive always means degradation. "
        "An INCONCLUSIVE verdict blocks the launch exactly like a FAIL — under a plain "
        "significance test an underpowered guardrail would have passed silently.",
        "",
        "**Denominator and margin, both declared before the data.** Refunds are measured "
        "over *all assigned users*, not just converters. Conditioning on conversion "
        "shrinks the denominator by an order of magnitude and costs roughly 11.6x in "
        "precision — enough to make this guardrail permanently INCONCLUSIVE and block "
        "every launch regardless of the treatment. The tolerance is "
        f"{r.inputs.refund_margin_relative:.0%} relative to the control rate, stated as a "
        "business rule rather than an absolute point value, because an absolute margin "
        "silently changes meaning when the denominator does.",
        "",
    ]
    if refund is not None:
        tighter = 0.25
        need = gr.required_n_for_margin(refund.control_value, tighter * refund.control_value)
        lines += [
            "**How much this verdict depends on the margin — stated, not buried.** The "
            f"smallest tolerance this sample can resolve is {refund.resolvable_margin:.4f}; "
            f"the declared margin of {refund.margin:.4f} sits above it, so the guardrail "
            "was answerable rather than merely lucky. At a stricter "
            f"{tighter:.0%} relative tolerance ({tighter * refund.control_value:.4f}) the "
            f"verdict would be INCONCLUSIVE and the check would need ~{need:,} users per "
            "arm instead of "
            f"{r.n_control:,}. Readers who hold a tighter view of acceptable refund drift "
            "should read this guardrail as unresolved, not as passed.",
            "",
        ]

    lines += [
        "## 3. Primary metric — did it win by enough?",
        "",
        "| | Control | Treatment | Absolute | Relative | 95% CI | p |",
        "|---|---:|---:|---:|---:|---|---:|",
        f"| Unadjusted | {r.primary.control_mean:.4f} | {r.primary.treatment_mean:.4f} | "
        f"{r.primary.absolute_effect:+.4f} | {r.primary.relative_effect:+.2%} | "
        f"[{r.primary.ci_low:+.4f}, {r.primary.ci_high:+.4f}] | {r.primary.p_value:.3g} |",
        f"| CUPED | {r.cuped.test.control_mean:.4f} | {r.cuped.test.treatment_mean:.4f} | "
        f"{r.cuped.test.absolute_effect:+.4f} | {r.cuped.test.relative_effect:+.2%} | "
        f"[{r.cuped.test.ci_low:+.4f}, {r.cuped.test.ci_high:+.4f}] | "
        f"{r.cuped.test.p_value:.3g} |",
        "",
        f"CUPED covariate correlation is {r.cuped.correlation:.3f}, giving a "
        f"{r.cuped.variance_reduction:.2%} variance reduction "
        f"(theory: rho^2 = {r.cuped.theoretical_reduction:.2%}) and a "
        f"{r.cuped.ci_width_reduction:.2%} narrower interval. Variance reduction scales "
        "as 1 - rho^2, so a weak covariate buys almost nothing; this one is weak and the "
        "readout says so rather than implying CUPED did work it did not do.",
        "",
        "## 4. Stopping rule — was it valid to look now?",
        "",
        f"Look {r.sequential_decision.look} of {len(r.inputs.sequential_looks)}: "
        f"|Z| = {abs(r.sequential_decision.z):.3f} against a boundary of "
        f"{r.sequential_decision.boundary:.3f} "
        f"({'crossed' if r.sequential_decision.crossed else 'not crossed'}).",
        "",
        "Boundaries come from O'Brien-Fleming alpha spending. Peeking five times at a "
        "fixed alpha=0.05 inflates the false-positive rate to roughly 14%; these "
        "boundaries hold it at 5% (both measured by simulation in "
        "`tests/test_srm_and_sequential.py`).",
        "",
    ]

    if r.bayesian is not None:
        b = r.bayesian
        # The two frameworks answer different questions; when they disagree the
        # disagreement is the finding, so it is surfaced rather than smoothed.
        freq_clears = r.primary.ci_low > i.practical_threshold
        bayes_clears = b.prob_exceeds_threshold >= 1.0 - i.alpha
        lines += [
            "## 5. Bayesian view — what does it cost us to be wrong?",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| P(treatment > control) | {b.prob_treatment_better:.2%} |",
            f"| P(lift > {b.threshold:+.2%} threshold) | {b.prob_exceeds_threshold:.2%} |",
            f"| Expected loss **if we ship** | {b.expected_loss_ship:.4%} |",
            f"| Expected loss **if we hold** | {b.expected_loss_stay:.4%} |",
            f"| {b.credible_level:.0%} credible interval on the lift | "
            f"[{b.lift_ci_low:+.4f}, {b.lift_ci_high:+.4f}] |",
            "",
            bayes.decision_summary(b),
            "",
            f"Posteriors are exact Beta distributions from a Beta{b.prior} prior "
            "(uniform, worth two observations against tens of thousands, so it is "
            "effectively invisible here). Every quantity is computed by numerical "
            "integration of those posteriors rather than by MCMC, so the figures are "
            "deterministic to integration tolerance and reproduce exactly on rerun.",
            "",
            "**Expected loss is the number to decide on.** A win probability near 100% "
            "still says nothing about magnitude; expected loss prices the downside "
            "directly. Note that the cost of *holding* is the error teams "
            "systematically ignore — here it is "
            f"{b.expected_loss_stay:.4%} of conversion rate against a cost of shipping "
            + ("that rounds to zero at this sample size."
               if b.expected_loss_ship < 1e-6 else
               f"of {b.expected_loss_ship:.4%}."),
            "",
            "> **This does not override the decision above.** The frequentist rule was "
            "pre-registered, and running two decision rules in parallel invites picking "
            "whichever one wins. The Bayesian layer is reported as corroborating "
            "evidence and as the risk framing for the business, not as a second "
            "verdict.",
            "",
            f"Agreement check: the frequentist interval "
            f"{'clears' if freq_clears else 'does not clear'} the threshold and the "
            f"posterior {'clears' if bayes_clears else 'does not clear'} it — "
            f"**{'the two frameworks agree' if freq_clears == bayes_clears else 'THE TWO FRAMEWORKS DISAGREE; investigate before acting'}**.",
            "",
        ]

    if true_lift is not None:
        covered = r.primary.ci_low <= true_lift <= r.primary.ci_high
        bayes_covered = (
            r.bayesian is not None
            and r.bayesian.lift_ci_low <= true_lift <= r.bayesian.lift_ci_high
        )
        lines += [
            "## 6. Validation against ground truth",
            "",
            f"This experiment was generated with a known injected lift of "
            f"**{true_lift:+.2%}**. The 95% interval "
            f"[{r.primary.ci_low:+.4f}, {r.primary.ci_high:+.4f}] "
            f"**{'covers' if covered else 'does NOT cover'}** it. The Bayesian credible "
            f"interval [{r.bayesian.lift_ci_low:+.4f}, {r.bayesian.lift_ci_high:+.4f}] "
            f"**{'covers' if bayes_covered else 'does NOT cover'}** it as well."
            if r.bayesian is not None else
            f"**{'covers' if covered else 'does NOT cover'}** it.",
            "",
            "Only synthetic data can falsify an experimentation engine this way: on real "
            "data there is no truth to check the interval against.",
            "",
        ]

    lines += [
        "## Appendix — the decision rule, stated in advance",
        "",
        "1. **Integrity gate.** SRM or balance failure ⇒ INVALID; the primary metric is "
        "not reported at all.",
        "2. **Guardrail gate.** Any guardrail FAIL or INCONCLUSIVE ⇒ DO NOT SHIP.",
        "3. **Practical significance.** CI entirely above the threshold ⇒ SHIP. CI "
        "entirely below ⇒ DO NOT SHIP. CI straddling the threshold ⇒ KEEP RUNNING, or "
        "DO NOT SHIP if the final look has been reached.",
        "",
        "The threshold, not the p-value, is the ship criterion. A statistically "
        "significant effect below the threshold is a true finding and a bad decision.",
        "",
    ]
    return "\n".join(lines)


def generate(
    experiment_id: str = EXPERIMENT.experiment_id,
    practical_threshold: float = 0.005,
    path: Path | str = WAREHOUSE_PATH,
) -> tuple[ExperimentReadout, str]:
    """Load, analyze, and render -- the whole pipeline in one call."""
    df = load_experiment(experiment_id, path)
    inputs = ReadoutInputs(experiment_id=experiment_id, practical_threshold=practical_threshold)
    readout = analyze(df, inputs)

    true_lift = None
    try:
        import json

        from streamly.config import GROUND_TRUTH_DIR

        with open(GROUND_TRUTH_DIR / "ground_truth.json") as f:
            true_lift = json.load(f)["experiment"]["true_treatment_lift_abs"]
    except (OSError, KeyError):
        pass  # ground truth is optional; real experiments have none

    return readout, to_markdown(readout, true_lift)


def _main() -> None:
    readout, md = generate()
    out = Path(__file__).resolve().parents[3] / "docs" / "experiment_readout_example.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    print(f"Decision: {readout.decision.value}")
    for n, reason in enumerate(readout.rationale, start=1):
        print(f"  {n}. {reason}")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    _main()
