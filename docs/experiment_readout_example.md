# Experiment Readout — `annual_paywall_v1`

**Decision: SHIP**

*Generated 2026-08-10 01:40 UTC · alpha=0.05 · practical threshold +0.50%*

## Why

1. Integrity checks passed: arm sizes and pre-period balance are as designed.
2. All guardrails cleared their tolerance margins.
3. The 95% confidence interval [+0.0051, +0.0140] lies entirely above the practical threshold of +0.0050. The effect is not just real, it is large enough to be worth shipping.

## 1. Integrity — is this experiment valid?

| Check | Result | Detail |
|---|---|---|
| Sample ratio | PASS | 29,967 control / 30,033 treatment; chi2=0.07, p=0.788 vs alpha=0.001 |
| Pre-period balance | PASS | std diff +0.0153, p=0.0604 |

SRM is screened at alpha=0.001 rather than 0.05 because it runs on every experiment; at 0.05 one healthy experiment in twenty would be flagged and the alarm would stop being believed. At this sample size the check cannot detect an arm loss below 2.67% — a pass is evidence, not a guarantee.

## 2. Guardrails — did it break anything?

| Metric | Verdict | Control | Treatment | Harm | 95% CI | Margin | Smallest resolvable margin |
|---|---|---:|---:|---:|---|---:|---:|
| refund_rate | **PASS** | 0.0050 | 0.0062 | +0.0012 | [+0.0000, +0.0024] | 0.0025 | 0.0017 |
| latency_ms | **PASS** | 220.3320 | 223.7940 | +3.4620 | [+2.9820, +3.9420] | 25.0000 | 0.6861 |

Guardrails are non-inferiority tests against a pre-agreed tolerance, not significance tests. `harm` is signed so positive always means degradation. An INCONCLUSIVE verdict blocks the launch exactly like a FAIL — under a plain significance test an underpowered guardrail would have passed silently.

**Denominator and margin, both declared before the data.** Refunds are measured over *all assigned users*, not just converters. Conditioning on conversion shrinks the denominator by an order of magnitude and costs roughly 11.6x in precision — enough to make this guardrail permanently INCONCLUSIVE and block every launch regardless of the treatment. The tolerance is 50% relative to the control rate, stated as a business rule rather than an absolute point value, because an absolute margin silently changes meaning when the denominator does.

**How much this verdict depends on the margin — stated, not buried.** The smallest tolerance this sample can resolve is 0.0017; the declared margin of 0.0025 sits above it, so the guardrail was answerable rather than merely lucky. At a stricter 25% relative tolerance (0.0012) the verdict would be INCONCLUSIVE and the check would need ~50,264 users per arm instead of 29,967. Readers who hold a tighter view of acceptable refund drift should read this guardrail as unresolved, not as passed.

## 3. Primary metric — did it win by enough?

| | Control | Treatment | Absolute | Relative | 95% CI | p |
|---|---:|---:|---:|---:|---|---:|
| Unadjusted | 0.0791 | 0.0886 | +0.0095 | +12.08% | [+0.0051, +0.0140] | 2.44e-05 |
| CUPED | 0.0792 | 0.0885 | +0.0094 | +11.82% | [+0.0049, +0.0138] | 3.49e-05 |

CUPED covariate correlation is 0.046, giving a 0.21% variance reduction (theory: rho^2 = 0.21%) and a 0.10% narrower interval. Variance reduction scales as 1 - rho^2, so a weak covariate buys almost nothing; this one is weak and the readout says so rather than implying CUPED did work it did not do.

## 4. Stopping rule — was it valid to look now?

Look 5 of 5: |Z| = 4.220 against a boundary of 2.064 (crossed).

Boundaries come from O'Brien-Fleming alpha spending. Peeking five times at a fixed alpha=0.05 inflates the false-positive rate to roughly 14%; these boundaries hold it at 5% (both measured by simulation in `tests/test_srm_and_sequential.py`).

## 5. Bayesian view — what does it cost us to be wrong?

| Quantity | Value |
|---|---:|
| P(treatment > control) | 100.00% |
| P(lift > +0.50% threshold) | 97.78% |
| Expected loss **if we ship** | 0.0000% |
| Expected loss **if we hold** | 0.9548% |
| 95% credible interval on the lift | [+0.0051, +0.0140] |

There is a 100.0% probability the treatment is better, and a 97.8% probability it beats the +0.50% threshold that makes shipping worthwhile. If we ship and are wrong, the expected cost is 0.0000% of conversion rate; if we hold and are wrong, we forgo 0.9548%. Against a tolerance of 0.0500%, shipping is the lower-risk choice.

Posteriors are exact Beta distributions from a Beta(1.0, 1.0) prior (uniform, worth two observations against tens of thousands, so it is effectively invisible here). Every quantity is computed by numerical integration of those posteriors rather than by MCMC, so the figures are deterministic to integration tolerance and reproduce exactly on rerun.

**Expected loss is the number to decide on.** A win probability near 100% still says nothing about magnitude; expected loss prices the downside directly. Note that the cost of *holding* is the error teams systematically ignore — here it is 0.9548% of conversion rate against a cost of shipping that rounds to zero at this sample size.

> **This does not override the decision above.** The frequentist rule was pre-registered, and running two decision rules in parallel invites picking whichever one wins. The Bayesian layer is reported as corroborating evidence and as the risk framing for the business, not as a second verdict.

Agreement check: the frequentist interval clears the threshold and the posterior clears it — **the two frameworks agree**.

## 6. Validation against ground truth

This experiment was generated with a known injected lift of **+1.20%**. The 95% interval [+0.0051, +0.0140] **covers** it. The Bayesian credible interval [+0.0051, +0.0140] **covers** it as well.

Only synthetic data can falsify an experimentation engine this way: on real data there is no truth to check the interval against.

## Appendix — the decision rule, stated in advance

1. **Integrity gate.** SRM or balance failure ⇒ INVALID; the primary metric is not reported at all.
2. **Guardrail gate.** Any guardrail FAIL or INCONCLUSIVE ⇒ DO NOT SHIP.
3. **Practical significance.** CI entirely above the threshold ⇒ SHIP. CI entirely below ⇒ DO NOT SHIP. CI straddling the threshold ⇒ KEEP RUNNING, or DO NOT SHIP if the final look has been reached.

The threshold, not the p-value, is the ship criterion. A statistically significant effect below the threshold is a true finding and a bad decision.
