# Experiment Intake

**Fill this in before the experiment starts. An intake completed afterwards is a narrative, not a design.**

Every field below exists because leaving it blank causes a specific, recurring failure. The reason is stated next to each one so the form can be argued with rather than merely obeyed.

---

## 1. Ownership

| Field | Value |
|---|---|
| Experiment ID | `snake_case_v1` |
| Owner (accountable for the decision) | |
| Analyst | |
| Engineer | |
| Target start / end | |

---

## 2. The hypothesis

> **Because** [evidence we already have], **we believe that** [change], **will cause** [effect on a named metric], **which we will measure by** [metric + threshold].

*Why this shape:* a hypothesis without a prior ("because…") is a guess, and a guess that wins is indistinguishable from noise you got lucky on. A hypothesis without a named metric can be re-aimed after the fact at whichever number happened to move.

**What would change our mind?** _State the result that would make you abandon this idea._ If no result would, this is not an experiment — it is a rollout, and it should be scheduled as one.

---

## 3. Primary metric — exactly one

| Field | Value |
|---|---|
| Metric name (must exist in `docs/metric_definitions.md`) | |
| Type | proportion / mean / ratio |
| Randomization unit | user / session / device |
| Analysis unit | *must match the randomization unit — see below* |

> **One primary metric.** Not two, not "one primary and a co-primary." Every additional metric that can declare victory raises the false-positive rate; two "primary" metrics at α=0.05 give roughly a 10% chance of a spurious win. Secondary metrics are for *understanding* the result, never for declaring it.

> **Analysis unit must equal randomization unit.** Randomizing users but analysing sessions treats one user's ten sessions as ten independent observations, understating the variance and manufacturing significance. If they must differ, the analysis needs clustered standard errors and that has to be agreed here.

---

## 4. Practical significance threshold

| Field | Value |
|---|---|
| Baseline rate (current) | |
| **Smallest effect worth shipping (absolute pp)** | |
| Why that number — what does shipping cost? | |

> This is the ship criterion, **not** p < 0.05. A statistically significant +0.02pp is a true finding and a bad decision. The threshold must reflect the real cost of shipping: engineering maintenance, support load, added product complexity. If nobody can articulate the cost, the honest threshold is "larger than we can detect," and the experiment should not run.

---

## 5. Design

| Field | Value |
|---|---|
| Traffic split (control / treatment) | |
| Users required per arm | *from `design.sample_size_two_proportions`* |
| Expected duration | *from `design.duration_days`* |
| Alpha | 0.05 |
| Power | 0.80 |
| Pre-experiment covariate for CUPED | |
| Expected covariate correlation ρ | |

```bash
PYTHONPATH=src python -c "
from streamly.experiment import design
d = design.sample_size_two_proportions(BASELINE, MDE, power=0.80); print(d)
print(f'{design.duration_days(d.n_total, DAILY_USERS):.1f} days')"
```

> **On CUPED:** variance reduction is `1 − ρ²`, so ρ=0.6 saves 36% of the sample and ρ=0.3 saves 9%. Below about ρ=0.4 the complexity is not worth it. Record the *expected* ρ here so the readout's *achieved* ρ can be checked against it — a large gap means the covariate is not what you thought.

> Run for **whole weeks**. Weekday/weekend composition is itself a covariate; stopping mid-week changes the population, not just the sample size.

---

## 6. Guardrails — what must not break

| Metric | Direction | **Tolerance margin** | Why this margin is acceptable |
|---|---|---|---|
| refund_rate | lower is better | | |
| p95_latency_ms | lower is better | | |
| d7_retention | higher is better | | |

> **Margins are set here, before any data exists.** Setting a margin after seeing the guardrail reading converts a safety check into a rationalization. This is the single most important pre-commitment on the form.

> Guardrails are **non-inferiority** tests: passing requires positive evidence that harm is below the margin. "Not statistically significant" is *not* a pass — an underpowered guardrail would always produce it. A guardrail that cannot resolve its margin returns INCONCLUSIVE and blocks the launch.

> **Check each guardrail's power on the population it is measured on — this is a computation, not a judgement call:**
>
> ```bash
> PYTHONPATH=src python -c "
> from streamly.experiment import guardrails as g
> print('smallest resolvable margin:', g.minimum_resolvable_margin(BASE_RATE, N_PER_ARM, N_PER_ARM))
> print('users needed for your margin:', g.required_n_for_margin(BASE_RATE, MARGIN))"
> ```
>
> If your declared margin is below the resolvable floor, the guardrail **cannot pass** — it will return INCONCLUSIVE and block the launch regardless of what the treatment does. Fix that here, by widening the margin with a stated business justification or by collecting more data, not at readout.

> **Prefer a denominator of all assigned users.** A guardrail computed on a conditional subpopulation inherits its much smaller sample. In this repo's reference experiment, measuring refunds among converters rather than all assigned users cost 11.6× in precision and made the guardrail permanently unresolvable.

---

## 7. Interim analyses

| Field | Value |
|---|---|
| Number of looks (including the final one) | |
| Information fractions | e.g. 0.2 / 0.4 / 0.6 / 0.8 / 1.0 |
| Spending function | `obrien_fleming` (default) / `pocock` |

> **Look times are pre-specified.** Choosing when to look after seeing the data reintroduces exactly the bias the boundaries remove. Five looks at a fixed α=0.05 produce a ~14% false-positive rate (measured in `tests/test_srm_and_sequential.py`); the boundaries hold it at 5%.

> Use O'Brien-Fleming unless there is a specific reason not to: it costs under 3pp of power versus a fixed-sample test, so it is close to free insurance. Choose Pocock only when stopping early is worth a real penalty at the final analysis.

---

## 8. Pre-registered decision rule

Complete these sentences *now*:

- **We will ship if** …
- **We will not ship if** …
- **We will keep running if** …

The engine's default rule, for reference:

1. **Integrity gate.** SRM or pre-period imbalance ⇒ INVALID; the primary metric is not reported at all.
2. **Guardrail gate.** Any guardrail FAIL or INCONCLUSIVE ⇒ DO NOT SHIP.
3. **Practical significance.** CI entirely above the threshold ⇒ SHIP. CI entirely below ⇒ DO NOT SHIP. CI straddling ⇒ KEEP RUNNING, or DO NOT SHIP at the final look.

---

## 9. Sign-off

| Role | Name | Date |
|---|---|---|
| Owner | | |
| Analyst | | |
| Reviewer (not on the team running it) | | |

> The independent reviewer exists because the person who wants the feature to win should not be the only person who approved how winning is defined.
