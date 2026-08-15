# Metric Definitions

The purpose of this document is to make every number on a dashboard or in a readout **traceable to a table and a rule**. Two teams quoting different conversion rates is not a data-quality problem; it is a definitions problem, and it is solved here or not at all.

Each metric states its grain, its exact computation, and — most importantly — the ambiguity it resolves. A definition that does not say what it *excludes* has not defined anything.

---

## Conventions

| Convention | Rule | Why |
|---|---|---|
| **Grain** | Every metric names its unit: user, conversion, channel-day. | A rate whose denominator is unstated can be computed three ways and defended all three times. |
| **Randomization = analysis unit** | Experiment metrics are computed per assigned user. | Analysing sessions from a user-randomized test understates variance and manufactures significance. |
| **Revenue is net of refunds** | Refunded conversions contribute zero. | Gross revenue rewards channels that acquire users who churn straight back out. |
| **Absolute before relative** | Effects are quoted in absolute points first. | "A 10% lift" is ambiguous — 10pp, or 10% of 8%? That ambiguity underpowers experiments. |
| **Time basis** | Money is normalized to a 30-day month; the warehouse holds 90 days. | So every figure reconciles against the $500K/month budget. |

---

## 1. Acquisition and attribution

### `conversion` (event)
**Grain:** one row per converting user.
**Source:** `conversions`.
A user's first paid subscription. The generator emits at most one per user; the journey loader defensively aggregates with `MIN(convert_ts)` so a duplicated row can never fan out the touch log.

### `conversion_rate`
**Grain:** user. **Source:** `conversions` / `users`.
```sql
COUNT(DISTINCT c.user_id) * 1.0 / COUNT(DISTINCT u.user_id)
```
**Excludes nothing** — the denominator is all signed-up users, not just marketing-exposed ones. Reference value: **10.6%**.

### `journey` / `path`
**Grain:** user. **Source:** `touchpoints`.
The channel sequence for one user, ordered by `ts` ascending with `touch_id` as a deterministic tie-break.
- **Lookback window: 45 days.** Touches older than this before the conversion are dropped.
- **Touches logged after the conversion are dropped** — they cannot have caused it.
- **Non-converters keep their full path** (no conversion anchor to measure the window from). This is the standard convention and it matters: Markov and Shapley are both defined against the full population, and dropping null paths inflates every channel's measured contribution.

### `credited_conversions`
**Grain:** channel × model. **Source:** journeys + the chosen attribution model.
Each converting journey distributes **exactly 1.0** of credit. Consequently total credited conversions always equals the conversion count, for every model — this conservation property is asserted inside `credit_table()`, not merely tested, because leaked credit would corrupt every downstream ROI figure.

Repeated touches on the same channel within one journey **accumulate**: a channel appearing three times in a linear journey earns three shares.

### `attribution_share`
**Grain:** channel × model.
`credited_conversions / total_conversions`. Each model's column sums to 1.0, which is what makes models comparable to each other and to the ground-truth importance vector.

### `recovery_error` (MAE)
**Grain:** model.
Mean absolute difference, in share points, between a model's `attribution_share` and the true channel importance in `data/ground_truth/ground_truth.json`. **This is only computable on synthetic data** — it is the whole reason the platform is synthetic-first. Reference values: last-touch **12.81pp**, Shapley **4.00pp**.

### `CAC`
**Grain:** channel × model.
`monthly_spend / credited_conversions`. Varies by model *only* because the models disagree about who earned the conversion — spend and conversions are identical across models.

### `ROAS` — read the caveat
**Grain:** channel × model.
`credited_net_revenue / monthly_spend`.
> **This is first-payment return, not lifetime.** The warehouse records only the initial subscription payment, so levels read far below 1.0. Quote it as a payback input, never as profitability. The reallocation decision is unaffected: a common LTV multiple rescales every channel identically and leaves ranking, value shares, and the proposed budget move unchanged.

---

## 2. Experimentation

### `assignment`
**Grain:** user × experiment. **Source:** `experiment_assignment`.
Variant is a pure function of `sha256(f"{salt}:{user_id}")` — no lookup, no assignment-time write. The salt is mixed into the hash rather than into a seed, so concurrent experiments are uncorrelated by construction.
> The literal string format `"{salt}:{unit_id}"` is a **contract**. Changing it re-randomizes every running experiment.

### `primary_metric`
**Grain:** assigned user.
Binary converted flag for the reference paywall experiment. One per experiment — see the intake template on why co-primary metrics inflate the false-positive rate.

### `absolute_effect` / `relative_effect`
`treatment_mean − control_mean`, and that difference over the control mean.
**Test statistic uses pooled variance; the confidence interval uses unpooled.** The null asserts a common rate, so the test pools; the interval estimates a difference under no such assumption, so it does not. Mixing these produces intervals that disagree with their own p-value near the boundary.

### `practical_significance_threshold`
**Grain:** experiment. Set at intake, never after.
The smallest absolute effect worth the cost of shipping. **This, not the p-value, is the ship criterion.**

### `SRM` (sample ratio mismatch)
**Grain:** experiment.
Chi-square goodness-of-fit of arm sizes against the intended split, **screened at α=0.001** rather than 0.05 — it runs on every experiment, and at 0.05 one healthy experiment in twenty would be flagged until the team stopped believing the alarm.
> **Sensitivity floor:** at 30,000 users per arm the check cannot detect an arm loss below **2.67%**. A passing SRM is evidence, not a guarantee. `srm_minimum_detectable_loss()` returns this figure for any sample size.

### `guardrail harm`
**Grain:** experiment × guardrail. **Denominator: all assigned users**, never a conditional subpopulation — conditioning on conversion cost 11.6× in precision here and made the refund guardrail unresolvable at any realistic margin.
Signed so **positive always means degradation**, regardless of whether the underlying metric is better high or low. Compared against a pre-agreed tolerance `margin` by a non-inferiority test:
- CI entirely below the margin → **PASS**
- CI entirely above → **FAIL**
- CI straddling → **INCONCLUSIVE**, which blocks the launch exactly like a FAIL.

Margins for rate guardrails are declared **relative to the control rate** (the refund tolerance is 50%), because an absolute margin silently changes meaning when the denominator does: 2pp is a quadrupling of a 0.5% base rate and a rounding error on a 6% one. `minimum_resolvable_margin()` reports the smallest margin a given sample can clear; a margin below it can never pass.

### `CUPED variance reduction`
**Grain:** experiment.
`1 − var(adjusted) / var(raw)`, theoretically `ρ²`. θ is estimated on the **pooled** sample; fitting it per arm lets the adjustment absorb part of the treatment effect and biases the estimate toward zero.
> The covariate must be **pre-experiment**. A covariate measured during the experiment can be moved by the treatment, and adjusting for it removes part of the effect being measured. No arithmetic can detect this mistake.

### `information_fraction` and `efficacy boundary`
**Grain:** experiment × look.
Fraction of planned sample accrued, and the `|Z|` required to stop at that look under O'Brien-Fleming alpha spending. Look times are **pre-specified** — choosing them after seeing data reintroduces the bias the boundaries exist to remove.

---

## 3. Known limitations of these definitions

Stated here so nobody has to rediscover them in a meeting.

1. **Attribution measures association, not incrementality.** No model here randomizes exposure. A channel systematically shown to already-high-intent users earns credit it did not cause. A geo holdout is the only fix.
2. **Shapley's organic baseline is unobservable.** Zero-touch users never appear in a marketing log, so `v(∅)` is fixed at 0 and the efficiency axiom spreads organic conversions across channels, compressing shares toward the mean. This *flatters* weak channels, making the case against over-credited channels a conservative floor.
3. **Markov's removal effect conflates reach with incrementality.** Deleting a channel that touches most journeys strands most paths regardless of persuasion. It is retained as a divergence alarm against Shapley, not as a decision model.
4. **`ROAS` is first-payment, not LTV.** See above.
5. **Ground-truth-based metrics exist only on synthetic data.** `recovery_error` cannot be computed in production. In production the equivalent evidence is a holdout.
