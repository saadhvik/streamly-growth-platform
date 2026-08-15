# Trustworthy Experimentation at Streamly

**One page. The five ways an experiment lies to you, and what we do about each.**

Every number below was measured by simulation in this repository — none is quoted from a paper. Reproduce them with `pytest tests/test_srm_and_sequential.py`.

---

## Why this exists

An experimentation program's value is not the number of experiments it runs. It is the fraction of its "wins" that survive contact with reality. A program that ships 30 wins a year, half of which are noise, is worse than one that ships 12 real ones — it burns engineering capacity on features that do nothing and, worse, teaches the organization that data is decorative.

The five failures below are ordered by how often they cause a false win.

---

## 1. Peeking — the most common way to manufacture a win

**The failure.** A team watches the dashboard daily and stops the moment p < 0.05. Every look is another chance to cross the line, so the false-positive rate compounds.

| Looks at a fixed α=0.05 | Actual Type-I error |
|---|---:|
| 1 | 5.0% |
| 5 | **14.2%** |
| 20 | **19.4%** |

Roughly **one "win" in seven** from a five-look peeking team is pure noise.

**What we do.** Pre-specified O'Brien-Fleming alpha-spending boundaries. The alpha budget is allocated across looks, so early stops demand overwhelming evidence and the boundary relaxes toward nominal at the end.

| Look | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| \|Z\| required | 4.383 | 3.099 | 2.551 | 2.254 | 2.064 |

Measured Type-I error under this rule, peeking at every look: **5.0%**. Power cost versus a fixed-sample test: **under 3pp**. Over 90% of genuinely large effects stop early.

> Telling people not to peek does not work. Give them boundaries that make peeking valid.

---

## 2. Sample ratio mismatch — the experiment that was never valid

**The failure.** Users did not arrive in the arms in the designed ratio. This almost never happens by chance; it means something is selectively dropping, misrouting, or double-counting users — and whatever it is, it correlates with the treatment. The measured effect is contaminated by a selection difference of unknown size and sign.

**There is no statistical repair.** Find the bug, rerun.

**What we do.** Chi-square screen on every experiment at **α=0.001**, not 0.05. At 0.05 one healthy experiment in twenty gets flagged, the team learns the alarm is noise, and the check stops being acted on. A fired alarm must mean "stop" every time.

**The honest limitation:** at 30,000 users per arm this check cannot detect an arm loss below **2.67%**. A passing SRM is evidence, not a guarantee. `srm_minimum_detectable_loss()` reports the floor for any sample size — quote it alongside the pass.

**And a check SRM cannot do:** arm sizes can be correct while assignment is wrong. We re-derive every variant from the bucketing hash and compare against what was recorded, which catches an assignment service that has drifted from its documented behaviour.

---

## 3. Underpowered guardrails — how harm ships

**The failure.** "Refund rate isn't significantly different, so we're fine." That treats *failure to detect* harm as *evidence of no harm*. An underpowered guardrail **always** passes a significance test. The less data you have, the safer everything looks.

**What we do.** Guardrails are **non-inferiority** tests against a tolerance margin agreed at intake:

> H₀: the metric degraded by at least `margin` — guilty until proven innocent.

Passing requires positive evidence that harm is below tolerance. Three outcomes, and **two of them block**:

| Verdict | Meaning | Launch |
|---|---|---|
| PASS | CI entirely below the margin | allowed |
| FAIL | CI entirely above the margin | blocked |
| **INCONCLUSIVE** | CI straddles the margin | **blocked** |

**No multiplicity correction across guardrails**, deliberately. Bonferroni makes each test *less* likely to fire, and for a safety check the expensive error is the missed regression. Alpha is spent generously on detecting harm and stingily on declaring wins.

**Power the guardrail, or it decides for you.** A margin can be a perfectly sensible business tolerance and still be unanswerable at the sample size available — in which case the guardrail returns INCONCLUSIVE and blocks the launch no matter what the treatment did. `minimum_resolvable_margin()` computes the smallest tolerance a given sample can clear, so this is settled at intake rather than discovered at readout.

**This is not hypothetical here.** Our reference experiment originally measured refunds among *converters only* — ~2,400 users per arm against 30,000 for the primary metric. That single denominator choice cost 11.6× in precision and left the guardrail permanently INCONCLUSIVE, blocking a clean, well-powered win (+0.95pp, p=2.4e-05). Measuring refunds over **all assigned users** — same metric, correct denominator — made it resolvable, and the experiment now ships.

> **Choose the denominator before the margin, and check the margin is resolvable before the experiment starts.**

---

## 4. Significance mistaken for importance

**The failure.** At Streamly's traffic, almost anything is "significant." A +0.02pp effect with p=0.001 is a true finding and a bad decision — it will not repay the engineering maintenance, support load, or product complexity of shipping it.

**What we do.** The ship criterion is the **practical significance threshold**, set at intake before any data exists: the smallest effect worth the cost of shipping. The decision compares the *confidence interval* to that threshold, not the p-value to 0.05.

| Interval vs threshold | Decision |
|---|---|
| Entirely above | SHIP |
| Entirely below | DO NOT SHIP — *a decisive negative, not a failure to detect* |
| Straddling | KEEP RUNNING (or DO NOT SHIP at the final look) |

This is also what produces an honest KEEP RUNNING instead of forcing a coin flip into a binary.

---

## 5. Metric shopping and the anchoring readout

**The failure.** Two "primary" metrics at α=0.05 give roughly a 10% chance of a spurious win. And a readout that prints "SRM detected" next to "+2.4% lift, p=0.001" gets shipped anyway — someone always argues the mismatch was probably harmless. Once the number is on the page, it anchors the decision.

**What we do.**
- **Exactly one primary metric**, named at intake. Secondary metrics explain a result; they never declare one.
- **The gates are ordered and short-circuit**: integrity → guardrails → primary. A large win cannot argue its way past an unresolved safety question.
- **An invalid experiment reports no effect size at all.** When the integrity gate fails, the readout refuses to compute the headline. The only honest readout of an invalid experiment is that it is invalid.

---

## The standard, in one paragraph

Pre-register the hypothesis, one primary metric, the practical threshold, the guardrail margins, and the look schedule — **before** data exists. Screen integrity first and stop if it fails. Treat guardrails as non-inferiority tests where "inconclusive" blocks. Decide on the interval against the threshold, not on the p-value. Report what the method actually achieved, including when that is nothing: our CUPED covariate correlates at ρ≈0.05 and delivers a 0.2% variance reduction, and the readout says so rather than implying CUPED did work it did not do.

**The goal is not more wins. It is that a win means something.**

---

*Reproduce every figure: `pytest tests/test_srm_and_sequential.py -q` · Decision rule: `src/streamly/experiment/readout.py` · Worked example: `docs/experiment_readout_example.md`*
