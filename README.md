# Streamly Growth Platform

**Multi-touch attribution and trustworthy experimentation, validated against a known ground truth.**

Two mandates: stop misallocating a $500K/month acquisition budget, and make experiment results mean something. Both hinge on trust, so the entire platform is built synthetic-first — the data generator writes the true channel importances and the true treatment lift to a file that no model is allowed to read until validation. That is the only way to *falsify* an attribution model. Real data cannot do it: there is nothing to check the answer against.

---

## The headline result

**Last-touch misattributes 2.7× more credit to Meta than it earned. Shapley recovers the truth with 69% less error.**

| Model | Recovery error (MAE vs truth) | vs last-touch |
|---|---:|---:|
| **Shapley** | **4.00 pp** | **−69%** |
| Markov removal-effect | 10.91 pp | −15% |
| First touch | 12.71 pp | −1% |
| Position-based (40/20/40) | 12.75 pp | 0% |
| Time decay | 12.78 pp | 0% |
| Linear | 12.79 pp | 0% |
| Last touch *(incumbent)* | 12.81 pp | — |

The finding worth pausing on: **the five heuristic rules are indistinguishable from each other.** Switching from last-touch to time-decay or 40/20/40 — the usual "let's use a fairer model" response — moves the error by less than 0.1pp. The problem is not which rule you pick; it is that no rule observes a counterfactual.

Consequence: **$122,516/month** of the budget is pointed at the wrong channel. The proposed move is $45.6K/month for an expected +209 conversions at identical spend → [`docs/budget_reallocation_memo.md`](docs/budget_reallocation_memo.md).

**On the experimentation side:** peeking five times at a fixed α=0.05 produces a **14.2%** false-positive rate; the alpha-spending boundaries hold it at **5.0%**. Both measured by simulation here, not cited.

The reference experiment runs the full gate sequence end to end and returns **SHIP**: integrity clean, both guardrails inside tolerance, and a +0.95pp effect whose 95% interval [+0.51pp, +1.40pp] **covers the injected +1.20pp ground truth**. The Bayesian layer prices the decision — expected loss from shipping rounds to zero against 0.95pp from holding → [`docs/experiment_readout_example.md`](docs/experiment_readout_example.md).

That verdict was originally **DO NOT SHIP**, blocked by a guardrail measured on the wrong denominator: refunds among *converters* (~2,400/arm) rather than among all assigned users (30,000/arm), costing 11.6× in precision and leaving the check permanently unresolvable. The fix was to the metric design, not the margin — and `minimum_resolvable_margin()` now makes that a computation at intake instead of a discovery at readout.

---

## Quick start

```bash
pip install -e ".[dev]"

python -m streamly.datagen.generator      # build the warehouse + ground truth
python -m streamly.attribution.validate   # score every model against truth
python -m streamly.attribution.roi        # ROI + the reallocation plan
python -m streamly.experiment.readout     # generate the decision document

pytest -q                                 # 197 tests
streamlit run app/streamlit_app.py        # the UI
```

`STREAMLY_DATA_DIR` relocates the warehouse if the repo directory is read-only or has restricted deletion (which breaks DuckDB's write-ahead-log cleanup).

---

## How it is validated

Every claim above is a test that fails loudly if it stops being true.

| Gate | Test |
|---|---|
| Credit conservation — every rule distributes exactly 100% | `test_rule_attribution.py` |
| Data-driven attribution beats last-touch on MAE | `test_attribution_recovery.py` |
| Shapley satisfies efficiency, symmetry, null-player | `test_attribution_recovery.py` |
| Statistics match statsmodels / scipy to 1e-10 | `test_experiment_stats.py` |
| Designed power and CI coverage confirmed by Monte Carlo | `test_experiment_stats.py` |
| A seeded SRM break is detected; clean splits are not flagged | `test_srm_and_sequential.py` |
| Type-I error held at α under peeking | `test_srm_and_sequential.py` |
| An invalid experiment never prints an effect size | `test_readout_and_app.py` |
| Posterior quantities match a 4M-draw Monte Carlo | `test_bayesian.py` |

---

## Architecture

```
src/streamly/
├── config.py            # every parameter and seed, in one place
├── warehouse.py         # DuckDB schema DDL
├── viz.py               # chart specs (validated palette, light + dark)
│
├── datagen/
│   ├── dgp.py           # the generative model + recorded ground truth
│   └── generator.py     # materializes 5 tables into DuckDB
│
├── attribution/
│   ├── sessionize.py    # touch log -> ordered journeys (one loader for all models)
│   ├── rules.py         # first / last / linear / time-decay / position
│   ├── markov.py        # absorbing chain + removal effect
│   ├── shapley.py       # exact coalitional values over 2^n coalitions
│   ├── roi.py           # CAC, ROAS, capped budget reallocation
│   └── validate.py      # recovery error vs ground truth
│
└── experiment/
    ├── design.py        # sample size, power, MDE, duration
    ├── assign.py        # deterministic salted-hash bucketing
    ├── integrity.py     # SRM, covariate balance, assignment reproducibility
    ├── frequentist.py   # 2-prop z, Welch t, CUPED
    ├── bayesian.py      # conjugate Beta-Binomial: P(B>A), expected loss
    ├── sequential.py    # Lan-DeMets alpha spending (exact recursion)
    ├── guardrails.py    # non-inferiority checks
    └── readout.py       # the gated decision rule
```

**Why DuckDB:** zero-infra, single-file, columnar, identical on a laptop and in CI. The SQL is ANSI-portable, so lifting to BigQuery is a connection change rather than a rewrite.

**Why both Markov and Shapley:** they fail differently. Markov captures sequence but conflates reach with incrementality; Shapley is order-agnostic and axiomatically fair but combinatorially expensive. Agreement between two independent methods is the confidence signal; divergence is a flag to investigate.

---

## Boardroom artifacts

| Document | What it is for |
|---|---|
| [Budget reallocation memo](docs/budget_reallocation_memo.md) | The finance-facing case, with a falsification section |
| [Experiment intake template](docs/experiment_intake_template.md) | Pre-registration — every field explains the failure it prevents |
| [Experiment readout (generated)](docs/experiment_readout_example.md) | A worked ship/no-ship decision, produced by code |
| [Metric definitions](docs/metric_definitions.md) | Every number traceable to a table and a rule |
| [Trustworthy experimentation one-pager](docs/trustworthy_experimentation_onepager.md) | The five ways an experiment lies, and the countermeasures |

---

## Known limitations

Stated here rather than discovered in a meeting. Each is also enforced or disclosed in code.

1. **No model here measures true incrementality.** Nothing randomizes channel exposure, so a channel shown to already-high-intent users earns credit it did not cause. A geo holdout is the recommended confirmation before any budget moves — the memo recommends one rather than acting on the model alone.
2. **Shapley's organic baseline is unobservable.** Zero-touch users never appear in a marketing log, so `v(∅)` is fixed at 0 and the efficiency axiom spreads organic conversions across channels. This compresses shares toward the mean and *flatters* weak channels — Shapley still gives Meta 18.6% against 14.0% true. The case against Meta is therefore a conservative floor, not an overstatement.
3. **Markov only beats last-touch by 15%, and no change to the generator would fix that.** The removal effect and the ground truth measure different things: removal effect asks what fraction of conversions flow through a channel when deleting it strands the path — reach-weighted throughput — while ground truth is per-channel incrementality. A generator change cannot repair an estimand mismatch, and the obvious alternative (splicing the channel out of paths rather than routing to null) is an algebraic identity that yields exactly zero removal effect for every channel. Markov is therefore retained as a divergence alarm against Shapley, not as a decision model. What sequence structure in the DGP *would* fix is the flatness of the five heuristic rules, which currently land within 0.1pp of each other.
4. **CUPED buys almost nothing on this dataset.** The pre-experiment covariate correlates at ρ≈0.05, giving a 0.21% variance reduction. CUPED is proven correct on simulated data at known ρ (0.7 → 49% reduction, unbiased over 300 replications), and the readout reports what it actually achieved rather than implying it helped.
5. **SRM has a sensitivity floor.** At 30,000 users per arm it cannot detect an arm loss below 2.67%. `srm_minimum_detectable_loss()` makes this quotable.
6. **ROAS is first-payment, not lifetime.** Levels read below 1.0 and are a payback input, not profitability. A common LTV multiple leaves the reallocation decision unchanged.
7. **The Bayesian layer does not drive the decision.** It reports P(B>A), expected loss, and a credible interval, but the pre-registered frequentist rule is what decides. Running two decision rules in parallel invites picking whichever one wins; when they disagree, the readout says so rather than resolving it silently.

---

## Reproducibility

Every figure comes from a fixed seed (`config.DataGenConfig.seed = 42`). Ground truth is written by the generator to `data/ground_truth/ground_truth.json` and read only by `validate.py`. CI regenerates the warehouse from scratch and re-runs the full pipeline, so a documented number that no longer reproduces fails the build.
