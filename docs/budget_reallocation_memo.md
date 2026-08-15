# Budget Reallocation Memo — Streamly Paid Acquisition

**To:** VP Growth, VP Finance
**From:** Growth Analytics
**Re:** $122K/month of the $500K paid budget is pointed at the wrong channel
**Status:** Recommendation — requires a geo holdout before full rollout

---

## 1. The recommendation in one line

Move **$45.6K/month** — 9% of the paid budget — out of Meta and TikTok and into Email, Referral, and Google. Expected return: **+209 conversions/month** at equal spend. Run it as a 6-week geo holdout before committing the full move.

## 2. Why the current numbers are wrong

Streamly credits conversions on **last touch**. Last touch gives 100% of the credit to whichever channel a user happened to see immediately before subscribing. It cannot distinguish a channel that *caused* a subscription from one that merely *witnessed* it.

That distinction is not academic here. Meta runs the highest exposure volume of any channel — it appears in more journeys than anything else, and therefore appears *last* in more journeys than anything else. Last touch reads that as performance.

We validated this on a synthetic replica of Streamly's funnel where the true per-channel contribution is known by construction. Every model was scored on how closely it recovered that known truth:

| Model | Mean abs. error vs truth | Error vs last-touch |
|---|---:|---:|
| **Shapley (recommended)** | **4.00 pp** | **−69%** |
| Markov removal-effect | 10.91 pp | −15% |
| First touch | 12.71 pp | −1% |
| Position-based (40/20/40) | 12.75 pp | 0% |
| Time decay | 12.78 pp | 0% |
| Linear | 12.79 pp | 0% |
| Last touch (incumbent) | 12.81 pp | — |

The finding that should change how this team thinks: **the five heuristic rules are indistinguishable from each other.** Switching from last-touch to time-decay or 40/20/40 — the usual "let's use a fairer model" response — moves the error by less than 0.1pp. The problem is not which rule; it is that no rule observes a counterfactual.

## 3. Where the credit actually goes

| Channel | Last touch | Shapley | True | Verdict |
|---|---:|---:|---:|---|
| meta | 37.5% | 18.6% | 14.0% | **over-credited by 2.7×** |
| google | 24.4% | 32.5% | 34.0% | under-credited |
| email | 11.0% | 19.5% | 28.0% | **most under-credited** |
| referral | 8.6% | 16.5% | 14.0% | under-credited |
| tiktok | 18.5% | 13.0% | 10.0% | over-credited |

Shapley does not recover truth perfectly — it still flatters Meta (18.6% vs 14.0% true) and understates Email. That residual is a known, one-directional bias explained in §6. It means our case against Meta is a **conservative floor**, not an overstatement.

## 4. Unit economics under each model

Same spend, same conversions — only the crediting differs.

| Channel | Monthly spend | CAC (last touch) | CAC (Shapley) | Change |
|---|---:|---:|---:|---|
| meta | $176,909 | $222 | **$449** | 2.0× worse than believed |
| tiktok | $89,785 | $229 | $326 | 1.4× worse |
| google | $148,461 | $287 | $216 | better |
| referral | $49,690 | $273 | $142 | 1.9× better |
| email | $35,237 | $151 | **$85** | best channel in the portfolio |

Email is currently the smallest line item in the budget (7%) and the cheapest source of subscribers by a factor of five. That is the single clearest misallocation on the page.

*Note on ROAS:* the warehouse records only the first subscription payment, so absolute ROAS values read below 1.0 and should be used as a payback input, not as profitability. The reallocation decision is unaffected — a common LTV multiple rescales every channel identically and leaves the ranking unchanged.

## 5. The proposed move

Budget-neutral, capped at ±30% per channel so it is executable in one cycle without stranding campaign commitments.

| Channel | Current | Proposed | Δ | Δ% |
|---|---:|---:|---:|---:|
| meta | $176,909 | $145,609 | −$31,300 | −17.7% |
| tiktok | $89,785 | $75,470 | −$14,315 | −15.9% |
| google | $148,461 | $168,598 | +$20,137 | +13.6% |
| referral | $49,690 | $64,598 | +$14,907 | +30.0% (capped) |
| email | $35,237 | $45,809 | +$10,571 | +30.0% (capped) |
| **Total** | **$500,082** | **$500,082** | **$0** | — |

**Expected: +209 conversions/month, +$7.4K/month first-payment revenue, at identical spend.**

Email and Referral both hit the cap, meaning the uncapped optimum would move more. We are deliberately not proposing it — see §6.

## 6. What would make this wrong

Stated plainly, because a reallocation memo without a falsification section is a sales pitch.

1. **The expected gain is first-order.** It prices moved dollars at each channel's *current* conversions-per-dollar and assumes locally constant returns. Email and Referral are small channels; they will hit diminishing returns and saturation earlier than the linear estimate implies. The +209 is a directional prior, not a forecast. The ±30% cap exists specifically to keep us inside the range where the assumption is least abused.

2. **Shapley inherits an unobservable baseline.** Users who convert with zero marketing exposure never appear in a marketing log, so the "no channels" coalition value cannot be estimated and is fixed at zero. The efficiency axiom then spreads organic conversions across the channels rather than excluding them, compressing every share toward the average. This **flatters weak channels** — it is why Shapley still gives Meta 18.6% against a true 14.0%. Correcting it requires an unexposed holdout, which is a media-plan change, not a modelling one.

3. **Markov and Shapley disagree here, and that is a signal, not a bug.** Markov's removal effect gave Meta 30.1% against Shapley's 18.6%. The two methods fail differently by design: removal-effect asks "what if this channel vanished?", which conflates a channel's *reach* with its *incrementality* — delete a channel that touches most journeys and you strand most paths regardless of whether it persuaded anyone. Shapley asks "what does this channel add on the margin?", which is reach-independent. On journeys where ordering carries little information, Markov degrades toward a volume measure, which is precisely the failure we are trying to correct. **We therefore recommend Shapley as the decision model and retain Markov as a divergence alarm.**

4. **Correlation is not incrementality — for either model.** Both methods observe channels; neither randomizes them. A channel that is systematically shown to already-high-intent users will earn credit it did not cause. No observational model fixes this.

## 7. Recommended path

1. **Do not reallocate on this memo alone.** Run a 6-week geo holdout: suppress Meta in a matched set of geos, hold everything else constant, and measure the actual conversion delta.
2. **Compare the holdout's measured Meta incrementality against the 18.6% Shapley figure.** If the holdout lands at or below it, execute the full move and adopt Shapley as the reporting standard.
3. **Instrument an unexposed control cell** (~2% of traffic, zero paid exposure) so the organic baseline in §6.2 becomes estimable and the remaining bias can be removed rather than merely disclosed.
4. **Report both models going forward.** Track the Markov/Shapley gap per channel; a widening gap is an early warning that journey structure has shifted.

## 8. Reproducing this

```bash
PYTHONPATH=src python -m streamly.attribution.validate   # recovery vs ground truth
PYTHONPATH=src python -m streamly.attribution.roi        # ROI + reallocation plan
pytest tests/test_attribution_recovery.py                # the gate, as a test
```

Every figure above is regenerated from a fixed seed. Ground truth lives in `data/ground_truth/ground_truth.json` and is written by the generator, never read by any model.
