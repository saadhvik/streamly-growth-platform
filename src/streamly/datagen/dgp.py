"""Data-generating process (DGP) with KNOWN ground truth.

Design goal: create marketing journeys whose conversion outcome depends on a
hidden, controllable per-channel importance vector, while channel *exposure
volume* is decoupled from that importance. This decoupling is what lets us
prove data-driven attribution beats last-touch:

  * Meta has the highest exposure share (0.40) and tends to be the LAST touch,
    so last-click will over-credit it.
  * Google and email carry the true conversion lift, so removal-effect (Markov)
    and Shapley must surface them to recover ground truth.

Conversion is a logistic function of which channels appear in a journey:

    logit(p) = base_logit + IMPORTANCE_GAIN * sum_c [ channel_c in journey ] * w_c

where ``w_c`` are the true importance weights (they sum to 1). The normalized
credit a model assigns per channel is compared against ``w_c`` in Phase 3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from streamly.config import CHANNELS, DataGenConfig, ExperimentConfig

# Overall strength of channel effects on the conversion logit. Tuned so overall
# conversion sits near a realistic freemium->paid rate.
IMPORTANCE_GAIN: float = 1.6


@dataclass
class Journeys:
    """Container for simulated per-user journeys and outcomes."""

    user_ids: np.ndarray            # (N,)
    channel_seqs: list[list[int]]   # per-user ordered channel indices
    converted: np.ndarray           # (N,) bool
    conv_prob: np.ndarray           # (N,) true conversion probability


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def simulate_journeys(cfg: DataGenConfig) -> Journeys:
    """Simulate marketing journeys and conversions from the true DGP."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_users

    exposure = np.array([cfg.channel_exposure_share[c] for c in CHANNELS])
    exposure = exposure / exposure.sum()
    importance = np.array([cfg.channel_importance[c] for c in CHANNELS])
    importance = importance / importance.sum()  # ground truth, normalized

    base_logit = _logit(cfg.baseline_conversion)

    # Journey length: 1..max_touches, right-skewed (most users few touches).
    lengths = rng.integers(1, cfg.max_touches + 1, size=n)

    # Ordering uses its own generator so it cannot perturb the main random
    # stream. That keeps the channel multiset, the conversion draws and every
    # count identical to a run without funnel structure -- only the ORDER of
    # each journey changes, which is precisely the variable under study.
    order_rng = np.random.default_rng(cfg.seed + 99)
    funnel = np.array([cfg.channel_funnel_position[c] for c in CHANNELS])

    channel_seqs: list[list[int]] = []
    conv_prob = np.empty(n)
    for i in range(n):
        L = int(lengths[i])
        seq = rng.choice(len(CHANNELS), size=L, p=exposure).tolist()

        # Sort the sampled touches by funnel position plus noise. Sorting a
        # multiset cannot change which channels appear, so exposure shares and
        # conversion probability are untouched -- but late-funnel channels now
        # land late, which is what lets the ordering-sensitive rules disagree.
        scores = funnel[seq] + order_rng.normal(0.0, cfg.funnel_position_noise, size=L)
        seq = [seq[k] for k in np.argsort(scores, kind="stable")]
        channel_seqs.append(seq)

        present = np.zeros(len(CHANNELS))
        present[np.unique(seq)] = 1.0
        logit = base_logit + IMPORTANCE_GAIN * float((present * importance).sum())
        conv_prob[i] = 1.0 / (1.0 + np.exp(-logit))

    converted = rng.random(n) < conv_prob
    return Journeys(
        user_ids=np.arange(1, n + 1, dtype=np.int64),
        channel_seqs=channel_seqs,
        converted=converted,
        conv_prob=conv_prob,
    )


def true_importance(cfg: DataGenConfig) -> dict[str, float]:
    """Return the normalized ground-truth channel importance vector."""
    imp = np.array([cfg.channel_importance[c] for c in CHANNELS])
    imp = imp / imp.sum()
    return {c: float(w) for c, w in zip(CHANNELS, imp)}


def simulate_experiment(
    ecfg: ExperimentConfig, user_ids: np.ndarray
) -> dict[str, np.ndarray]:
    """Simulate the annual-paywall A/B test with a KNOWN injected lift.

    Assignment is by deterministic hash of user_id + salt (implemented in the
    experiment engine, Phase 4). Here we reproduce that bucketing to attach
    outcomes with a real, recoverable treatment effect and guardrail drift.
    """
    rng = np.random.default_rng(ecfg.seed)
    n = user_ids.shape[0]

    # Deterministic bucketing hash in [0,1) -- mirrors experiment.assign.
    import hashlib

    def h(uid: int) -> float:
        digest = hashlib.sha256(f"{ecfg.salt}:{uid}".encode()).hexdigest()
        return int(digest[:16], 16) / 16**16

    hashes = np.array([h(int(u)) for u in user_ids])
    variant = np.where(hashes < ecfg.true_traffic_split, "control", "treatment")
    is_treat = variant == "treatment"

    # Pre-experiment covariate correlated with the outcome (drives CUPED).
    latent = np.asarray(rng.standard_normal(n), dtype=float)
    pre_cov = latent  # standardized pre-period LTV proxy

    p = np.where(is_treat,
                 ecfg.control_conversion + ecfg.true_treatment_lift_abs,
                 ecfg.control_conversion)
    # Nudge conversion prob with the covariate so CUPED can strip variance.
    p = np.clip(p + ecfg.cuped_corr * 0.02 * latent, 0.001, 0.999)
    converted = (rng.random(n) < p).astype(float)

    refund_p = np.where(is_treat, ecfg.treatment_refund_rate, ecfg.control_refund_rate)
    guardrail_refund = (rng.random(n) < refund_p) & (converted > 0)

    lat_mean = np.where(is_treat, ecfg.treatment_latency_ms, ecfg.control_latency_ms)
    guardrail_latency = rng.normal(lat_mean, 30.0)

    return {
        "user_id": user_ids,
        "variant": variant,
        "bucketing_hash": hashes,
        "pre_covariate": pre_cov,
        "primary_metric": converted,
        "guardrail_refund": guardrail_refund,
        "guardrail_latency_ms": guardrail_latency,
    }
