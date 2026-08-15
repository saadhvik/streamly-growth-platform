"""Materialize the synthetic DGP into the DuckDB warehouse + ground-truth file.

Writes the five raw tables and persists the true channel-importance vector and
injected experiment effects to ``data/ground_truth/`` so validation phases can
score model recovery against a locked reference.

Run:  python -m streamly.datagen.generator
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from streamly import marts, warehouse
from streamly.config import (
    CHANNELS,
    DATAGEN,
    EXPERIMENT,
    GROUND_TRUTH_DIR,
    DataGenConfig,
    ExperimentConfig,
)
from streamly.datagen import dgp

_START = datetime(2026, 5, 10)  # 90 days back from "today" (2026-08-08)


def _build_frames(
    cfg: DataGenConfig, ecfg: ExperimentConfig
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed + 1)
    j = dgp.simulate_journeys(cfg)
    n = cfg.n_users

    # --- users ---
    platforms = rng.choice(["ios", "android", "web"], size=n, p=[0.45, 0.40, 0.15])
    countries = rng.choice(["US", "UK", "DE", "IN", "BR"], size=n,
                           p=[0.45, 0.15, 0.12, 0.18, 0.10])
    signup_offsets = rng.integers(0, cfg.days_of_history, size=n)
    signup_ts = [_START + timedelta(days=int(d), minutes=int(rng.integers(0, 1440)))
                 for d in signup_offsets]
    users = pd.DataFrame({
        "user_id": j.user_ids,
        "signup_ts": signup_ts,
        "platform": platforms,
        "country": countries,
        "acquisition_cohort": [t.strftime("%Y-%m") for t in signup_ts],
    })

    # --- touchpoints (one row per touch) ---
    monthly_spend = cfg.monthly_spend
    # cost per touch ~ monthly spend / expected touches on that channel
    exposure = np.array([cfg.channel_exposure_share[c] for c in CHANNELS])
    exposure = exposure / exposure.sum()
    total_touches_est = sum(len(s) for s in j.channel_seqs)
    per_channel_touches = np.maximum(exposure * total_touches_est, 1.0)
    cost_per_touch = {
        c: (monthly_spend[c] * (cfg.days_of_history / 30.0)) / per_channel_touches[i]
        for i, c in enumerate(CHANNELS)
    }

    t_user, t_channel, t_ts, t_cost, t_pos, t_campaign = [], [], [], [], [], []
    for uid, seq, s_ts in zip(j.user_ids, j.channel_seqs, signup_ts):
        # touches occur in the days leading up to signup
        n_t = len(seq)
        deltas = sorted(rng.integers(0, 30, size=n_t), reverse=True)
        for pos, (ch_idx, dd) in enumerate(zip(seq, deltas)):
            ch = CHANNELS[ch_idx]
            t_user.append(uid)
            t_channel.append(ch)
            t_ts.append(s_ts - timedelta(days=int(dd), hours=int(rng.integers(0, 24))))
            t_cost.append(round(cost_per_touch[ch], 4))
            t_pos.append(pos)
            t_campaign.append(f"{ch}_cmp_{rng.integers(1, 4)}")
    touchpoints = pd.DataFrame({
        "touch_id": np.arange(1, len(t_user) + 1, dtype=np.int64),
        "user_id": t_user, "channel": t_channel, "campaign": t_campaign,
        "ts": t_ts, "cost_attributed": t_cost, "position_in_journey": t_pos,
    })

    # --- conversions (only for converted users) ---
    conv_mask = j.converted
    conv_uids = j.user_ids[conv_mask]
    n_c = conv_uids.shape[0]
    plan = rng.choice(["annual", "monthly"], size=n_c,
                      p=[cfg.annual_plan_share, 1 - cfg.annual_plan_share])
    revenue = np.where(plan == "annual", cfg.revenue_annual, cfg.revenue_monthly)
    conv_signup = users.set_index("user_id").loc[conv_uids, "signup_ts"].values
    convert_ts = [pd.Timestamp(s) + timedelta(hours=int(rng.integers(1, 72)))
                  for s in conv_signup]
    is_refunded = rng.random(n_c) < cfg.refund_rate
    refund_ts = [ct + timedelta(days=int(rng.integers(1, 20))) if r else pd.NaT
                 for ct, r in zip(convert_ts, is_refunded)]
    conversions = pd.DataFrame({
        "conversion_id": np.arange(1, n_c + 1, dtype=np.int64),
        "user_id": conv_uids, "convert_ts": convert_ts, "plan_type": plan,
        "revenue": revenue.astype(float), "is_refunded": is_refunded,
        "refund_ts": refund_ts,
    })

    # --- spend (daily channel-level; reconciles to monthly brief) ---
    dates = [(_START + timedelta(days=d)).date() for d in range(cfg.days_of_history)]
    s_rows = []
    for d in dates:
        for c in CHANNELS:
            daily = monthly_spend[c] / 30.0
            daily *= float(rng.normal(1.0, 0.08))  # day-to-day noise
            imps = int(daily * rng.integers(80, 140))
            clicks = int(imps * rng.uniform(0.01, 0.05))
            s_rows.append((d, c, f"{c}_cmp_1", round(daily, 2), imps, clicks))
    spend = pd.DataFrame(s_rows, columns=["date", "channel", "campaign",
                                          "spend", "impressions", "clicks"])

    # --- experiment assignment ---
    ex = dgp.simulate_experiment(ecfg, j.user_ids)
    assign_ts = [s + timedelta(days=int(rng.integers(0, 5))) for s in signup_ts]
    experiment = pd.DataFrame({
        "user_id": ex["user_id"],
        "experiment_id": ecfg.experiment_id,
        "variant": ex["variant"],
        "assign_ts": assign_ts,
        "bucketing_hash": ex["bucketing_hash"],
        "pre_covariate": ex["pre_covariate"],
        "primary_metric": ex["primary_metric"],
        "guardrail_refund": ex["guardrail_refund"],
        "guardrail_latency_ms": ex["guardrail_latency_ms"],
    })

    return {
        "users": users, "touchpoints": touchpoints, "conversions": conversions,
        "spend": spend, "experiment_assignment": experiment,
    }


def _write_ground_truth(cfg: DataGenConfig, ecfg: ExperimentConfig) -> None:
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "channel_importance_true": dgp.true_importance(cfg),
        "channel_exposure_share": cfg.channel_exposure_share,
        "monthly_spend": cfg.monthly_spend,
        "experiment": {
            "experiment_id": ecfg.experiment_id,
            "control_conversion": ecfg.control_conversion,
            "true_treatment_lift_abs": ecfg.true_treatment_lift_abs,
            "true_traffic_split": ecfg.true_traffic_split,
            "control_refund_rate": ecfg.control_refund_rate,
            "treatment_refund_rate": ecfg.treatment_refund_rate,
        },
        "seed": cfg.seed,
    }
    with open(GROUND_TRUTH_DIR / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def generate(cfg: DataGenConfig = DATAGEN, ecfg: ExperimentConfig = EXPERIMENT) -> dict[str, int]:
    """Generate all tables, load into DuckDB, persist ground truth."""
    frames = _build_frames(cfg, ecfg)
    con = warehouse.connect()
    warehouse.reset_schema(con)
    for name, df in frames.items():
        con.register("_tmp", df)
        con.execute(f"INSERT INTO {name} SELECT * FROM _tmp")
        con.unregister("_tmp")
    # Marts are views, so building them here costs nothing and guarantees the
    # analytical layer can never be stale relative to the raw tables.
    marts.build_marts(con)

    counts: dict[str, int] = {}
    for name in frames:
        row = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()
        counts[name] = int(row[0]) if row else 0
    con.close()
    _write_ground_truth(cfg, ecfg)
    return counts


if __name__ == "__main__":
    counts = generate()
    print("Loaded rows:")
    for k, v in counts.items():
        print(f"  {k:24s} {v:>10,}")
    print("Ground truth -> data/ground_truth/ground_truth.json")
