"""Typed, central configuration for the Streamly growth platform.

Every phase reads its parameters from here so runs are reproducible and the
data-generating assumptions live in exactly one place.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
# The Cowork mount is create-only (no unlink), which breaks DuckDB's WAL
# cleanup. Default the live warehouse to a locally-writable build dir; snapshots
# are copied back to the repo's data/ for delivery. Override with STREAMLY_DATA_DIR.
DATA_DIR: Path = Path(os.environ.get("STREAMLY_DATA_DIR", str(REPO_ROOT / "data")))
WAREHOUSE_PATH: Path = DATA_DIR / "warehouse.duckdb"
GROUND_TRUTH_DIR: Path = DATA_DIR / "ground_truth"

CHANNELS: tuple[str, ...] = ("google", "meta", "tiktok", "email", "referral")


@dataclass(frozen=True)
class DataGenConfig:
    """Parameters controlling the synthetic data-generating process (DGP).

    The ``channel_importance`` weights are the GROUND TRUTH that the attribution
    models must recover. They govern how much each channel actually moves a
    user toward conversion in the simulation -- nothing else in the pipeline
    is allowed to read them until validation (Phase 3).
    """

    seed: int = 42
    n_users: int = 60_000

    # Monthly marketing spend by channel (sums to the $500K/month brief).
    monthly_spend: dict[str, float] = field(
        default_factory=lambda: {
            "google": 150_000.0,
            "meta": 175_000.0,
            "tiktok": 90_000.0,
            "email": 35_000.0,
            "referral": 50_000.0,
        }
    )

    # === GROUND TRUTH: true per-channel conversion importance ===
    # Higher => a touch on this channel raises conversion probability more.
    # Scenario baked in to match the brief: Meta over-credited by last-click
    # (high volume, low true lift); Google is a strong high-intent driver.
    channel_importance: dict[str, float] = field(
        default_factory=lambda: {
            "google": 0.34,    # high-intent search: true driver
            "meta": 0.14,      # high volume, last touch, modest true lift
            "tiktok": 0.10,    # awareness, weak direct lift
            "email": 0.28,     # strong nurture/retargeting lift
            "referral": 0.14,  # warm, high-intent but low volume
        }
    )

    # How often each channel APPEARS in journeys (exposure volume), which is
    # deliberately decoupled from true importance so last-touch is misled.
    channel_exposure_share: dict[str, float] = field(
        default_factory=lambda: {
            "google": 0.22,
            "meta": 0.40,      # Meta floods late-funnel => wins last-click
            "tiktok": 0.20,
            "email": 0.10,
            "referral": 0.08,
        }
    )

    # Where each channel tends to sit in the funnel: 0.0 = first thing a user
    # sees, 1.0 = last touch before converting. This governs journey ORDER only.
    # It is what makes the heuristic rules disagree with each other: without it
    # every channel is equally likely at every position, so first-touch,
    # last-touch and time-decay all collapse onto the same exposure ranking.
    #
    # Meta sits late on purpose -- high-volume retargeting that lands right
    # before the decision is exactly the profile that wins last-click credit it
    # did not earn, which is the scenario in the brief.
    channel_funnel_position: dict[str, float] = field(
        default_factory=lambda: {
            "tiktok": 0.20,    # awareness, top of funnel
            "referral": 0.30,  # word of mouth, early
            "google": 0.55,    # intent search, middle
            "email": 0.65,     # nurture, mid-late
            "meta": 0.85,      # retargeting, closes the journey
        }
    )
    # Spread around each channel's funnel position. At 0 the ordering would be
    # deterministic by channel, which is not how real journeys look; this keeps
    # the tendency while leaving plenty of overlap.
    funnel_position_noise: float = 0.25

    baseline_conversion: float = 0.045   # base prob before channel lift
    max_touches: int = 8
    revenue_monthly: float = 12.0        # $/mo plan
    revenue_annual: float = 96.0         # discounted annual plan
    annual_plan_share: float = 0.30
    refund_rate: float = 0.06

    days_of_history: int = 90


@dataclass(frozen=True)
class AttributionConfig:
    """Parameters for the attribution layer (Phase 2 rules, Phase 3 DDM).

    Kept separate from the DGP so the attribution models never inherit a
    parameter from the generator -- they only see the event log.
    """

    # Lookback window: touches older than this before the conversion are not
    # credited. Set wider than the DGP's 30-day touch spread so no converting
    # journey is truncated to empty; narrowing it is a modelling choice, not a
    # data artifact.
    lookback_days: int = 45

    # Time-decay: credit halves every ``half_life_days`` before conversion.
    # 7 days is the common industry default (GA4, Adobe).
    half_life_days: float = 7.0

    # Position-based (a.k.a. U-shaped) weights: first / middle-pool / last.
    position_first: float = 0.40
    position_middle: float = 0.20
    position_last: float = 0.40

    # Rule models evaluated in the comparison table, in presentation order.
    rule_models: tuple[str, ...] = (
        "first_touch",
        "last_touch",
        "linear",
        "time_decay",
        "position_based",
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters for the injected paywall experiment (validated in Phase 4-5)."""

    experiment_id: str = "annual_paywall_v1"
    seed: int = 7
    salt: str = "streamly-annual-paywall-2026"

    # True injected effects -- the experimentation engine must recover these.
    control_conversion: float = 0.080       # baseline paywall conversion
    true_treatment_lift_abs: float = 0.012  # +1.2pp true lift (annual paywall)
    true_traffic_split: float = 0.50        # intended 50/50 -> SRM must pass

    # Guardrails: treatment must NOT regress these.
    control_refund_rate: float = 0.06
    treatment_refund_rate: float = 0.066    # slight, within-tolerance drift
    control_latency_ms: float = 220.0
    treatment_latency_ms: float = 224.0

    # Pre-experiment covariate correlated w/ outcome -> powers CUPED.
    cuped_corr: float = 0.6


DATAGEN = DataGenConfig()
ATTRIBUTION = AttributionConfig()
EXPERIMENT = ExperimentConfig()
