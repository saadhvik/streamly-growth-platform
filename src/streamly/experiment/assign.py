"""Deterministic hash-based variant assignment.

Assignment is a pure function of ``(salt, unit_id)`` -- no database, no random
state, no assignment-time write. That buys three properties that matter more
than they sound:

1. **Reproducible.** Re-running analysis months later re-derives the exact same
   buckets. An assignment table can drift or be backfilled; a hash cannot.
2. **Stateless and idempotent.** Any service can compute a user's variant
   without a lookup, so a retry or a cache miss cannot flip someone's
   experience mid-session.
3. **Independent across experiments.** The salt is mixed *into* the hash rather
   than the seed, so two concurrently running experiments produce
   uncorrelated bucketings. Salting only the RNG seed (or reusing one salt) is
   the classic cause of experiments that silently confound each other.

SHA-256 is used for its avalanche property, not for security: flipping one bit
of ``unit_id`` redistributes the output uniformly, which is exactly what keeps
sequential user IDs from landing in alternating buckets.

The DGP in :mod:`streamly.datagen.dgp` mirrors this construction; the test suite
asserts the two agree bit-for-bit, so a change here that broke the warehouse's
recorded assignments would fail loudly.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

# Bits of the digest consumed. 64 bits gives ~1e-19 granularity -- far finer
# than any traffic split needs, and cheap.
_HASH_BITS = 64
_HASH_HEX = _HASH_BITS // 4
_HASH_MAX = float(2 ** _HASH_BITS)


@dataclass(frozen=True)
class VariantSpec:
    """A variant and its share of traffic."""

    name: str
    weight: float


def bucket_hash(unit_id: int | str, salt: str) -> float:
    """Map ``(salt, unit_id)`` to a deterministic float in [0, 1).

    The literal input is ``f"{salt}:{unit_id}"``. That exact format is a
    contract -- changing it re-randomizes every running experiment -- so it is
    pinned here and asserted in the tests.
    """
    digest = hashlib.sha256(f"{salt}:{unit_id}".encode()).hexdigest()
    return int(digest[:_HASH_HEX], 16) / _HASH_MAX


def bucket_hashes(unit_ids: np.ndarray, salt: str) -> np.ndarray:
    """Vectorized :func:`bucket_hash` over an array of unit ids."""
    return np.array([bucket_hash(int(u), salt) for u in unit_ids], dtype=float)


def assign(
    unit_id: int | str,
    salt: str,
    split: float = 0.5,
    control_name: str = "control",
    treatment_name: str = "treatment",
) -> str:
    """Assign one unit to control or treatment.

    ``split`` is the **control** share: 0.5 is 50/50, 0.9 is a 10% treatment
    ramp. Units with hash < split go to control.
    """
    if not 0.0 <= split <= 1.0:
        raise ValueError(f"split must be in [0, 1], got {split}")
    return control_name if bucket_hash(unit_id, salt) < split else treatment_name


def assign_many(
    unit_ids: np.ndarray,
    salt: str,
    split: float = 0.5,
    control_name: str = "control",
    treatment_name: str = "treatment",
) -> np.ndarray:
    """Vectorized :func:`assign`."""
    if not 0.0 <= split <= 1.0:
        raise ValueError(f"split must be in [0, 1], got {split}")
    return np.where(bucket_hashes(unit_ids, salt) < split, control_name, treatment_name)


def assign_multivariate(unit_ids: np.ndarray, salt: str, variants: list[VariantSpec]) -> np.ndarray:
    """Assign to 2+ variants by cumulative weight.

    Weights must sum to 1. Ordering is significant and must stay stable: the
    boundaries are cumulative, so reordering variants reshuffles live users.
    """
    total = sum(v.weight for v in variants)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"variant weights must sum to 1.0, got {total}")
    if any(v.weight < 0 for v in variants):
        raise ValueError("variant weights must be non-negative")

    h = bucket_hashes(unit_ids, salt)
    edges = np.cumsum([v.weight for v in variants])
    idx = np.searchsorted(edges, h, side="right")
    idx = np.clip(idx, 0, len(variants) - 1)
    return np.array([variants[i].name for i in idx], dtype=object)


def observed_split(variants: np.ndarray, control_name: str = "control") -> float:
    """Realized control share -- the input to the SRM check in Phase 5."""
    return float((variants == control_name).mean())
