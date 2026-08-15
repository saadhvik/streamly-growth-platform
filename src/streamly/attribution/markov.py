"""Markov-chain attribution via the removal effect.

Model
-----
Journeys are treated as realizations of a first-order Markov chain over states

    start -> {google, meta, tiktok, email, referral}* -> {conversion | null}

Transitions are counted across the **entire** population, converters and
non-converters alike -- the null-absorbing state is what makes the chain a model
of conversion *probability* rather than of conversion volume.

The removal effect of channel ``c`` is the relative drop in the chain's
start-to-conversion probability when ``c`` is deleted and every transition into
it is rerouted to null:

    RE(c) = (P(conv) - P(conv | c removed)) / P(conv)

Normalized removal effects are the channel's attribution share.

What this method can and cannot see
-----------------------------------
The removal effect is a *reachability* counterfactual: it asks what happens if
a channel disappears entirely, which conflates a channel's **reach** with its
**incrementality**. A channel touching 80% of journeys scores a large removal
effect even when its per-touch lift is small, because deleting it strands most
paths. That is a real property of the estimator, not an implementation
artifact, and it is why this module ships alongside :mod:`shapley` rather than
alone -- see :mod:`streamly.attribution.validate` for the measured consequence.

Computation is exact, not simulated: the chain is solved as an absorbing Markov
chain via ``(I - Q)^-1 R``, so results are deterministic and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from streamly.attribution.sessionize import JourneySet
from streamly.config import CHANNELS

_START = "(start)"
_CONV = "(conversion)"
_NULL = "(null)"


@dataclass(frozen=True)
class MarkovResult:
    """Removal effects and the attribution shares derived from them."""

    channels: tuple[str, ...]
    base_conversion_prob: float          # chain P(start -> conversion)
    removal_effect: dict[str, float]     # relative drop when channel is removed
    attribution: dict[str, float]        # normalized removal effects (sums to 1)
    transition_matrix: np.ndarray        # full row-stochastic matrix, for audit
    states: tuple[str, ...]


def transition_matrix(
    js: JourneySet, channels: tuple[str, ...] = CHANNELS
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Estimate the row-stochastic first-order transition matrix.

    State order is ``(start, *channels, conversion, null)``; the two absorbing
    states are last so the transient block ``Q`` is a leading submatrix.
    """
    states = (_START, *channels, _CONV, _NULL)
    index = {s: i for i, s in enumerate(states)}
    n = len(states)
    counts = np.zeros((n, n), dtype=float)

    for i in range(len(js)):
        path = js.path(i)
        if not path:
            continue
        prev = index[_START]
        for ch in path:
            cur = index[ch]
            counts[prev, cur] += 1.0
            prev = cur
        counts[prev, index[_CONV] if js.converted[i] else index[_NULL]] += 1.0

    # Absorbing states self-loop with probability 1.
    counts[index[_CONV], index[_CONV]] = 1.0
    counts[index[_NULL], index[_NULL]] = 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    # A state never visited would divide by zero; send it to null instead.
    dead = (row_sums.ravel() == 0.0)
    counts[dead, index[_NULL]] = 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums, states


def _absorption_probability(
    trans: np.ndarray, states: tuple[str, ...], drop: set[str] | None = None
) -> float:
    """P(start -> conversion), optionally with some channel states removed.

    Removed states are deleted from the chain and all their inbound probability
    mass is rerouted to null -- the standard removal-effect construction.
    """
    drop = drop or set()
    index = {s: i for i, s in enumerate(states)}
    conv_i, null_i = index[_CONV], index[_NULL]

    m = trans.copy()
    for s in drop:
        j = index[s]
        m[:, null_i] += m[:, j]   # inbound mass -> null
        m[:, j] = 0.0
        m[j, :] = 0.0
        m[j, null_i] = 1.0        # keep the row stochastic and absorbing

    transient = [i for i in range(len(states)) if i not in (conv_i, null_i)]
    q = m[np.ix_(transient, transient)]
    r = m[np.ix_(transient, [conv_i])]

    # Fundamental matrix N = (I - Q)^-1; absorption probabilities B = N R.
    fundamental = np.linalg.solve(np.eye(len(transient)) - q, r)
    start_row = transient.index(index[_START])
    return float(fundamental[start_row, 0])


def markov_attribution(
    js: JourneySet, channels: tuple[str, ...] = CHANNELS
) -> MarkovResult:
    """Compute removal effects and normalized Markov attribution shares."""
    trans, states = transition_matrix(js, channels)
    base = _absorption_probability(trans, states)
    if base <= 0:
        raise ValueError("chain has zero conversion probability; cannot attribute")

    removal = {}
    for c in channels:
        without = _absorption_probability(trans, states, drop={c})
        removal[c] = (base - without) / base

    total = sum(removal.values())
    if total <= 0:
        raise ValueError("all removal effects are zero; chain is degenerate")
    attribution = {c: removal[c] / total for c in channels}

    return MarkovResult(
        channels=channels,
        base_conversion_prob=base,
        removal_effect=removal,
        attribution=attribution,
        transition_matrix=trans,
        states=states,
    )
