"""Shapley-value attribution over channel coalitions.

Model
-----
Each journey is reduced to the **set** of channels it contains. The
characteristic function is the conversion *rate* of every journey whose channel
set is contained in the coalition:

    v(C) = P(convert | channel_set ⊆ C),    v(∅) = 0

Using a rate rather than a conversion count matters: counts make v(C) grow with
how many users happened to be exposed to C, which smuggles reach back into a
method whose entire purpose is to be reach-independent.

The Shapley value is the average marginal contribution of a channel across all
orderings of the full channel set:

    φ_c = Σ_{C ⊆ N\\{c}}  |C|!(n-|C|-1)!/n!  · [ v(C ∪ {c}) - v(C) ]

With five channels this is 32 coalitions -- computed **exactly**, no Monte
Carlo sampling, so the result is deterministic.

Why this is the honest counterfactual
-------------------------------------
Shapley is the unique allocation satisfying efficiency, symmetry, null-player,
and additivity. Practically: two channels with identical marginal effects get
identical credit regardless of how often each was fired, which is exactly the
property last-touch lacks.

Known bias, stated plainly
--------------------------
Zero-touch users are unobservable in a marketing log, so v(∅) cannot be
estimated from the data and is fixed at 0. Efficiency then forces the organic
baseline -- conversions that would have happened with no marketing at all -- to
be split across channels rather than excluded. This compresses every share
toward 1/n and shrinks the spread between strong and weak channels. It biases
*against* the finding this analysis reports (it flatters weak channels like
meta), so the measured over-crediting of meta is a conservative floor. Removing
the bias needs an unexposed holdout, which is a media-plan change, not a
modelling one.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial

import numpy as np

from streamly.attribution.sessionize import JourneySet
from streamly.config import CHANNELS


@dataclass(frozen=True)
class ShapleyResult:
    """Shapley values and the shares derived from them."""

    channels: tuple[str, ...]
    shapley_value: dict[str, float]       # marginal conversion-rate contribution
    attribution: dict[str, float]         # normalized (sums to 1)
    coalition_value: dict[frozenset[str], float]
    coalition_support: dict[frozenset[str], int]   # journeys behind each v(C)


def coalition_values(
    js: JourneySet, channels: tuple[str, ...] = CHANNELS
) -> tuple[dict[frozenset[str], float], dict[frozenset[str], int]]:
    """Conversion rate of journeys whose channel set is a subset of each coalition.

    Returns ``(values, support)``; ``support`` is the journey count behind each
    rate, exposed so thin coalitions can be spotted rather than trusted blindly.
    """
    # Reduce every journey to (channel set, converted) once.
    set_conv: dict[frozenset[str], list[int]] = {}
    for i in range(len(js)):
        path = js.path(i)
        if not path:
            continue
        key = frozenset(path)
        rec = set_conv.setdefault(key, [0, 0])
        rec[0] += 1
        rec[1] += int(js.converted[i])

    values: dict[frozenset[str], float] = {}
    support: dict[frozenset[str], int] = {}
    for size in range(len(channels) + 1):
        for combo in combinations(channels, size):
            coalition = frozenset(combo)
            n = c = 0
            for key, (cnt, conv) in set_conv.items():
                if key <= coalition:
                    n += cnt
                    c += conv
            support[coalition] = n
            values[coalition] = (c / n) if n else 0.0
    values[frozenset()] = 0.0   # zero-touch users are unobservable; see module docstring
    support[frozenset()] = 0
    return values, support


def shapley_attribution(
    js: JourneySet, channels: tuple[str, ...] = CHANNELS
) -> ShapleyResult:
    """Exact Shapley values over all 2^n channel coalitions."""
    values, support = coalition_values(js, channels)
    n = len(channels)
    phi = {c: 0.0 for c in channels}

    for c in channels:
        others = [x for x in channels if x != c]
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for combo in combinations(others, size):
                coalition = frozenset(combo)
                marginal = values[coalition | {c}] - values[coalition]
                phi[c] += weight * marginal

    total = sum(phi.values())
    if total <= 0:
        raise ValueError("total Shapley value is non-positive; cannot normalize shares")
    # Negative marginals are possible from sampling noise in thin coalitions;
    # clip at zero before normalizing so shares stay a valid distribution.
    clipped = {c: max(v, 0.0) for c, v in phi.items()}
    denom = sum(clipped.values())
    attribution = {c: clipped[c] / denom for c in channels}

    return ShapleyResult(
        channels=channels,
        shapley_value=phi,
        attribution=attribution,
        coalition_value=values,
        coalition_support=support,
    )


def efficiency_residual(res: ShapleyResult) -> float:
    """Check the efficiency axiom: Σφ_c must equal v(N) - v(∅).

    Returned as an absolute residual so tests and the app can assert on it.
    """
    grand = res.coalition_value[frozenset(res.channels)]
    return float(abs(sum(res.shapley_value.values()) - (grand - res.coalition_value[frozenset()])))


def journeys_channel_sets(js: JourneySet) -> np.ndarray:
    """Distinct channel-set count per journey -- a coverage diagnostic."""
    return np.array([len(set(js.path(i))) for i in range(len(js))])
