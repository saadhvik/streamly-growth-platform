"""Group-sequential testing with Lan-DeMets alpha spending.

The problem this solves
-----------------------
Teams peek. They watch dashboards daily and stop the moment the p-value dips
below 0.05. Each look is another chance to cross the threshold, so the real
false-positive rate compounds: five equally spaced looks at a fixed alpha=0.05
produce a Type-I error rate near 14%, not 5%. Roughly one "win" in seven from a
peeking team is noise. Telling people not to peek does not work; giving them
boundaries that make peeking valid does.

How it works
------------
An alpha-*spending function* alpha(t) allocates the total false-positive budget
across information fraction t in (0, 1]. At each look, only the incremental
budget alpha(t_k) - alpha(t_{k-1}) may be spent, so the boundaries are strict
early and relax toward the nominal critical value at the end.

* **O'Brien-Fleming** spends almost nothing early -- an early stop demands
  overwhelming evidence -- and preserves nearly the full alpha for the final
  look. This is the default: it costs very little power versus a fixed-sample
  test, so it is close to free insurance.
* **Pocock** spends evenly, stopping earlier on average but paying a real
  penalty at the final analysis. Appropriate when the cost of running long is
  high and the effect is expected to be large.

Boundaries are computed **exactly**, by numerically propagating the joint
density of the test statistic across looks (the Armitage-McPherson recursion
underlying Lan-DeMets), not by a closed-form approximation. The Brownian-motion
structure -- independent increments with variance equal to the information
increment -- is what makes this valid.

Scope
-----
These are efficacy boundaries only. Futility (non-binding beta-spending)
boundaries are a natural extension but are deliberately not implemented rather
than implemented approximately.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

# Grid for the density recursion. +/-8 SD covers the standard normal to ~1e-15,
# and 2001 points puts the trapezoid error well below the 1e-6 bisection
# tolerance on the boundaries.
_GRID_LIMIT = 8.0
_GRID_POINTS = 2001

SpendingFn = Callable[[float, float], float]


@dataclass(frozen=True)
class SequentialPlan:
    """Efficacy boundaries for a pre-specified sequence of looks."""

    information_fractions: tuple[float, ...]
    z_boundaries: tuple[float, ...]        # |Z| >= boundary -> stop for efficacy
    nominal_alphas: tuple[float, ...]      # two-sided p-value threshold per look
    cumulative_alpha_spent: tuple[float, ...]
    incremental_alpha: tuple[float, ...]
    alpha: float
    spending: str

    def __len__(self) -> int:
        return len(self.information_fractions)

    def __str__(self) -> str:
        rows = "\n".join(
            f"  look {i + 1}: t={t:.2f}  |Z| >= {z:.3f}  (nominal p < {p:.5f}, "
            f"cumulative alpha {c:.4f})"
            for i, (t, z, p, c) in enumerate(
                zip(self.information_fractions, self.z_boundaries,
                    self.nominal_alphas, self.cumulative_alpha_spent)
            )
        )
        return f"{self.spending} spending, total alpha={self.alpha}\n{rows}"


@dataclass(frozen=True)
class SequentialDecision:
    """The outcome of evaluating observed statistics against a plan."""

    look: int                    # 1-indexed look evaluated
    z: float
    boundary: float
    crossed: bool
    nominal_p: float
    stop: bool
    reason: str

    def __str__(self) -> str:
        return (
            f"look {self.look}: Z={self.z:+.3f} vs boundary {self.boundary:.3f} "
            f"-> {self.reason}"
        )


def obrien_fleming_spending(t: float, alpha: float) -> float:
    """Lan-DeMets O'Brien-Fleming spending: ``2 - 2*Phi(z_{alpha/2} / sqrt(t))``."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return float(2.0 - 2.0 * stats.norm.cdf(z / np.sqrt(t)))


def pocock_spending(t: float, alpha: float) -> float:
    """Lan-DeMets Pocock spending: ``alpha * ln(1 + (e - 1) t)``."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha
    return float(alpha * np.log(1.0 + (np.e - 1.0) * t))


def linear_spending(t: float, alpha: float) -> float:
    """Uniform spending, ``alpha * t`` -- included mainly as a reference case."""
    return float(alpha * min(max(t, 0.0), 1.0))


SPENDING_FUNCTIONS: dict[str, SpendingFn] = {
    "obrien_fleming": obrien_fleming_spending,
    "pocock": pocock_spending,
    "linear": linear_spending,
}


def _exit_probability(density: np.ndarray, grid: np.ndarray, bound_b: float) -> float:
    """Mass of the sub-density falling outside +/-``bound_b`` (B-value scale)."""
    outside = np.abs(grid) >= bound_b
    return float(np.trapezoid(np.where(outside, density, 0.0), grid))


def compute_boundaries(
    information_fractions: tuple[float, ...] | list[float],
    alpha: float = 0.05,
    spending: str = "obrien_fleming",
) -> SequentialPlan:
    """Solve for the efficacy boundary at each look.

    The recursion tracks the sub-density of the B-value ``B(t) = Z(t) * sqrt(t)``
    restricted to paths that have not yet crossed a boundary. Under the null,
    B has independent increments with ``Var = t_k - t_{k-1}``, so each step is a
    Gaussian convolution of the surviving density; the boundary at look ``k`` is
    the value whose exceedance probability equals that look's incremental alpha,
    found by bisection.

    Parameters
    ----------
    information_fractions:
        Strictly increasing values in (0, 1]; ``(0.2, 0.4, 0.6, 0.8, 1.0)`` is
        five equally spaced looks. These must be **pre-specified**: choosing
        look times after seeing data reintroduces the very bias being removed.
    """
    t = tuple(float(x) for x in information_fractions)
    if not t:
        raise ValueError("at least one look is required")
    if any(x <= 0 or x > 1 for x in t):
        raise ValueError("information fractions must lie in (0, 1]")
    if any(b <= a for a, b in zip(t, t[1:])):
        raise ValueError("information fractions must be strictly increasing")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    try:
        spend = SPENDING_FUNCTIONS[spending]
    except KeyError:
        raise ValueError(
            f"unknown spending function {spending!r}; "
            f"expected one of {sorted(SPENDING_FUNCTIONS)}"
        ) from None

    cumulative = [spend(x, alpha) for x in t]
    # Guard against a non-monotone user-supplied spending function.
    for a, b in zip(cumulative, cumulative[1:]):
        if b < a - 1e-12:
            raise ValueError("spending function must be non-decreasing in t")
    incremental = [cumulative[0]] + [b - a for a, b in zip(cumulative, cumulative[1:])]

    grid = np.linspace(-_GRID_LIMIT, _GRID_LIMIT, _GRID_POINTS)
    boundaries: list[float] = []

    # Look 1: B(t_1) ~ N(0, t_1), nothing has been truncated yet.
    density = stats.norm.pdf(grid, scale=np.sqrt(t[0]))

    for k, t_k in enumerate(t):
        if k > 0:
            # Propagate the surviving density across the information increment.
            delta = t_k - t[k - 1]
            kernel = stats.norm.pdf(
                grid[:, None] - grid[None, :], scale=np.sqrt(delta)
            )
            density = np.trapezoid(kernel * density[None, :], grid, axis=1)

        target = incremental[k]
        # Bisect on the B-scale boundary; exit probability decreases in b.
        lo, hi = 0.0, _GRID_LIMIT
        if _exit_probability(density, grid, hi) > target:
            b_k = hi                       # cannot spend this little; clamp
        else:
            for _ in range(200):
                mid = (lo + hi) / 2.0
                if _exit_probability(density, grid, mid) > target:
                    lo = mid
                else:
                    hi = mid
            b_k = hi
        boundaries.append(b_k / np.sqrt(t_k))     # convert B-scale -> Z-scale

        # Truncate: only non-crossing paths continue to the next look.
        density = np.where(np.abs(grid) >= b_k, 0.0, density)

    z_bounds = tuple(float(z) for z in boundaries)
    return SequentialPlan(
        information_fractions=t,
        z_boundaries=z_bounds,
        nominal_alphas=tuple(float(2.0 * stats.norm.sf(z)) for z in z_bounds),
        cumulative_alpha_spent=tuple(float(c) for c in cumulative),
        incremental_alpha=tuple(float(i) for i in incremental),
        alpha=alpha,
        spending=spending,
    )


def evaluate_look(plan: SequentialPlan, look: int, z: float) -> SequentialDecision:
    """Compare an observed z-statistic against the boundary for a given look.

    ``look`` is 1-indexed. Crossing at a non-final look means stop and declare
    an effect; not crossing at the final look means stop and declare no effect.
    """
    if not 1 <= look <= len(plan):
        raise ValueError(f"look must be in 1..{len(plan)}, got {look}")
    boundary = plan.z_boundaries[look - 1]
    crossed = abs(z) >= boundary
    is_final = look == len(plan)

    if crossed:
        reason = "STOP: efficacy boundary crossed"
    elif is_final:
        reason = "STOP: final look, no effect detected"
    else:
        reason = "CONTINUE: insufficient evidence to stop"

    return SequentialDecision(
        look=look,
        z=float(z),
        boundary=float(boundary),
        crossed=bool(crossed),
        nominal_p=float(2.0 * stats.norm.sf(abs(z))),
        stop=bool(crossed or is_final),
        reason=reason,
    )


def evaluate_sequence(plan: SequentialPlan, z_values: list[float] | np.ndarray) -> SequentialDecision:
    """Walk a run of looks in order and return the decision that stops it."""
    z_values = list(z_values)
    if len(z_values) > len(plan):
        raise ValueError(f"got {len(z_values)} looks for a {len(plan)}-look plan")
    decision = None
    for i, z in enumerate(z_values, start=1):
        decision = evaluate_look(plan, i, z)
        if decision.crossed:
            return decision
    assert decision is not None
    return decision


def naive_peeking_type_one_error(
    n_looks: int, alpha: float = 0.05, simulations: int = 20_000, seed: int = 0
) -> float:
    """Simulate the Type-I error of peeking at a fixed alpha with no correction.

    Provided as an in-repo demonstration of the problem the boundaries solve --
    the number quoted in the trustworthy-experimentation one-pager comes from
    here, not from a citation.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(1, n_looks + 1) / n_looks
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    increments = rng.normal(
        0.0, np.sqrt(np.diff(np.concatenate([[0.0], t]))), size=(simulations, n_looks)
    )
    b = np.cumsum(increments, axis=1)
    z = b / np.sqrt(t)
    return float((np.abs(z) >= crit).any(axis=1).mean())
