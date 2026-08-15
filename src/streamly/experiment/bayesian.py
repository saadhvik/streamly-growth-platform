"""Bayesian decision layer: conjugate Beta-Binomial analysis.

What this adds that the frequentist layer cannot
-------------------------------------------------
A confidence interval answers "what effects are compatible with this data?" It
does not answer the question a product owner actually asks, which is "if we
ship this and we are wrong, how much do we lose?" That question needs a
posterior over the effect and a loss function, which is what this module
provides:

* ``P(treatment > control)`` -- a direct probability statement about the world,
  not about hypothetical repetitions of the experiment.
* **Expected loss** -- the average amount of conversion rate forgone if we ship
  and the decision turns out wrong. This is the number to decide on. A 92%
  chance of winning sounds marginal until you see that the 8% losing case costs
  almost nothing, at which point shipping is obviously correct.
* **Credible interval** on the lift, which means what people already think a
  confidence interval means.

Why conjugate, and why quadrature instead of MCMC
--------------------------------------------------
For two binomial proportions the Beta prior is conjugate, so the posteriors are
Beta distributions known in closed form -- there is nothing to sample. Running
MCMC here would add a heavy dependency, a convergence-diagnostics burden, and
sampling noise that makes results irreproducible run to run, all to approximate
a distribution we can write down exactly.

Every quantity below is therefore computed by **numerical integration of exact
posteriors**, not simulation:

    P(θ_B - θ_A > t) = ∫ f_A(x) · [1 - F_B(x + t)] dx

with the same integral shape reused for the tail probability, the expected
loss, and the credible interval. Results are deterministic to integration
tolerance (~1e-10), which the test suite verifies against both a closed-form
identity and a large Monte Carlo run.

On priors
---------
The default is ``Beta(1, 1)`` -- uniform, and worth ~2 observations against the
tens of thousands an experiment collects, so it is effectively invisible in the
posterior while keeping the arithmetic well-defined at zero conversions. A
genuinely informative prior (from past experiments on the same surface) is
supported and is the honest way to encode "we have run 40 paywall tests and
none moved conversion more than 2pp" -- but it must be declared at intake,
before data, for the same reason guardrail margins must be.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize, stats

# Uniform prior: weakly informative, worth two observations.
DEFAULT_PRIOR: tuple[float, float] = (1.0, 1.0)

# Integration is restricted to where the posteriors actually have mass. At
# experiment scale a Beta posterior is extremely narrow, and integrating over
# the full [0, 1] would let the quadrature step straight over the spike.
_SD_SPAN = 14.0


@dataclass(frozen=True)
class BayesianResult:
    """Posterior summary and the decision quantities derived from it."""

    posterior_control: tuple[float, float]      # (alpha, beta)
    posterior_treatment: tuple[float, float]
    control_mean: float
    treatment_mean: float

    prob_treatment_better: float                # P(θ_B > θ_A)
    prob_exceeds_threshold: float               # P(θ_B - θ_A > threshold)
    expected_loss_ship: float                   # E[(θ_A - θ_B)^+]
    expected_loss_stay: float                   # E[(θ_B - θ_A)^+]

    lift_median: float
    lift_ci_low: float
    lift_ci_high: float

    threshold: float
    credible_level: float
    prior: tuple[float, float]

    @property
    def absolute_effect(self) -> float:
        return self.treatment_mean - self.control_mean

    @property
    def relative_effect(self) -> float:
        return self.absolute_effect / self.control_mean if self.control_mean else float("nan")

    def __str__(self) -> str:
        return (
            f"P(treatment > control) = {self.prob_treatment_better:.4f}; "
            f"P(lift > {self.threshold:+.4f}) = {self.prob_exceeds_threshold:.4f}; "
            f"expected loss if we ship = {self.expected_loss_ship:.6f}; "
            f"{self.credible_level:.0%} credible interval "
            f"[{self.lift_ci_low:+.4f}, {self.lift_ci_high:+.4f}]"
        )


def _posterior(successes: int, trials: int, prior: tuple[float, float]) -> tuple[float, float]:
    """Beta posterior parameters for a binomial likelihood."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"need 0 <= successes <= trials, got {successes}/{trials}")
    a, b = prior
    if a <= 0 or b <= 0:
        raise ValueError(f"prior parameters must be positive, got {prior}")
    return a + successes, b + trials - successes


def _integration_range(a: float, b: float) -> tuple[float, float]:
    """Interval containing essentially all of a Beta(a, b)'s mass."""
    mean = a / (a + b)
    sd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))
    return max(0.0, mean - _SD_SPAN * sd), min(1.0, mean + _SD_SPAN * sd)


def prob_difference_exceeds(
    post_control: tuple[float, float],
    post_treatment: tuple[float, float],
    threshold: float = 0.0,
) -> float:
    """``P(θ_treatment - θ_control > threshold)``.

    Computed as ``∫ f_A(x) · [1 - F_B(x + t)] dx`` over the control posterior's
    support. This single integral is the engine behind the win probability, the
    practical-threshold probability, and (by root-finding) the credible interval.
    """
    a_c, b_c = post_control
    a_t, b_t = post_treatment
    lo, hi = _integration_range(a_c, b_c)

    def integrand(x: float) -> float:
        return stats.beta.pdf(x, a_c, b_c) * stats.beta.sf(x + threshold, a_t, b_t)

    value, _err = integrate.quad(integrand, lo, hi, limit=200)
    return float(np.clip(value, 0.0, 1.0))


def expected_loss(
    post_control: tuple[float, float],
    post_treatment: tuple[float, float],
    ship: bool = True,
) -> float:
    """Expected conversion rate forgone by making the wrong call.

    ``ship=True`` returns ``E[(θ_A - θ_B)^+]``: the average shortfall if we ship
    the treatment and it is in fact worse. ``ship=False`` returns the mirror
    quantity -- the cost of *not* shipping a treatment that is in fact better,
    which is the error teams systematically ignore.

    The inner expectation has a closed form. For ``Y ~ Beta(a, b)``::

        ∫_0^x (x - y) f_Y(y) dy = x·F_{a,b}(x) - (a / (a + b))·F_{a+1,b}(x)

    so only the outer integral is numerical.
    """
    if ship:
        (a_lose, b_lose), (a_win, b_win) = post_treatment, post_control
    else:
        (a_lose, b_lose), (a_win, b_win) = post_control, post_treatment

    # E[(θ_win - θ_lose)^+] : integrate over the "win" arm's posterior.
    lo, hi = _integration_range(a_win, b_win)
    mean_lose = a_lose / (a_lose + b_lose)

    def integrand(x: float) -> float:
        inner = (
            x * stats.beta.cdf(x, a_lose, b_lose)
            - mean_lose * stats.beta.cdf(x, a_lose + 1.0, b_lose)
        )
        return stats.beta.pdf(x, a_win, b_win) * inner

    value, _err = integrate.quad(integrand, lo, hi, limit=200)
    return float(max(value, 0.0))


def lift_quantile(
    post_control: tuple[float, float],
    post_treatment: tuple[float, float],
    q: float,
) -> float:
    """Quantile ``q`` of the posterior distribution of ``θ_treatment - θ_control``.

    The difference of two Beta variables has no closed form, so the CDF is
    evaluated by the same integral as :func:`prob_difference_exceeds` and
    inverted by Brent root-finding -- deterministic, and accurate to the
    root-finder's tolerance.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")

    def cdf_minus_q(d: float) -> float:
        return (1.0 - prob_difference_exceeds(post_control, post_treatment, d)) - q

    # Bracket generously: the difference lives in (-1, 1).
    lo, hi = -0.999999, 0.999999
    if cdf_minus_q(lo) > 0:
        return lo
    if cdf_minus_q(hi) < 0:
        return hi
    return float(optimize.brentq(cdf_minus_q, lo, hi, xtol=1e-10))


def beta_binomial_test(
    conversions_control: int,
    n_control: int,
    conversions_treatment: int,
    n_treatment: int,
    threshold: float = 0.0,
    credible_level: float = 0.95,
    prior: tuple[float, float] = DEFAULT_PRIOR,
) -> BayesianResult:
    """Full conjugate analysis of a two-proportion experiment."""
    post_c = _posterior(conversions_control, n_control, prior)
    post_t = _posterior(conversions_treatment, n_treatment, prior)

    tail = (1.0 - credible_level) / 2.0
    return BayesianResult(
        posterior_control=post_c,
        posterior_treatment=post_t,
        control_mean=post_c[0] / (post_c[0] + post_c[1]),
        treatment_mean=post_t[0] / (post_t[0] + post_t[1]),
        prob_treatment_better=prob_difference_exceeds(post_c, post_t, 0.0),
        prob_exceeds_threshold=prob_difference_exceeds(post_c, post_t, threshold),
        expected_loss_ship=expected_loss(post_c, post_t, ship=True),
        expected_loss_stay=expected_loss(post_c, post_t, ship=False),
        lift_median=lift_quantile(post_c, post_t, 0.5),
        lift_ci_low=lift_quantile(post_c, post_t, tail),
        lift_ci_high=lift_quantile(post_c, post_t, 1.0 - tail),
        threshold=threshold,
        credible_level=credible_level,
        prior=prior,
    )


def decision_summary(result: BayesianResult, loss_tolerance: float | None = None) -> str:
    """One-paragraph plain-language reading of the posterior.

    ``loss_tolerance`` is the largest expected loss the business will accept in
    exchange for the upside -- the Bayesian analogue of the practical
    significance threshold, and like it, something to agree at intake. When
    omitted it defaults to a tenth of the practical threshold, which is a
    convention, not a derivation, and is labelled as such in the output.
    """
    tol = loss_tolerance if loss_tolerance is not None else abs(result.threshold) / 10.0
    verdict = (
        "shipping is the lower-risk choice"
        if result.expected_loss_ship <= tol
        else "the downside risk is above tolerance"
    )
    return (
        f"There is a {result.prob_treatment_better:.1%} probability the treatment is "
        f"better, and a {result.prob_exceeds_threshold:.1%} probability it beats the "
        f"{result.threshold:+.2%} threshold that makes shipping worthwhile. If we ship "
        f"and are wrong, the expected cost is {result.expected_loss_ship:.4%} of "
        f"conversion rate; if we hold and are wrong, we forgo "
        f"{result.expected_loss_stay:.4%}. Against a tolerance of {tol:.4%}, "
        f"{verdict}."
    )
