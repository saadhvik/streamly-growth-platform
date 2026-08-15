"""Phase 8 acceptance tests: the conjugate Bayesian decision layer.

Every quantity is validated against an *independent* computation, not against
itself:

* ``P(B > A)`` against a large Monte Carlo draw from the same posteriors, and
  against the exact analytic answer for a symmetric case.
* Expected loss against Monte Carlo, and against the identity
  ``E[(B-A)^+] - E[(A-B)^+] = E[B] - E[A]``, which must hold exactly.
* The credible interval against Monte Carlo quantiles, and against the
  frequentist interval, which a flat prior must reproduce at large n.

The point of using quadrature rather than MCMC is determinism, so there is also
a test that the whole analysis is bit-for-bit reproducible.
"""
from __future__ import annotations

import numpy as np
import pytest

from streamly.experiment import bayesian as by
from streamly.experiment.frequentist import two_proportion_z_test

# The reference paywall experiment's actual counts.
REF = (2_370, 29_967, 2_661, 30_033)


@pytest.fixture(scope="module")
def reference() -> by.BayesianResult:
    return by.beta_binomial_test(*REF, threshold=0.005)


@pytest.fixture(scope="module")
def mc_draws(reference: by.BayesianResult) -> np.ndarray:
    """4M draws from the same posteriors -- the independent check."""
    rng = np.random.default_rng(0)
    n = 4_000_000
    a = rng.beta(*reference.posterior_control, n)
    b = rng.beta(*reference.posterior_treatment, n)
    return b - a


# ---------------------------------------------------------------------------
# Posterior construction
# ---------------------------------------------------------------------------
def test_posterior_is_the_conjugate_update() -> None:
    result = by.beta_binomial_test(30, 100, 40, 100)
    assert result.posterior_control == (1.0 + 30, 1.0 + 70)
    assert result.posterior_treatment == (1.0 + 40, 1.0 + 60)
    assert result.control_mean == pytest.approx(31 / 102)


def test_a_flat_prior_barely_moves_a_large_sample() -> None:
    """Beta(1,1) shifts the posterior mean toward 0.5 by exactly (1-2p)/n.

    Asserting the analytic shift rather than a hand-picked tolerance: the pull
    is real and should be checked, not waved through. At n≈30,000 and p≈0.079 it
    is 2.8e-5 — three orders of magnitude below the 0.0095 effect being
    measured, so it is negligible *for this decision*, which is the claim worth
    making.
    """
    result = by.beta_binomial_test(*REF)
    x_c, n_c, x_t, n_t = REF
    assert result.control_mean == pytest.approx((x_c + 1) / (n_c + 2), rel=1e-12)
    assert result.treatment_mean == pytest.approx((x_t + 1) / (n_t + 2), rel=1e-12)

    shift = result.control_mean - x_c / n_c
    assert shift == pytest.approx((1 - 2 * x_c / n_c) / n_c, rel=0.01)
    assert abs(shift) < result.absolute_effect / 100


def test_an_informative_prior_shrinks_a_small_sample() -> None:
    """A prior worth 200 observations must visibly pull a 20-observation result."""
    flat = by.beta_binomial_test(10, 20, 10, 20)
    informed = by.beta_binomial_test(10, 20, 10, 20, prior=(10.0, 190.0))
    assert flat.control_mean == pytest.approx(0.5, abs=0.02)
    assert informed.control_mean < 0.15, "an informative prior must dominate n=20"


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((11, 10, 1, 10), "successes <= trials"),
        ((-1, 10, 1, 10), "successes <= trials"),
    ],
)
def test_invalid_counts_are_rejected(args: tuple, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        by.beta_binomial_test(*args)


def test_a_non_positive_prior_is_rejected() -> None:
    with pytest.raises(ValueError, match="prior parameters must be positive"):
        by.beta_binomial_test(10, 100, 12, 100, prior=(0.0, 1.0))


# ---------------------------------------------------------------------------
# P(B > A)
# ---------------------------------------------------------------------------
def test_win_probability_matches_monte_carlo(
    reference: by.BayesianResult, mc_draws: np.ndarray
) -> None:
    assert reference.prob_treatment_better == pytest.approx(
        float((mc_draws > 0).mean()), abs=5e-4
    )


def test_threshold_probability_matches_monte_carlo(
    reference: by.BayesianResult, mc_draws: np.ndarray
) -> None:
    assert reference.prob_exceeds_threshold == pytest.approx(
        float((mc_draws > 0.005).mean()), abs=1e-3
    )


def test_identical_arms_give_a_coin_flip() -> None:
    """Exact analytic case: identical data must give P(B>A) = 0.5."""
    result = by.beta_binomial_test(500, 5_000, 500, 5_000)
    assert result.prob_treatment_better == pytest.approx(0.5, abs=1e-6)
    assert result.lift_median == pytest.approx(0.0, abs=1e-6)


def test_win_probability_is_monotone_in_the_treatment_count() -> None:
    probs = [
        by.beta_binomial_test(500, 10_000, x, 10_000).prob_treatment_better
        for x in (450, 500, 550, 600)
    ]
    assert probs == sorted(probs)
    assert probs[0] < 0.5 < probs[-1]


def test_raising_the_threshold_lowers_the_probability_of_clearing_it(
    reference: by.BayesianResult,
) -> None:
    assert reference.prob_exceeds_threshold < reference.prob_treatment_better


# ---------------------------------------------------------------------------
# Expected loss
# ---------------------------------------------------------------------------
def test_expected_loss_matches_monte_carlo(
    reference: by.BayesianResult, mc_draws: np.ndarray
) -> None:
    assert reference.expected_loss_ship == pytest.approx(
        float(np.maximum(-mc_draws, 0).mean()), abs=1e-6
    )
    assert reference.expected_loss_stay == pytest.approx(
        float(np.maximum(mc_draws, 0).mean()), abs=1e-5
    )


def test_expected_loss_satisfies_the_exact_identity() -> None:
    """E[(B-A)^+] - E[(A-B)^+] must equal E[B] - E[A], exactly.

    This is an algebraic identity, so any quadrature error shows up here
    directly -- a stronger check than agreeing with a simulation.
    """
    for counts in [(500, 10_000, 560, 10_000), (30, 200, 25, 200), (2, 50, 9, 50)]:
        r = by.beta_binomial_test(*counts)
        identity = r.expected_loss_stay - r.expected_loss_ship
        assert identity == pytest.approx(r.absolute_effect, abs=1e-9), counts


def test_losses_are_symmetric_for_identical_arms() -> None:
    r = by.beta_binomial_test(500, 5_000, 500, 5_000)
    assert r.expected_loss_ship == pytest.approx(r.expected_loss_stay, rel=1e-6)
    assert r.expected_loss_ship > 0, "uncertainty means neither choice is free"


def test_a_clear_winner_makes_shipping_nearly_costless(
    reference: by.BayesianResult,
) -> None:
    """The decision-theoretic payoff: high win probability, negligible downside."""
    assert reference.prob_treatment_better > 0.99
    assert reference.expected_loss_ship < 1e-5
    assert reference.expected_loss_stay > 100 * reference.expected_loss_ship


def test_expected_loss_grows_as_the_evidence_weakens() -> None:
    strong = by.beta_binomial_test(500, 10_000, 700, 10_000).expected_loss_ship
    weak = by.beta_binomial_test(50, 1_000, 70, 1_000).expected_loss_ship
    assert weak > strong, "less data must mean more risk in shipping"


# ---------------------------------------------------------------------------
# Credible interval
# ---------------------------------------------------------------------------
def test_credible_interval_matches_monte_carlo_quantiles(
    reference: by.BayesianResult, mc_draws: np.ndarray
) -> None:
    lo, mid, hi = np.quantile(mc_draws, [0.025, 0.5, 0.975])
    assert reference.lift_ci_low == pytest.approx(float(lo), abs=2e-5)
    assert reference.lift_median == pytest.approx(float(mid), abs=2e-5)
    assert reference.lift_ci_high == pytest.approx(float(hi), abs=2e-5)


def test_flat_prior_reproduces_the_frequentist_interval_at_scale(
    reference: by.BayesianResult,
) -> None:
    """With a uniform prior and n=60k the two frameworks must nearly coincide.

    They answer different questions, but at this sample size the arithmetic
    converges -- which is a useful cross-validation of both implementations.
    """
    freq = two_proportion_z_test(*REF)
    assert reference.lift_ci_low == pytest.approx(freq.ci_low, abs=2e-4)
    assert reference.lift_ci_high == pytest.approx(freq.ci_high, abs=2e-4)


def test_credible_interval_ordering_and_coverage_of_the_truth(
    reference: by.BayesianResult,
) -> None:
    from streamly.config import EXPERIMENT

    assert reference.lift_ci_low < reference.lift_median < reference.lift_ci_high
    assert reference.lift_ci_low <= EXPERIMENT.true_treatment_lift_abs <= reference.lift_ci_high


def test_a_wider_credible_level_gives_a_wider_interval() -> None:
    narrow = by.beta_binomial_test(*REF, credible_level=0.80)
    wide = by.beta_binomial_test(*REF, credible_level=0.99)
    assert wide.lift_ci_low < narrow.lift_ci_low
    assert wide.lift_ci_high > narrow.lift_ci_high


def test_quantile_rejects_out_of_range_input(reference: by.BayesianResult) -> None:
    with pytest.raises(ValueError, match="q must be in"):
        by.lift_quantile(reference.posterior_control, reference.posterior_treatment, 1.0)


# ---------------------------------------------------------------------------
# Determinism -- the reason for choosing quadrature over MCMC
# ---------------------------------------------------------------------------
def test_the_analysis_is_bit_for_bit_reproducible() -> None:
    a = by.beta_binomial_test(*REF, threshold=0.005)
    b = by.beta_binomial_test(*REF, threshold=0.005)
    assert a == b, "quadrature must give identical results across runs"


def test_decision_summary_reads_as_plain_language(reference: by.BayesianResult) -> None:
    text = by.decision_summary(reference, loss_tolerance=0.0005)
    assert "probability the treatment is better" in text
    assert "expected cost" in text
    assert "lower-risk choice" in text

    risky = by.beta_binomial_test(50, 1_000, 55, 1_000, threshold=0.02)
    assert "above tolerance" in by.decision_summary(risky, loss_tolerance=1e-6)


# ---------------------------------------------------------------------------
# Integration with the readout
# ---------------------------------------------------------------------------
def test_readout_reports_but_does_not_defer_to_the_posterior() -> None:
    """Two decision rules in parallel would invite picking whichever one wins."""
    import inspect

    from streamly.experiment import readout

    source = inspect.getsource(readout._decide)
    assert "bayes" not in source.lower(), (
        "_decide must not consult the posterior; the frequentist rule is the "
        "pre-registered one"
    )
    # And the parameter list must not even offer it.
    assert "bayesian" not in inspect.signature(readout._decide).parameters
