"""Checks of the fitted planner inputs. Run with `python -m pytest tests` from the repository root."""
import numpy as np
from scipy.special import expit
import planner


def sample(level=5, max_size=200, a=60., b=-14., full=400, reduced=20, seed=0):
    """Binomial samples like GpuWorker.super_sample_all_k: fewer samples after the first size without a safe node."""
    rng = np.random.default_rng(seed)
    sizes = np.arange(level, max_size + 1, dtype=float)
    p = expit(a + b * np.log(sizes))
    counts, successes, n = [], [], full
    for q in p:
        counts.append(n)
        successes.append(rng.binomial(n, q))
        if successes[-1] == 0: n = reduced
    return sizes, np.array(counts, float), np.array(successes, float), p


def test_safe_rate_fit_recovers_curve_and_never_saturates():
    sizes, counts, successes, p = sample()
    fitted = planner.fit_safe_rate(sizes, counts, successes)
    assert np.all(np.diff(fitted) <= 0)
    inside = (fitted > 1e-12) & (fitted < 1 - 1e-12)  # elsewhere the curve rounds to 0 or 1 in float64
    assert np.all(np.diff(fitted)[inside[1:] & inside[:-1]] < 0), "the fit must be strictly decreasing"
    assert np.abs(fitted - p).max() < 0.05
    saturated = np.nonzero(successes == counts)[0]  # raw rate 1.0, i.e. no failure observed
    assert len(saturated) and 1 - fitted[saturated[-1]] > 1e-4, "the largest all-safe size still gets a failure rate"


def test_safe_rate_fit_degenerate_samples():
    sizes = np.arange(3, 50, dtype=float)
    counts = np.full(len(sizes), 400.)
    assert np.all(planner.fit_safe_rate(sizes, counts, counts) == 1)
    assert np.all(planner.fit_safe_rate(sizes, counts, np.zeros(len(sizes))) == 0)
    separated = np.where(sizes < 20, counts, 0.)  # perfectly separated: the slope is bounded, the fit stays finite
    fitted = planner.fit_safe_rate(sizes, counts, separated)
    assert np.all(np.isfinite(fitted)) and fitted[0] > 0.99 and fitted[-1] < 0.01


def test_node_time_fit_ignores_warmup_and_overhead_dominated_sizes():
    sizes, counts, successes, _ = sample()
    per_node = 30e-6 + 0.02e-6 * sizes
    durations = per_node * counts
    durations[counts < counts.max()] = 3e-3  # a few nodes per batch: the per-batch overhead dominates
    durations[0] *= 5  # warm-up
    fitted = planner.fit_node_time(sizes, counts, durations)
    np.testing.assert_allclose(fitted, per_node, rtol=1e-6)


def test_refinement_strategy_is_stable_under_timing_noise():
    sizes, counts, successes, _ = sample(seed=1)
    covering_sizes = {(v, k): int(np.ceil((v / k) ** 2)) for v in range(5, 201) for k in range(5, v)}
    chains = set()
    for seed in range(5):
        noise = np.random.default_rng(seed).normal(1, 0.02, len(sizes))
        durations = (30e-6 + 0.02e-6 * sizes) * counts * noise
        sr, mt = planner.aggregate_super_samples([(counts, successes, durations)], 5)
        _, strategy = planner.compute_super_branching_strategy(planner.fraction_robust(sr, 5, 200), mt, sr, covering_sizes, 5, 200)
        v, chain = 72, [72]
        while strategy.get(v) is not None:
            v = strategy[v]
            chain.append(v)
        chains.add(tuple(chain))
    assert len(chains) == 1, chains
