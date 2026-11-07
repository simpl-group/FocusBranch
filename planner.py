"""
Verification planning by node sampling (Section 3.5 and Appendix F.2 of the paper): from sampled safe rates and bounding
times, pick the super branching strategy R (refinement size for every focus-set size) and the covering verification design
(finite geometry) with the smallest estimated verification time at a level.
"""
# Reference: https://github.com/YuvShap/CoVerD/blob/main/tf_verify/calzone_robustness_analyzer.py
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import norm, theilslopes
from coverd_finite_geometries import get_possible_geometries, Geometry


def fit_safe_rate(sizes: np.ndarray, counts: np.ndarray, successes: np.ndarray) -> np.ndarray:
    """
    P(a size-k super node is safe), fitted as a logistic curve in log k, sigma(a + b log k) with b <= 0, by maximum
    likelihood over the binomial samples of every size.

    The raw per-size rates are 1.0 wherever no sampled node failed (e.g. 400/400), and the refinement planning then
    assumes such nodes never fail, so the refinement sizes below them are decided by timing noise. The fit pools the
    samples of all sizes: it is smooth, strictly decreasing and gives small but nonzero failure rates there.
    """
    if successes.sum() == counts.sum(): return np.ones(len(sizes))
    if successes.sum() == 0: return np.zeros(len(sizes))
    x = np.log(sizes)
    x = x - x.mean()

    def nll(theta: np.ndarray) -> tuple[float, np.ndarray]:
        z = theta[0] + theta[1] * x
        p = expit(z)
        loss = -(successes * log_expit(z) + (counts - successes) * log_expit(-z)).sum()
        r = successes - counts * p
        return float(loss), -np.array([r.sum(), (r * x).sum()])

    # bounded slope: perfectly separated samples would push it to -inf
    theta = minimize(nll, np.zeros(2), jac=True, method='L-BFGS-B', bounds=[(None, None), (-1e3, 0.)]).x
    return expit(theta[0] + theta[1] * x)


def fit_node_time(sizes: np.ndarray, counts: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """
    Bounding time of one size-k super node, fitted as a + b k (b >= 0, Theil-Sen) on the sizes that were sampled with
    the full sample count. Sizes sampled after a size without any safe node get far fewer samples, and their time per
    node is dominated by the per-batch overhead; the first size also pays the warm-up.
    """
    full = np.nonzero(counts == counts.max())[0][1:]  # skip the first size (warm-up)
    if len(full) < 2:
        return np.full(len(sizes), float(np.median(durations / counts)))
    per_node = durations[full] / counts[full]
    slope = max(float(theilslopes(per_node, sizes[full])[0]), 0.)
    intercept = float(np.median(per_node - slope * sizes[full]))
    return np.maximum(intercept + slope * sizes, 0.5 * float(per_node.min()))


def aggregate_super_samples(per_worker: list[tuple[list[int], list[int], list[float]]], level: int) -> tuple[dict[int, float], dict[int, float]]:
    """Safe rate and bounding time of a super node per focus-set size, fitted from every worker's samples."""
    counts = np.sum([np.array(c) for c, _, _ in per_worker], axis=0).astype(float)
    successes = np.sum([np.array(s) for _, s, _ in per_worker], axis=0).astype(float)
    durations = np.sum([np.array(t) for _, _, t in per_worker], axis=0).astype(float)
    sizes = np.arange(level, level + len(counts), dtype=float)
    sr = fit_safe_rate(sizes, counts, successes)
    mean_time = fit_node_time(sizes, counts, durations)
    return {k + level: float(v) for k, v in enumerate(sr)}, {k + level: float(v) for k, v in enumerate(mean_time)}


def fraction_robust(safe_rate: dict[int, float], level: int, max_size: int) -> dict[int, dict[int, float]]:
    """Estimated fraction of the size-k children of an unknown size-v node that are safe."""
    fraction: dict[int, dict[int, float]] = {}
    for v in range(level, max_size + 1):
        fraction[v] = {}
        for k in range(level, v):
            if 1 - safe_rate[v] < 1e-7:
                fraction[v][k] = 1.0
            else:
                fraction[v][k] = min(1, max((safe_rate[k] - safe_rate[v]) / (1 - safe_rate[v]), 0))
    return fraction


def compute_super_branching_strategy(
    fraction: dict[int, dict[int, float]],
    mean_time: dict[int, float],
    safe_rate: dict[int, float],
    covering_sizes: dict[tuple[int, int], int],
    level: int,
    max_size: int,
) -> tuple[dict[int, float], dict[int, int | None]]:
    """Dynamic programming over focus-set sizes: the best refinement size k for every size v and the expected
    verification time VT[v] of a size-v super node."""
    RT: dict[int, float] = {level: 0.}  # expected refinement time
    VT: dict[int, float] = {level: mean_time[level]}  # expected verification time
    strategy: dict[int, int | None] = {level: None}  # best next k or None
    for v in range(level + 1, max_size + 1):
        best_k, best_time = None, float('inf')
        for k in range(level, v):
            if (v, k) not in covering_sizes:
                continue
            refine_time = covering_sizes[(v, k)] * (mean_time[k] + (1 - fraction[v][k]) * RT[k])
            if refine_time < best_time:
                best_k, best_time = k, refine_time
        RT[v], strategy[v] = best_time, best_k
        VT[v] = mean_time[v] + best_time * (1 - safe_rate[v])
    return VT, strategy


def choose_geometry(VT: dict[int, float], num_pixels: int, level: int, max_size: int, mean_size_limit: int) -> tuple[Geometry | None, list[dict[str, object]]]:
    """The covering verification design with the smallest estimated time (None if no design fits), and the top 3
    candidates for logging."""
    best_geometry, best_time = None, float('inf')
    candidates = []
    for geometry in get_possible_geometries(num_pixels, level):
        mean_size = geometry.block_size() * num_pixels / geometry.number_of_points()
        std_dev = math.sqrt(mean_size * (1 - mean_size + (geometry.block_size() - 1) * (num_pixels - 1) / (geometry.number_of_points() - 1)))
        total_blocks = geometry.number_of_blocks()
        if (1 - norm.cdf(max_size, mean_size, std_dev)) * total_blocks > 1e-2 or mean_size > mean_size_limit:
            continue
        estimated_time = 0
        cdf_before = norm.cdf(level - 0.5, mean_size, std_dev)
        for block_size in range(level, max_size + 1):
            cdf_after = norm.cdf(block_size + 0.5, mean_size, std_dev)
            estimated_time += (cdf_after - cdf_before) * total_blocks * VT[block_size]  # expected number of blocks of this size
            cdf_before = cdf_after
        candidates.append({str(geometry): f"{total_blocks} blocks, mean size {round(mean_size, 3)}",
                           "estimated_summed_duration": round(float(estimated_time), 3)})
        if estimated_time < best_time:
            best_time, best_geometry = estimated_time, geometry
    return best_geometry, sorted(candidates, key=lambda x: x["estimated_summed_duration"])[:3]
