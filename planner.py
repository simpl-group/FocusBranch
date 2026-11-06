"""
Strategy planning by node sampling (Section 3.4 and Appendix F.3 of the paper): from sampled safe rates and bounding
times, pick the super branching strategy R (refinement size for every focus-set size) and the covering verification design
(finite geometry) with the smallest estimated verification time at a level.
"""
# Reference: https://github.com/YuvShap/CoVerD/blob/main/tf_verify/calzone_robustness_analyzer.py
import math
import random
import numpy as np
from scipy.stats import norm
from coverd_finite_geometries import get_possible_geometries, Geometry


def aggregate_super_samples(per_worker: list[tuple[list[int], list[int], list[float]]], level: int) -> tuple[dict[int, float], dict[int, float]]:
    """Safe rate and mean bounding time of super nodes per focus-set size, from every worker's samples."""
    counts = np.sum([np.array(c) for c, _, _ in per_worker], axis=0)
    successes = np.sum([np.array(s) for _, s, _ in per_worker], axis=0)
    durations = np.sum([np.array(t) for _, _, t in per_worker], axis=0)
    sr = successes / counts
    mean_time = durations / counts
    mean_time[0] = mean_time[1]
    # The fraction-robust estimate needs sr[i - 1] > sr[i]: enforce a monotone decrease by interpolating offending entries.
    for i in range(1, len(sr) - 1):
        if sr[i] >= sr[i - 1]:
            sr[i] = min((sr[i - 1] + sr[i + 1]) / 2, sr[i - 1])
    return {k + level: v for k, v in enumerate(sr)}, {k + level: v for k, v in enumerate(mean_time)}


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
            expected = (cdf_after - cdf_before) * total_blocks  # expected number of blocks of this size
            cdf_before = cdf_after
            count = math.floor(expected)
            if random.random() < expected - count:
                count += 1
            estimated_time += count * VT[block_size]
        candidates.append({str(geometry): f"{total_blocks} blocks, mean size {round(mean_size, 3)}",
                           "estimated_summed_duration": round(float(estimated_time), 3)})
        if estimated_time < best_time:
            best_time, best_geometry = estimated_time, geometry
    return best_geometry, sorted(candidates, key=lambda x: x["estimated_summed_duration"])[:3]
