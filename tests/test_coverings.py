"""The covering verification designs and the refinement coverings cover every level-subset. Run with
`python -m pytest tests` from the repository root."""
import random
from itertools import combinations
import pytest
from covering_generator import CoveringConstructor
from coverd_finite_geometries import Geometry, get_possible_geometries
from data import BASE_DIR, load_covering, load_covering_sizes


def covers(blocks: list[list[int]], size: int, level: int) -> bool:
    sets = [set(block) for block in blocks]
    return all(any(set(t) <= s for s in sets) for t in combinations(range(size), level))


@pytest.mark.parametrize("q,m,t", [(2, 4, 3), (3, 3, 3), (4, 3, 3), (3, 4, 2), (2, 5, 4)])
def test_finite_geometry_covering(q, m, t):
    geometry = Geometry(q, m, t)
    random.seed(q * 100 + m * 10 + t)
    num_pixels = min(geometry.number_of_points(), 14)
    chosen = random.sample(range(geometry.number_of_points()), num_pixels)
    blocks = [block.tolist() for block in CoveringConstructor(q, m, t, chosen).construct_induced_covering()]
    assert len(blocks) == geometry.number_of_blocks()
    assert all(0 <= i < num_pixels for block in blocks for i in block)
    assert covers(blocks, num_pixels, t)


def test_possible_geometries_hold_every_pixel():
    for t in (3, 4, 5, 6):
        geometries = list(get_possible_geometries(784, t))
        assert geometries and all(g.t == t and g.number_of_points() >= 784 for g in geometries)


@pytest.mark.skipif(not (BASE_DIR / "refinement_coverings" / "3").is_dir(), reason="refinement coverings not downloaded (see README)")
@pytest.mark.parametrize("t", [3, 4])
def test_refinement_coverings(t):
    sizes = load_covering_sizes(t)
    for v, k in [(v, k) for (v, k) in sorted(sizes) if v <= t + 9][:12]:
        covering = load_covering(v, k, t)
        assert len(covering) == sizes[(v, k)] and all(len(block) == k for block in covering)
        assert covers(covering, v, t)
