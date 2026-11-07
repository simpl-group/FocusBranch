"""Randomized checks of the node-constrained concretization (Eq. (10)/(12)) against a pure PyTorch reference: concrete
nodes (L0BoundedInput.selecting) and super nodes (L0BoundedInput.partial with a level), both computed by the l0bound
top-k kernels. Run with `python -m pytest tests` from the repository root."""
import pytest
import torch
from AbstractInterpreter import L0BoundedInput, RelationalPolyhedra

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def reference(W, b, x, hi, lo, focus, eps, level, upper, channels):
    """max (upper) / min (lower) of W x + b over every node-constrained l0-ball {(focus[i], level)}."""
    num_nodes = focus.shape[0]
    W = W.expand(num_nodes, -1, -1)
    sign = 1. if upper else -1.
    sw = sign * W
    d = torch.where(sw > 0, sw * hi, sw * lo).reshape(num_nodes, W.shape[1], channels, -1).sum(2)
    total = torch.zeros(num_nodes, W.shape[1], device=W.device)
    if level > 0:
        top_s, pos = d.gather(2, focus[:, None, :].expand(-1, d.shape[1], -1)).topk(level, dim=2)
        total += top_s.sum(2)
        d = d.scatter(2, focus[:, None, :].expand(-1, d.shape[1], -1).gather(2, pos), float('-inf'))
    if eps > level:
        total += d.topk(eps - level, dim=2).values.sum(2)
    return W @ x.reshape(-1, 1) + b + sign * total.unsqueeze(2)


def bounds(bounded: L0BoundedInput, rp: RelationalPolyhedra) -> tuple[torch.Tensor, torch.Tensor]:
    return bounded.compute_ubound(rp), bounded.compute_lbound(rp)


@pytest.mark.parametrize("channels,num_pixels", [(1, 784), (3, 1024), (1, 37)])
@pytest.mark.parametrize("eps", [1, 3, 5, 10])
def test_concrete_and_super_nodes(channels, num_pixels, eps):
    torch.manual_seed(eps * 7 + channels)
    features = channels * num_pixels
    x = torch.rand(features, device="cuda") * (torch.rand(features, device="cuda") > 0.4)
    hi, lo = 1 - x, -x
    num_nodes, rows = 50, 10
    W = torch.randn(1, rows, features, device="cuda")
    W *= torch.rand_like(W) > 0.5  # exact zeros and ties
    b = torch.randn(1, rows, 1, device="cuda")
    rp = RelationalPolyhedra(W, b)
    bounded = L0BoundedInput(x=x.reshape(1, -1, 1), eps=eps, channel_size=channels, num_pixels=num_pixels)
    bounded.update_x(x.reshape(1, -1, 1))
    for level in range(eps + 1):  # concrete nodes: the level is the size of the focus set
        focus = torch.rand(num_nodes, num_pixels, device="cuda").topk(level, dim=1).indices.sort(1).values
        with bounded.selecting(focus):
            upper, lower = bounds(bounded, rp)
        torch.testing.assert_close(upper, reference(W, b, x, hi, lo, focus, eps, level, True, channels), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(lower, reference(W, b, x, hi, lo, focus, eps, level, False, channels), rtol=1e-4, atol=1e-4)
    for level in range(min(3, eps), eps + 1):  # super nodes, as bounded by super branching (levels >= 3)
        for size in sorted({level, level + 1, 40} & set(range(eps, num_pixels + 1))):
            focus = torch.rand(num_nodes, num_pixels, device="cuda").topk(size, dim=1).indices.sort(1).values
            bounded.level = level
            with bounded.partial(focus):
                upper, lower = bounds(bounded, rp)
            bounded.level = None
            torch.testing.assert_close(upper, reference(W, b, x, hi, lo, focus, eps, level, True, channels), rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(lower, reference(W, b, x, hi, lo, focus, eps, level, False, channels), rtol=1e-4, atol=1e-4)
