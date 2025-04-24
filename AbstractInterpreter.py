import torch
import torch.nn as nn
from typing import Self
import warnings
from contextlib import contextmanager

torch.set_default_dtype(torch.float32)

class RelationalPolyhedra:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        self.W = weight  # (batch_size, output_dim, input_dim)
        self.b = bias  # (batch_size, output_dim, 1)

    @classmethod
    def clone(cls, other: Self) -> Self:
        return cls(other.W.clone(), other.b.clone())

    def clone(self) -> Self:
        return RelationalPolyhedra(self.W.clone(), self.b.clone())

    def posW(self) -> torch.Tensor:
        return self.W.clamp_min(0)

    def negW(self) -> torch.Tensor:
        return self.W.clamp_max(0)

class Bounds:
    def __init__(self, lbounds: torch.Tensor, ubounds: torch.Tensor):
        self.l = lbounds  # (batch_size, features, 1)
        self.u = ubounds

    @classmethod
    def clone(cls, other: Self) -> Self:
        return cls(other.l.clone(), other.u.clone())

    def __add__(self, other: torch.Tensor | Self) -> Self:
        if isinstance(other, Bounds):
            return Bounds(self.l + other.l, self.u + other.u)
        else:
            return Bounds(self.l + other, self.u + other)

    def __radd__(self, other: torch.Tensor) -> Self:
        return self.__add__(other)

    def __sub__(self, other: torch.Tensor | Self) -> Self:
        if isinstance(other, Bounds):
            return Bounds(self.l - other.l, self.u - other.u)
        else:
            return Bounds(self.l - other, self.u - other)

    def __rsub__(self, other: torch.Tensor) -> Self:
        return Bounds(other - self.l, other - self.u)

class BoundedInput:
    def __init__(self, x: torch.Tensor, eps, pixel_value_range=(0, 1)):
        self.x = x
        self.device = x.device
        self.eps = eps
        self.L, self.U = pixel_value_range
        self.batch_size, self.num_features, _ = x.shape
        self.batch_indices = torch.arange(self.batch_size).unsqueeze(1)

    def compute_ubound(self, rp: RelationalPolyhedra) -> torch.Tensor:
        raise NotImplementedError

    def compute_lbound(self, rp: RelationalPolyhedra) -> torch.Tensor:
        raise NotImplementedError

    def compute_bounds(self, lrp: RelationalPolyhedra, urp: RelationalPolyhedra) -> Bounds:
        return Bounds(lbounds=self.compute_lbound(lrp), ubounds=self.compute_ubound(urp))

class L0BoundedInput(BoundedInput):
    def __init__(self, x: torch.Tensor, eps: int, pixel_value_range=(0, 1)):
        super().__init__(x, eps, pixel_value_range)
        self.selected_idx: torch.Tensor = torch.empty((self.batch_size, 0), dtype=torch.long).to(self.device)
        """indices pre-selected before the ranking to compute the sum which is then added to the diff"""

    def num_selected(self):
        return self.selected_idx[0].numel()

    def expand(self, batch_size: int) -> None:
        self.batch_indices = torch.arange(batch_size).unsqueeze(1)
        self.batch_size = batch_size
        assert self.num_selected() == 0, "Cannot expand L0BoundedInput with selected indices"
        self.selected_idx = torch.empty((self.batch_size, 0), dtype=torch.long).to(self.device)

    def clear_cache(self) -> None:
        del self.x
        self.batch_indices = torch.arange(1).unsqueeze(1)
        self.batch_size = 1
        assert self.num_selected() == 0, "Cannot clear cache of L0BoundedInput with selected indices"
        self.selected_idx = torch.empty((self.batch_size, 0), dtype=torch.long).to(self.device)

    def select_idx(self, idx: torch.Tensor):
        self.selected_idx = torch.cat((self.selected_idx, idx.to(self.device)), dim=1)
        self.eps -= idx[0].numel()

    def undo_select(self, last: int=1):
        self.selected_idx = self.selected_idx[:, :-last]
        self.eps += last

    @contextmanager
    def selecting(self, idx: torch.Tensor):
        if idx.ndim == 1:
            idx = idx.unsqueeze(0).expand(self.batch_size, -1)
        self.select_idx(idx)
        try:
            yield
        finally:
            self.undo_select(idx[0].numel())

    def wx_selected_sum(self, WX_diff: torch.Tensor) -> torch.Tensor:
        indices = self.selected_idx.unsqueeze(1).expand(-1, WX_diff.shape[1], -1)
        return WX_diff.gather(dim=2, index=indices).sum(dim=-1, keepdim=True)

    def compute_udiff(self, rpW: torch.Tensor) -> torch.Tensor:
        X = self.x.transpose(-1,-2) # rpW.shape[1]: number of nuerons in next layer
        WX_diff = torch.where(rpW > 0, rpW * (self.U - X), rpW * (self.L - X))
        if WX_diff.shape[0] != self.batch_size:
            WX_diff = WX_diff.expand(self.batch_size, -1, -1).clone()

        WX_selected_sum = self.wx_selected_sum(WX_diff)
        if self.eps == 0: return WX_selected_sum
        WX_diff[self.batch_indices, :, self.selected_idx] = 0

        WX_tope_diff = WX_diff.topk(self.eps, dim=2, largest=True, sorted=False).values
        return WX_tope_diff.sum(dim=-1, keepdim=True) + WX_selected_sum

    def compute_ldiff(self, rpW: torch.Tensor) -> torch.Tensor:
        X = self.x.transpose(-1,-2)
        WX_diff = torch.where(rpW < 0, rpW * (self.U - X), rpW * (self.L - X))
        if WX_diff.shape[0] != self.batch_size:
            WX_diff = WX_diff.expand(self.batch_size, -1, -1).clone()

        WX_selected_sum = self.wx_selected_sum(WX_diff)
        if self.eps == 0: return WX_selected_sum
        WX_diff[self.batch_indices, :, self.selected_idx] = 0

        WX_tope_diff = WX_diff.topk(self.eps, dim=2, largest=False, sorted=False).values
        return WX_tope_diff.sum(dim=-1, keepdim=True) + WX_selected_sum

    def compute_ubound(self, rp: RelationalPolyhedra) -> torch.Tensor:
        center = rp.W @ self.x + rp.b
        diff = self.compute_udiff(rp.W)
        result = center + diff
        if result.shape[0] != self.batch_size:
            result = result.expand(self.batch_size, -1, -1)
        return result

    def compute_lbound(self, rp: RelationalPolyhedra) -> torch.Tensor:
        center = rp.W @ self.x + rp.b
        diff = self.compute_ldiff(rp.W)
        result = center + diff
        if result.shape[0] != self.batch_size:
            result = result.expand(self.batch_size, -1, -1)
        return result

class LinfBoundedInput(BoundedInput):
    def __init__(self, x: torch.Tensor, eps: float, pixel_value_range=(0, 1), bounds: Bounds=None):
        super().__init__(x, eps, pixel_value_range)
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = Bounds((x-eps).clamp(self.L, self.U), (x+eps).clamp(self.L, self.U))

    def compute_ubound(self, rp: RelationalPolyhedra) -> torch.Tensor:
        return rp.posW() @ self.bounds.u + rp.negW() @ self.bounds.l + rp.b

    def compute_lbound(self, rp: RelationalPolyhedra) -> torch.Tensor:
        return rp.posW() @ self.bounds.l + rp.negW() @ self.bounds.u + rp.b

class AbstractTransformer:
    def __init__(self):
        self.parent: Self | None = None

    def backsubstitution(self) -> tuple[RelationalPolyhedra, RelationalPolyhedra]:
        assert isinstance(self, AffineAT)
        lrp, urp = self.rp.clone(), self.rp.clone()
        prev = self.parent

        while prev is not None:
            if isinstance(prev, ActivationAT):
                lrp.b = lrp.posW() @ prev.lrp.b + lrp.negW() @ prev.urp.b + lrp.b
                urp.b = urp.posW() @ prev.urp.b + urp.negW() @ prev.lrp.b + urp.b
                lrp.W = lrp.posW() * prev.lrp.W + lrp.negW() * prev.urp.W
                urp.W = urp.posW() * prev.urp.W + urp.negW() * prev.lrp.W
            elif isinstance(prev, AffineAT):
                lrp.b = lrp.W @ prev.rp.b + lrp.b
                urp.b = urp.W @ prev.rp.b + urp.b
                lrp.W = lrp.W @ prev.rp.W
                urp.W = urp.W @ prev.rp.W
            else: raise NotImplementedError(f"Transform for {prev} is unclear")
            prev = prev.parent

        return lrp, urp

    def backsubstitution_lrp(self) -> RelationalPolyhedra:
        assert isinstance(self, AffineAT)
        lrp = self.rp.clone()
        prev = self.parent

        while prev is not None:
            if isinstance(prev, ActivationAT):
                lrp.b = lrp.posW() @ prev.lrp.b + lrp.negW() @ prev.urp.b + lrp.b
                lrp.W = lrp.posW() * prev.lrp.W + lrp.negW() * prev.urp.W
            elif isinstance(prev, AffineAT):
                lrp.b = lrp.W @ prev.rp.b + lrp.b
                lrp.W = lrp.W @ prev.rp.W
            else: raise NotImplementedError(f"Transform for {prev} is unclear")
            prev = prev.parent

        return lrp

class AffineAT(AbstractTransformer):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.W = weight
        self.b = bias
        self.rp = RelationalPolyhedra(self.W, self.b)

    @classmethod
    def from_linear(cls, layer: nn.Linear) -> Self:
        return cls(layer.weight.detach().unsqueeze(0), layer.bias.detach().reshape(1, -1, 1))

    @classmethod
    def from_conv2d(cls, layer: nn.Conv2d, input_shape) -> Self:
        kernel, bias = layer.weight.detach(), layer.bias.detach()
        stride, padding = layer.stride, layer.padding
        _, ci, hi, wi = input_shape
        kh, kw = kernel.shape[2], kernel.shape[3]
        co = kernel.shape[0]
        ho = (hi + 2 * padding[0] - kh) // stride[0] + 1
        wo = (wi + 2 * padding[1] - kw) // stride[1] + 1

        W = torch.zeros(co * ho * wo, ci * hi * wi, device=bias.device)
        b = torch.zeros(co * ho * wo, device=bias.device)
        for co_idx in range(co):
            for h_out in range(ho):
                for w_out in range(wo):
                    row = co_idx * ho * wo + h_out * wo + w_out
                    b[row] = bias[co_idx]
                    for ci_idx in range(ci):
                        for kh_idx in range(kh):
                            for kw_idx in range(kw):
                                h_in = h_out * stride[0] - padding[0] + kh_idx
                                w_in = w_out * stride[1] - padding[1] + kw_idx
                                if 0 <= h_in < hi and 0 <= w_in < wi:
                                    input_idx = ci_idx * hi * wi + h_in * wi + w_in
                                    W[row, input_idx] = kernel[co_idx, ci_idx, kh_idx, kw_idx]
        return cls(weight=W.unsqueeze(0), bias=b.reshape(1, -1, 1))

    def make_marginal(self, true_labels: torch.Tensor, target_label: int, device: str="cpu") -> None:
        if true_labels.shape[0] != 1:
            self.W = self.W.expand(true_labels.shape[0], -1, -1)
            self.b = self.b.expand(true_labels.shape[0], -1, -1)
        batch_idx = torch.arange(true_labels.shape[0])
        W = (self.W[batch_idx, true_labels, :] - self.W[:, target_label, :]).unsqueeze(1)
        b = (self.b[batch_idx, true_labels, :] - self.b[:, target_label, :]).unsqueeze(1)
        self.rp = RelationalPolyhedra(W.to(device), b.to(device))

    def make_marginals(self, true_labels: torch.Tensor, device: str="cpu") -> None:
        if true_labels.shape[0] != 1:
            self.W = self.W.expand(true_labels.shape[0], -1, -1)
            self.b = self.b.expand(true_labels.shape[0], -1, -1)
        batch_idx = torch.arange(true_labels.shape[0])
        # each row of W should be the difference between the true label and others
        W = self.W[batch_idx, true_labels, :].unsqueeze(1) - self.W
        b = self.b[batch_idx, true_labels, :].unsqueeze(1) - self.b
        self.rp = RelationalPolyhedra(W.to(device), b.to(device))

class ActivationAT(AbstractTransformer):
    def __init__(self):
        super().__init__()
        self.lrp: RelationalPolyhedra = None
        self.urp: RelationalPolyhedra = None

    @staticmethod
    def compute_rp(bounds: Bounds) -> tuple[RelationalPolyhedra, RelationalPolyhedra]:
        raise NotImplementedError("Subclasses must implement compute_rp method.")

class ReLUAT(ActivationAT):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_rp(bounds: Bounds) -> tuple[RelationalPolyhedra, RelationalPolyhedra]:
        # Initialization
        lrp = RelationalPolyhedra(torch.ones_like(bounds.l), torch.zeros_like(bounds.l))
        urp = lrp.clone()

        # Case 1: ubound <= 0
        below_nodes = bounds.u <= 0
        urp.W = torch.where(below_nodes, 0, urp.W)

        # Case 2: lbound >= 0
        above_nodes = bounds.l >= 0
        urp.W = torch.where(above_nodes, 1, urp.W)

        # Case 3: lbound <= 0 <= ubound
        crossing_nodes = ~(below_nodes | above_nodes)
        uk = bounds.u / (bounds.u - bounds.l)
        ub = (1 - uk) * bounds.u
        urp.W = torch.where(crossing_nodes, uk, urp.W)
        urp.b = torch.where(crossing_nodes, ub, urp.b)

        # Case a: lbound + ubound > 0
        lrp.W = torch.where(bounds.l + bounds.u > 0, 1, lrp.W)
        # Case b: lbound + ubound <= 0
        lrp.W = torch.where(bounds.l + bounds.u <= 0, 0, lrp.W)

        lrp.W.transpose_(-1, -2)
        urp.W.transpose_(-1, -2)
        return lrp, urp

class SequentialAT:
    def __init__(self, ATs: list[AbstractTransformer]):
        tail = ATs[-1]
        assert isinstance(tail, AffineAT)
        self.ATs = ATs
        self.tail = tail
        self.tail.W = self.tail.W.cpu()
        self.tail.b = self.tail.b.cpu()

    @staticmethod
    def parse_layers(seq_net: nn.Sequential, x: torch.Tensor, prev: AbstractTransformer | None=None) -> list[AbstractTransformer]:
        ATs = []
        current: AbstractTransformer | None = None
        for layer in seq_net:
            x_in = x
            x = layer(x_in)
            if isinstance(layer, nn.Linear):
                current = AffineAT.from_linear(layer)
            elif isinstance(layer, nn.Conv2d):
                current = AffineAT.from_conv2d(layer, x_in.shape)
            elif isinstance(layer, nn.ReLU):
                current = ReLUAT()
            elif isinstance(layer, nn.Sequential):
                ATs += SequentialAT.parse_layers(layer, x_in, prev)
                prev = ATs[-1]
            elif isinstance(layer, nn.Flatten):
                if layer.start_dim != 1 or layer.end_dim != -1:
                    warnings.warn("No Abstract Transformer for Flatten with non-default start_dim and end_dim; skip for now", RuntimeWarning)
            elif not isinstance(layer, (nn.Identity, nn.Dropout)):
                warnings.warn(f"No Abstract Transformer for {layer}; skip for now", RuntimeWarning)
            if current is not None:
                # feed forward computation graph
                current.parent = prev
                ATs.append(current)
                prev = current
                current = None
        return ATs

    @classmethod
    def from_sequential(cls, seq_net: nn.Sequential, x: torch.Tensor, prev: AbstractTransformer | None=None) -> Self:
        return cls(SequentialAT.parse_layers(seq_net, x, prev))

    def update_activation_rp(self, bounded_in: BoundedInput) -> None:
        for AT in self.ATs:
            if isinstance(AT, ActivationAT):
                lrp_wrt_inputs, urp_wrt_inputs = AT.parent.backsubstitution()
                pre_activation_bounds = bounded_in.compute_bounds(lrp_wrt_inputs, urp_wrt_inputs)
                AT.lrp, AT.urp = AT.compute_rp(pre_activation_bounds)
