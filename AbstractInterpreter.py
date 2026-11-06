import torch
import torch.nn as nn
from typing_extensions import Self
import warnings
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import NamedTuple
from l0bound import selected_top_k_sum, selected_bottom_k_sum, top_top_sum, bottom_bottom_sum

torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True

class RelationalPolyhedra:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        self.W = weight  # (batch_size, output_dim, input_dim)
        self.b = bias  # (batch_size, output_dim, 1)

    def clone(self) -> Self:
        return RelationalPolyhedra(self.W.clone(), self.b.clone())

    def posW(self) -> torch.Tensor:
        return self.W.clamp_min(0)

    def negW(self) -> torch.Tensor:
        return self.W.clamp_max(0)

    def to_(self, device: str | torch.device) -> None:
        self.W = self.W.to(device)
        self.b = self.b.to(device)

class Bounds(NamedTuple):
    l: torch.Tensor
    u: torch.Tensor

class BoundedInput(ABC):
    def __init__(self, x: torch.Tensor, eps, pixel_value_range: tuple[float, float]=(0., 1.)):
        self.x = x
        self.device = x.device
        self.eps = eps
        self.L, self.U = pixel_value_range
        self.batch_size, self.num_features, _ = x.shape
        self.batch_indices = torch.arange(self.batch_size).unsqueeze(1)

    @abstractmethod
    def compute_ubound(self, rp: RelationalPolyhedra) -> torch.Tensor:
        """Upper bound of the affine function `rp` over the input region."""

    @abstractmethod
    def compute_lbound(self, rp: RelationalPolyhedra) -> torch.Tensor:
        """Lower bound of the affine function `rp` over the input region."""

    def compute_bounds(self, lrp: RelationalPolyhedra, urp: RelationalPolyhedra) -> Bounds:
        return Bounds(self.compute_lbound(lrp), self.compute_ubound(urp))

class L0BoundedInput(BoundedInput):
    def __init__(self, x: torch.Tensor, eps: int, pixel_value_range: tuple[float, float]=(0., 1.), channel_size: int=1, num_pixels: int=784):
        super().__init__(x, eps, pixel_value_range)
        self.selected_idx: torch.Tensor = torch.empty((self.batch_size, 0), dtype=torch.long).to(self.device)
        """indices pre-selected before the ranking to compute the sum which is then added to the diff"""
        self.channel_size = channel_size
        self.channel_offsets = (torch.arange(channel_size, device=self.device) * num_pixels).reshape(1, -1, 1)  # shape (1, channel_size, 1)
        self.num_pixels = int(self.num_features / channel_size)
        assert self.num_pixels == num_pixels, f"num_pixels should be {num_pixels}, but got {self.num_pixels}"
        self.partial_indices: torch.Tensor | None = None
        """indices of the features to be considered in the computation of the diff, if None, all pixels are considered"""
        self.partial_pixel_indices: torch.Tensor | None = None
        """pixel-level (not channel-expanded) batched partial indices, used by the node-constrained concretization"""
        self.level: int|None = None

    def update_x(self, x: torch.Tensor) -> None:
        assert x.ndim == 3 and x.shape[1] == self.num_features and x.shape[2] == 1, \
            f"x should have shape (batch_size, {self.num_features}, 1), but got {x.shape}"
        if x.device != self.device:
            x = x.to(self.device)
        self.x = x
        X = self.x.transpose(1, 2)  # shape (self.batch_size, 1, num_features)
        self.__U_X = self.U_X = (self.U - X).clamp_min(0)
        self.__L_X = self.L_X = (self.L - X).clamp_max(0)
        self.__update_batch_size(x.shape[0])

    def __update_batch_size(self, new_batch_size: int) -> None:
        if new_batch_size == self.batch_size: return
        self.batch_indices = torch.arange(new_batch_size).unsqueeze(1)
        self.batch_size = new_batch_size
        assert self.num_selected() == 0 and (self.partial_indices is None or self.partial_indices.ndim == 1), \
            "Cannot update batch size with selected indices or batched partial indices"
        self.selected_idx = torch.empty((self.batch_size, 0), dtype=torch.long).to(self.device)

    @contextmanager
    def partial(self, partial_indices: torch.Tensor | None):
        if isinstance(partial_indices, torch.Tensor):
            if self.level is None:
                assert self.eps + self.num_selected() <= partial_indices.shape[-1] <= self.x.shape[1], \
                f"the number of partial indices should be at least {self.eps + self.num_selected()} "\
                f"and at most {self.x.shape[1]}, but got {partial_indices.shape[-1]}."
            else:
                assert self.level <= partial_indices.shape[-1] <= self.x.shape[1], \
                f"the number of partial indices should be at least {self.level} "\
                f"and at most {self.x.shape[1]}, but got {partial_indices.shape[-1]}."
            if partial_indices.device != self.device:
                partial_indices = partial_indices.to(self.device)
            if partial_indices.ndim == 1:
                if self.channel_size > 1:
                    partial_indices = (partial_indices.unsqueeze(0) + self.channel_offsets.squeeze(0)).reshape(-1)
                self.U_X = self.__U_X[:, :, partial_indices]
                self.L_X = self.__L_X[:, :, partial_indices]
            elif partial_indices.ndim == 2:  # (batch_size, num_partial_pixels): batched partial
                partial_batch_size = partial_indices.shape[0]
                if self.batch_size != partial_batch_size:
                    self.__update_batch_size(partial_batch_size)
                self.partial_pixel_indices = partial_indices
                partial_indices = partial_indices.unsqueeze(1)  # (batch_size, 1, num_partial_pixels)
                if self.channel_size > 1:
                    partial_indices = (partial_indices + self.channel_offsets).reshape(partial_batch_size, 1, -1)
                self.U_X = self.__U_X.expand(self.batch_size, 1, -1).gather(dim=2, index=partial_indices)
                self.L_X = self.__L_X.expand(self.batch_size, 1, -1).gather(dim=2, index=partial_indices)
            else: assert False  # should not happen, partial_indices should be 1D or 2D tensor
            self.partial_indices = partial_indices  # (batch_size, 1, num_partial_pixels)
        try:
            yield
        finally:
            self.partial_indices = None
            self.partial_pixel_indices = None
            self.U_X = self.__U_X
            self.L_X = self.__L_X

    def num_selected(self) -> int:
        return self.selected_idx[0].numel()

    def clear_cache(self) -> None:
        del self.x, self.U_X, self.L_X, self.__L_X, self.__U_X
        self.__update_batch_size(1)

    @contextmanager
    def selecting(self, idx: torch.Tensor):
        if idx.ndim == 1:
            idx = idx.expand(self.batch_size, -1)
        idx_batch_size = idx.shape[0]
        if self.batch_size != idx_batch_size:
            self.__update_batch_size(idx_batch_size)
        self.selected_idx = torch.cat((self.selected_idx, idx), dim=1)
        increment = idx[0].numel()
        self.eps -= increment
        try:
            yield
        finally:
            self.selected_idx = self.selected_idx[:, :-increment]
            self.eps += increment

    def wx_selected_sum(self, WX_diff: torch.Tensor) -> torch.Tensor:
        indices = self.selected_idx.unsqueeze(1).expand(-1, WX_diff.shape[1], -1)
        return WX_diff.gather(dim=2, index=indices).sum(dim=2, keepdim=True)

    def compute_udiff(self, rpW: torch.Tensor) -> torch.Tensor:
        if self.level is not None and self.level < self.eps:
            assert self.partial_indices is not None and self.partial_indices.ndim == 3
            assert self.num_selected() == 0
            # two ways:
            # 1. partial rpW and all rpW -> partial WX_diff and all WX_diff -> top-level for partial WX_diff and top-(eps-level) for all WX_diff. But this is infeasible since we need to mute those top-level indices in all WX_diff
            # 2. only have all rpw, but pass in partial indices and level to compute_udiff
            WX_diff = torch.where(rpW > 0, rpW * self.__U_X, rpW * self.__L_X)
            if self.channel_size > 1:
                WX_diff = WX_diff.reshape(WX_diff.shape[0], WX_diff.shape[1], self.channel_size, -1).sum(dim=2)
            if WX_diff.shape[0] != self.batch_size:
                WX_diff = WX_diff.repeat(self.batch_size, 1, 1)
            assert WX_diff.is_cuda
            return top_top_sum(WX_diff, self.partial_pixel_indices, self.eps, self.level)  # WX_diff is per pixel, so index by pixels
        if self.partial_indices is not None:
            if self.partial_indices.ndim == 1:
                rpW = rpW[:, :, self.partial_indices]
            else:  # batched partial indices
                rpW = rpW.expand(self.batch_size, -1, -1).gather(dim=2, index=self.partial_indices.expand(-1, rpW.shape[1], -1))
        WX_diff = torch.where(rpW > 0, rpW * self.U_X, rpW * self.L_X)
        if self.channel_size > 1:
            WX_diff = WX_diff.reshape(WX_diff.shape[0], WX_diff.shape[1], self.channel_size, -1).sum(dim=2)
        if WX_diff.shape[0] != self.batch_size:
            WX_diff = WX_diff.repeat(self.batch_size, 1, 1)

        if WX_diff.is_cuda: return selected_top_k_sum(WX_diff, self.selected_idx, self.eps)

        WX_selected_sum = self.wx_selected_sum(WX_diff)
        if self.eps == 0: return WX_selected_sum
        WX_diff[self.batch_indices, :, self.selected_idx] = 0

        for _ in range(self.eps):
            v, idx = WX_diff.max(-1, keepdim=True)
            WX_selected_sum += v
            WX_diff.scatter_(-1, idx, 0)

        return WX_selected_sum

    def compute_ldiff(self, rpW: torch.Tensor) -> torch.Tensor:
        if self.level is not None and self.level < self.eps:
            assert self.partial_indices is not None and self.partial_indices.ndim == 3
            assert self.num_selected() == 0
            # print(f"rpW shape: {rpW.shape}, self.U_X shape: {self.U_X.shape}, self.__U_X shape: {self.__U_X.shape}")
            WX_diff = torch.where(rpW < 0, rpW * self.__U_X, rpW * self.__L_X)
            if self.channel_size > 1:
                WX_diff = WX_diff.reshape(WX_diff.shape[0], WX_diff.shape[1], self.channel_size, -1).sum(dim=2)
            if WX_diff.shape[0] != self.batch_size:
                WX_diff = WX_diff.repeat(self.batch_size, 1, 1)
            assert WX_diff.is_cuda
            return bottom_bottom_sum(WX_diff, self.partial_pixel_indices, self.eps, self.level)  # WX_diff is per pixel, so index by pixels
        if self.partial_indices is not None:
            if self.partial_indices.ndim == 1:
                rpW = rpW[:, :, self.partial_indices]
            else:  # batched partial indices
                rpW = rpW.expand(self.batch_size, -1, -1).gather(dim=2, index=self.partial_indices.expand(-1, rpW.shape[1], -1))
        WX_diff = torch.where(rpW < 0, rpW * self.U_X, rpW * self.L_X)
        if self.channel_size > 1:
            WX_diff = WX_diff.reshape(WX_diff.shape[0], WX_diff.shape[1], self.channel_size, -1).sum(dim=2)
        if WX_diff.shape[0] != self.batch_size:
            WX_diff = WX_diff.repeat(self.batch_size, 1, 1)

        if WX_diff.is_cuda: return selected_bottom_k_sum(WX_diff, self.selected_idx, self.eps)

        WX_selected_sum = self.wx_selected_sum(WX_diff)
        if self.eps == 0: return WX_selected_sum
        WX_diff[self.batch_indices, :, self.selected_idx] = 0

        for _ in range(self.eps):
            v, idx = WX_diff.min(-1, keepdim=True)
            WX_selected_sum += v
            WX_diff.scatter_(-1, idx, 0)

        return WX_selected_sum

    def compute_ubound(self, rp: RelationalPolyhedra) -> torch.Tensor:
        center = rp.W @ self.x + rp.b
        diff = self.compute_udiff(rp.W)
        return center + diff

    def compute_lbound(self, rp: RelationalPolyhedra) -> torch.Tensor:
        center = rp.W @ self.x + rp.b
        diff = self.compute_ldiff(rp.W)
        return center + diff

class LinfBoundedInput(BoundedInput):
    def __init__(self, x: torch.Tensor, eps: float, pixel_value_range: tuple[float, float]=(0., 1.), bounds: Bounds=None):
        super().__init__(x, eps, pixel_value_range)
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = Bounds((x-eps).clamp(self.L, self.U), (x+eps).clamp(self.L, self.U))

    def compute_ubound(self, rp: RelationalPolyhedra) -> torch.Tensor:
        return rp.posW() @ self.bounds.u + rp.negW() @ self.bounds.l + rp.b

    def compute_lbound(self, rp: RelationalPolyhedra) -> torch.Tensor:
        return rp.posW() @ self.bounds.l + rp.negW() @ self.bounds.u + rp.b

class AbstractTransformer(ABC):
    def __init__(self):
        self.parent: Self | None = None

    @abstractmethod
    def backward(self, rp: RelationalPolyhedra, *args, **kwargs) -> None:
        """Back-substitute an affine bound `rp` on this transformer's output into one on its input, in place."""

    def backsubstitution(self) -> tuple[RelationalPolyhedra, RelationalPolyhedra]:
        assert isinstance(self, AffineAT)
        lrp, urp = self.rp.clone(), self.rp.clone()
        prev = self.parent

        while prev is not None:
            prev.backward(lrp, upper=False)
            prev.backward(urp, upper=True)
            prev = prev.parent

        return lrp, urp

    def backsubstitution_lrp(self) -> RelationalPolyhedra:
        assert isinstance(self, AffineAT)
        lrp = self.rp.clone()
        prev = self.parent

        while prev is not None:
            prev.backward(lrp, upper=False)
            prev = prev.parent

        return lrp

class AffineAT(AbstractTransformer):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.W = weight
        self.b = bias
        self.rp = RelationalPolyhedra(self.W, self.b)

    def backward(self, rp: RelationalPolyhedra, *args, **kwargs) -> None:
        """Back-substitute an affine bound through this affine layer, in place; lower and upper bounds pass the same way."""
        rp.b = rp.W @ self.rp.b + rp.b
        rp.W = rp.W @ self.rp.W

    @classmethod
    def from_linear(cls, layer: nn.Linear) -> Self:
        weight = layer.weight.detach().unsqueeze(0)
        if layer.bias is None:
            bias = torch.zeros(1, layer.out_features, 1, device=weight.device)
        else:
            bias = layer.bias.detach().reshape(1, layer.out_features, 1)
        return cls(weight, bias)

    @classmethod
    def from_conv2d(cls, layer: nn.Conv2d, input_shape) -> Self:
        kernel, bias = layer.weight.detach(), layer.bias.detach()
        stride, padding = layer.stride, layer.padding
        _, ci, hi, wi = input_shape
        kh, kw = kernel.shape[2], kernel.shape[3]
        co = kernel.shape[0]
        ho = (hi + 2 * padding[0] - kh) // stride[0] + 1
        wo = (wi + 2 * padding[1] - kw) // stride[1] + 1

        W = torch.zeros(co * ho * wo, ci * hi * wi, device=kernel.device)
        b = torch.zeros(co * ho * wo, device=kernel.device)
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

    def make_marginals(self, true_label: int, device: str | torch.device="cpu") -> None:
        """The margins of the true label over the other labels: row j is output[true_label] - output[other_j], the
        other labels in increasing order (the true label's own row, identically zero, is left out)."""
        W = self.W[:, true_label:true_label + 1, :] - torch.cat((self.W[:, :true_label, :], self.W[:, true_label + 1:, :]), dim=1)
        b = self.b[:, true_label:true_label + 1, :] - torch.cat((self.b[:, :true_label, :], self.b[:, true_label + 1:, :]), dim=1)
        self.rp = RelationalPolyhedra(W.to(device), b.to(device))

class ActivationAT(AbstractTransformer):
    def __init__(self):
        super().__init__()
        self.lrp: RelationalPolyhedra = None
        self.urp: RelationalPolyhedra = None

    def backward(self, rp: RelationalPolyhedra, *, upper: bool, **kwargs) -> None:
        """Back-substitute a lower (upper=False) or upper (upper=True) affine bound through this activation's relaxation, in place."""
        same, other = (self.urp, self.lrp) if upper else (self.lrp, self.urp)
        rp.b = rp.posW() @ same.b + rp.negW() @ other.b + rp.b
        rp.W = rp.posW() * same.W + rp.negW() * other.W

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

        lrp.W.transpose_(1, 2)
        urp.W.transpose_(1, 2)
        return lrp, urp

class SequentialAT:
    def __init__(self, ATs: list[AbstractTransformer]):
        self.ATs = SequentialAT.merge_consecutive_affine_layers(ATs)
        tail = self.ATs[-1]
        assert isinstance(tail, AffineAT)
        self.tail = tail
        self.tail.W = self.tail.rp.W.cpu()
        self.tail.b = self.tail.rp.b.cpu()

    @staticmethod
    def parse_layers(seq_net: nn.Sequential, x: torch.Tensor, prev: AbstractTransformer | None=None) -> list[AbstractTransformer]:
        ATs: list[AbstractTransformer] = []
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

    @staticmethod
    def merge_consecutive_affine_layers(ATs: list[AbstractTransformer]) -> list[AbstractTransformer]:
        merged_ATs: list[AbstractTransformer] = []
        prev_affine: AffineAT | None = None
        for AT in ATs:
            if isinstance(AT, AffineAT):
                if prev_affine is None:
                    prev_affine = AT
                else:
                    prev_affine.rp.W = AT.rp.W @ prev_affine.rp.W
                    prev_affine.rp.b = AT.rp.W @ prev_affine.rp.b + AT.rp.b
            else:
                if prev_affine is not None:
                    merged_ATs.append(prev_affine)
                    AT.parent = prev_affine
                    prev_affine = None
                merged_ATs.append(AT)
        merged_ATs.append(prev_affine)
        return merged_ATs

    def update_activation_rp(self, bounded_in: BoundedInput) -> None:
        for AT in self.ATs:
            if isinstance(AT, ActivationAT):
                lrp_wrt_inputs, urp_wrt_inputs = AT.parent.backsubstitution()
                pre_activation_bounds = bounded_in.compute_bounds(lrp_wrt_inputs, urp_wrt_inputs)
                AT.lrp, AT.urp = AT.compute_rp(pre_activation_bounds)

    def get_preactivation_bounds(self, bounded_in: BoundedInput) -> list[Bounds]:
        pre_activation_bounds_list = []
        for AT in self.ATs:
            if isinstance(AT, ActivationAT):
                lrp_wrt_inputs, urp_wrt_inputs = AT.parent.backsubstitution()
                pre_activation_bounds = bounded_in.compute_bounds(lrp_wrt_inputs, urp_wrt_inputs)
                AT.lrp, AT.urp = AT.compute_rp(pre_activation_bounds)
                pre_activation_bounds_list.append(pre_activation_bounds)
        return pre_activation_bounds_list

    def remove_activation_rp(self) -> None:
        for AT in self.ATs:
            if isinstance(AT, ActivationAT):
                AT.lrp = None
                AT.urp = None
