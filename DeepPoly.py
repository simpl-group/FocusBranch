import torch
import torch.nn as nn
from typing import Self, Optional, Tuple, Union
import warnings

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
        self.u = ubounds  # (batch_size, features, 1)
    
    @classmethod
    def clone(cls, other: Self) -> Self:
        return cls(other.l.clone(), other.u.clone())

    def __add__(self, other: Union[torch.Tensor, Self]) -> Self:
        if isinstance(other, Bounds):
            return Bounds(self.l + other.l, self.u + other.u)
        else:
            return Bounds(self.l + other, self.u + other)

    def __radd__(self, other: torch.Tensor) -> Self:
        return self.__add__(other)

    def __sub__(self, other: Union[torch.Tensor, Self]) -> Self:
        if isinstance(other, Bounds):
            return Bounds(self.l - other.l, self.u - other.u)
        else:
            return Bounds(self.l - other, self.u - other)

    def __rsub__(self, other: torch.Tensor) -> Self:
        return Bounds(other - self.l, other - self.u)

class BoundedInput:
    def __init__(self, x: torch.Tensor, eps, pixel_value_range=(0, 1)):
        self.x = x
        self.eps = eps
        self.L, self.U = pixel_value_range
        self.batch_size, self.num_features, _ = x.shape
    
    def compute_diff(self, rpW: torch.Tensor, lower: bool=True) -> torch.Tensor:
        raise NotImplementedError
    
    def compute_bound(self, rp: RelationalPolyhedra, lower: bool=True) -> torch.Tensor:
        center = torch.bmm(rp.W, self.x) + rp.b  # (batch, d1, 1)
        diff = self.compute_diff(rp.W, lower=lower)
        return center + diff

    def compute_bounds(self, lrp: RelationalPolyhedra, urp: RelationalPolyhedra) -> Bounds:
        lcenter = torch.bmm(lrp.W, self.x) + lrp.b  # (batch, d1, 1)
        ucenter = torch.bmm(urp.W, self.x) + urp.b
        ldiff = self.compute_diff(lrp.W, lower=True)
        udiff = self.compute_diff(urp.W, lower=False)
        return Bounds(lbounds=lcenter+ldiff, ubounds=ucenter+udiff)

class L0BoundedInput(BoundedInput):
    def __init__(self, x, eps: int, pixel_value_range=(0, 1)):
        super().__init__(x, eps, pixel_value_range)

    def compute_diff(self, rpW: torch.Tensor, lower: bool=True) -> torch.Tensor:
        # rpW.shape[-2] number of nuerons in next layer
        X = self.x.reshape(self.batch_size, 1, -1).expand(-1, rpW.shape[-2], -1)
        WX_L = rpW * (X - self.L)  # (batch, d1, d0)
        WX_U = rpW * (self.U - X)
        if lower:
            WX_ldiff = torch.where(rpW > 0, -WX_L, WX_U)
            WX_ldiff_sorted = WX_ldiff.sort(dim=2, descending=False).values
            diff = WX_ldiff_sorted[:, :, :self.eps].sum(dim=-1, keepdim=True)
        else:
            WX_udiff = torch.where(rpW < 0, -WX_L, WX_U)
            WX_udiff_sorted = WX_udiff.sort(dim=2, descending=True).values
            diff = WX_udiff_sorted[:, :, :self.eps].sum(dim=-1, keepdim=True)
        return diff

class LinfBoundedInput(BoundedInput):
    def __init__(self, x: torch.Tensor, eps: float, pixel_value_range=(0, 1), bounds: Bounds=None):
        super().__init__(x, eps, pixel_value_range)
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = Bounds((x-eps).clamp(self.L, self.U), (x+eps).clamp(self.L, self.U))

    def compute_bounds(self, lrp: RelationalPolyhedra, urp: RelationalPolyhedra) -> Bounds:
        return Bounds(torch.bmm(lrp.posW(), self.bounds.l) + torch.bmm(lrp.negW(), self.bounds.u) + lrp.b,
                      torch.bmm(urp.posW(), self.bounds.u) + torch.bmm(urp.negW(), self.bounds.l) + urp.b)

class AbstractElement:
    bounded_in = None
    def __init__(self,
                 lrp: Optional[RelationalPolyhedra]=None,
                 urp: Optional[RelationalPolyhedra]=None,
                 parent: Optional[Self]=None
                 ):
        # Lower and upper Relational Polyhedra
        self.lrp = lrp
        self.urp = urp
        self.parent = parent

    @classmethod
    def from_bounded_input(cls, bounded_in: BoundedInput) -> Self:
        cls.bounded_in = bounded_in
        rp = RelationalPolyhedra(
            weight=torch.eye(bounded_in.num_features).expand(bounded_in.batch_size, -1, -1),
            bias=torch.zeros_like(bounded_in.x))
        return cls(rp, rp, parent=None)
    
    def get_bounds(self) -> Bounds:
        lrp_wrt_inputs, urp_wrt_inputs = self.backsubstitution()
        return AbstractElement.bounded_in.compute_bounds(lrp_wrt_inputs, urp_wrt_inputs)

    def backsubstitution(self) -> Tuple[RelationalPolyhedra]:
        lrp, urp = self.lrp.clone(), self.urp.clone()
        prev = self.parent

        while prev is not None:
            lrp.b = torch.bmm(lrp.posW(), prev.lrp.b) + torch.bmm(lrp.negW(), prev.urp.b) + lrp.b
            urp.b = torch.bmm(urp.posW(), prev.urp.b) + torch.bmm(urp.negW(), prev.lrp.b) + urp.b
            lrp.W = torch.bmm(lrp.posW(), prev.lrp.W) + torch.bmm(lrp.negW(), prev.urp.W)
            urp.W = torch.bmm(urp.posW(), prev.urp.W) + torch.bmm(urp.negW(), prev.lrp.W)

            prev = prev.parent
        
        return lrp, urp

class AbstractTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, parent: AbstractElement) -> AbstractElement:
        raise NotImplementedError("Subclasses must implement forward method.")
    
    def __call__(self, ae: AbstractElement) -> AbstractElement:
        return self.forward(ae)

class AffineAT(AbstractTransformer):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.W = weight
        self.b = bias
        self.rp = RelationalPolyhedra(self.W, self.b)
    
    @staticmethod
    def expand(weight: torch.Tensor, bias: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor]:
        return (weight.unsqueeze(0).expand(batch_size, -1, -1),
                bias.reshape(1, -1, 1).expand(batch_size, -1, -1))

    @classmethod
    def from_linear(cls, layer: nn.Linear, batch_size: int) -> Self:
        W, b = AffineAT.expand(layer.weight.detach(), layer.bias.detach(), batch_size)
        return cls(W, b)

    @classmethod
    def from_conv2d(cls, layer: nn.Conv2d, input_shape) -> Self:
        kernel, bias = layer.weight.detach(), layer.bias.detach()
        stride, padding = layer.stride, layer.padding
        batch_size, ci, hi, wi = input_shape
        kh, kw = kernel.shape[2], kernel.shape[3]
        co = kernel.shape[0]
        ho = (hi + 2 * padding[0] - kh) // stride[0] + 1
        wo = (wi + 2 * padding[1] - kw) // stride[1] + 1

        W = torch.zeros(co * ho * wo, ci * hi * wi)
        b = torch.zeros(co * ho * wo)
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
        W, b = AffineAT.expand(W, b, batch_size)
        return cls(weight=W, bias=b)

    def make_marginal(self, true_labels: torch.Tensor, target_label: int) -> Self:
        batch_size = true_labels.shape[0]
        batch_idx = torch.arange(batch_size, device=self.W.device)
        W = (self.W[batch_idx, true_labels, :] - self.W[:, target_label, :]).unsqueeze(1)
        b = (self.b[batch_idx, true_labels, :] - self.b[:, target_label, :]).unsqueeze(1)
        self.rp = RelationalPolyhedra(W, b)

    def forward(self, parent: AbstractElement) -> AbstractElement:
        return AbstractElement(self.rp, self.rp, parent)

class ActivationAT(AbstractTransformer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_rp(bounds: Bounds) -> (RelationalPolyhedra):
        raise NotImplementedError("Subclasses must implement compute_rp method.")
    
    def forward(self, parent: AbstractElement) -> AbstractElement:
        lrp, urp = self.compute_rp(parent.get_bounds())
        return AbstractElement(lrp, urp, parent)

class ReLUAT(ActivationAT):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_rp(bounds: Bounds) -> (RelationalPolyhedra):
        batch_size, N, _ = bounds.l.shape

        # Initialization
        lrp = RelationalPolyhedra(torch.ones((batch_size, N, 1)), torch.zeros((batch_size, N, 1)))
        urp = RelationalPolyhedra(torch.ones((batch_size, N, 1)), torch.zeros((batch_size, N, 1)))

        # Case 1: ubound <= 0
        below_nodes = bounds.u <= 0
        lrp.W = torch.where(below_nodes, torch.zeros_like(lrp.W), lrp.W)
        urp.W = torch.where(below_nodes, torch.zeros_like(urp.W), urp.W)

        # Case 2: lbound >= 0
        above_nodes = bounds.l >= 0
        lrp.W = torch.where(above_nodes, torch.ones_like(lrp.W), lrp.W)
        urp.W = torch.where(above_nodes, torch.ones_like(urp.W), urp.W)

        # Case 3: lbound <= 0 <= ubound
        crossing_nodes = ~(below_nodes | above_nodes)

        uk = bounds.u / (bounds.u - bounds.l)
        ub = (1 - uk) * bounds.u
        urp.W = torch.where(crossing_nodes, uk, urp.W)
        urp.b = torch.where(crossing_nodes, ub, urp.b)

        # Case 3a: lbound + ubound > 0
        lrp.W = torch.where(crossing_nodes & (bounds.l + bounds.u > 0), torch.ones_like(lrp.W), lrp.W)
        # Case 3b: lbound + ubound <= 0
        lrp.W = torch.where(crossing_nodes & (bounds.l + bounds.u <= 0), torch.zeros_like(lrp.W), lrp.W)

        lrp.W = torch.diag_embed(lrp.W.squeeze(-1))
        urp.W = torch.diag_embed(urp.W.squeeze(-1))
        return lrp, urp

class SequentialAT(nn.Sequential):
    def __init__(self, layers: AbstractTransformer):
        super().__init__(*layers)
        assert isinstance(layers[0], AffineAT)
        assert isinstance(layers[-1], AffineAT)

    @classmethod
    def from_sequential(cls, seq_net: nn.Sequential, x: torch.Tensor):
        layers = []
        for layer in seq_net:
            if isinstance(layer, nn.Linear):
                layers.append(AffineAT.from_linear(layer, x.shape[0]))
            elif isinstance(layer, nn.Conv2d):
                layers.append(AffineAT.from_conv2d(layer, x.shape))
            elif isinstance(layer, nn.ReLU):
                layers.append(ReLUAT())
            elif isinstance(layer, nn.Sequential):  # check if this is needed
                layers.append(cls.from_sequential(layer, x))
            else:
                warnings.warn(f"No Abstract Transformer for layer {layer}, skip for now", RuntimeWarning)
            x = layer(x)

        return cls(layers)

    def __call__(self, bounded_in: BoundedInput) -> AbstractElement:
        ae = AbstractElement.from_bounded_input(bounded_in=bounded_in)
        return self.forward(ae)
