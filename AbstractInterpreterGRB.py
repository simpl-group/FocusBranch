import numpy as np
import torch.nn as nn
from typing import List, Self, Optional, Tuple, Union
import warnings
import gurobipy as gp
from gurobipy import GRB

def bmm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.shape[0] == B.shape[0], "Batch sizes must match"
    assert A.shape[2] == B.shape[1], "Inner dimensions must match for matrix multiplication"
    return np.einsum('bij,bjk->bik', A, B)

class RelationalPolyhedra:
    def __init__(self, weight: np.ndarray, bias: np.ndarray):
        self.W = weight  # (batch_size, output_dim, input_dim)
        self.b = bias  # (batch_size, output_dim, 1)

    @classmethod
    def copy(cls, other: Self) -> Self:
        return cls(other.W.copy(), other.b.copy())

    def copy(self) -> Self:
        return RelationalPolyhedra(self.W.copy(), self.b.copy())

    def posW(self) -> np.ndarray:
        return np.maximum(self.W, 0)

    def negW(self) -> np.ndarray:
        return np.minimum(self.W, 0)

class Bounds:
    def __init__(self, lbounds: np.ndarray, ubounds: np.ndarray):
        self.l = lbounds  # (batch_size, features, 1)
        self.u = ubounds

    @classmethod
    def copy(cls, other: Self) -> Self:
        return cls(other.l.copy(), other.u.copy())

    def __add__(self, other: Union[np.ndarray, Self]) -> Self:
        if isinstance(other, Bounds):
            return Bounds(self.l + other.l, self.u + other.u)
        else:
            return Bounds(self.l + other, self.u + other)

    def __radd__(self, other: np.ndarray) -> Self:
        return self.__add__(other)

    def __sub__(self, other: Union[np.ndarray, Self]) -> Self:
        if isinstance(other, Bounds):
            return Bounds(self.l - other.l, self.u - other.u)
        else:
            return Bounds(self.l - other, self.u - other)

    def __rsub__(self, other: np.ndarray) -> Self:
        return Bounds(other - self.l, other - self.u)

class BoundedInput:
    def __init__(self, x: np.ndarray, eps, pixel_value_range=(0, 1)):
        self.x = x
        self.eps = eps
        self.L, self.U = pixel_value_range
        self.batch_size, self.num_features, _ = x.shape
        self.batch_indices = np.arange(self.batch_size)[:, np.newaxis]

    def compute_diff(self, rpW: np.ndarray, sign: int=1) -> np.ndarray:
        raise NotImplementedError

    def compute_bound(self, rp: RelationalPolyhedra, sign: int=1) -> np.ndarray:
        center = bmm(rp.W, self.x) + rp.b  # (batch, d1, 1)
        diff = self.compute_diff(rp.W, sign=sign)
        return center + diff

    def compute_bounds(self, lrp: RelationalPolyhedra, urp: RelationalPolyhedra) -> Bounds:
        return Bounds(lbounds=self.compute_bound(lrp, -1), ubounds=self.compute_bound(urp, 1))

class L0BoundedInput(BoundedInput):
    def __init__(self, x: np.ndarray, eps: int, gmodel: gp.Model, pixel_value_range=(0, 1)):
        super().__init__(x, eps, pixel_value_range)
        self.gmodel: gp.Model = gmodel
        self.locs = self.gmodel.addMVar((self.num_features,), vtype=GRB.BINARY, name="loc")
        self.gmodel.addConstr(self.locs.sum() == eps)

    def compute_diff(self, rpW: np.ndarray, sign: int=1) -> np.ndarray:
        # rpW.shape[1] number of nuerons in next layer
        X = np.tile(self.x.reshape(self.batch_size, 1, -1), (1, rpW.shape[1], 1))
        WX_diff = np.where(rpW * sign > 0, rpW * (self.U - X), rpW * (self.L - X))
        return (WX_diff * self.locs).sum(axis=2)[:,:, np.newaxis]

class LinfBoundedInput(BoundedInput):
    def __init__(self, x: np.ndarray, eps: float, pixel_value_range=(0, 1), bounds: Bounds=None):
        super().__init__(x, eps, pixel_value_range)
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = Bounds((x-eps).clip(self.L, self.U), (x+eps).clip(self.L, self.U))

    def compute_bounds(self, lrp: RelationalPolyhedra, urp: RelationalPolyhedra) -> Bounds:
        return Bounds(bmm(lrp.posW(), self.bounds.l) + bmm(lrp.negW(), self.bounds.u) + lrp.b,
                      bmm(urp.posW(), self.bounds.u) + bmm(urp.negW(), self.bounds.l) + urp.b)

class AbstractTransformer(nn.Module):

    bounded_in: Optional[BoundedInput] = None

    def __init__(self):
        self.parent: Optional[Self] = None
        super().__init__()

    def forward(self, parent: Optional[Self]=None):
        ...

    def __call__(self, parent: Optional[Self]=None) -> Self:
        self.parent = parent
        self.forward(parent)
        return self

    def get_bounds(self) -> Bounds:
        lrp_wrt_inputs, urp_wrt_inputs = self.backsubstitution()
        return AbstractTransformer.bounded_in.compute_bounds(lrp_wrt_inputs, urp_wrt_inputs)

    def backsubstitution(self) -> Tuple[RelationalPolyhedra]:
        assert isinstance(self, AffineAT)
        lrp, urp = self.rp.copy(), self.rp.copy()
        prev = self.parent

        while prev is not None:
            if isinstance(prev, ActivationAT):
                lrp.b = bmm(lrp.posW(), prev.lrp.b) + bmm(lrp.negW(), prev.urp.b) + lrp.b
                urp.b = bmm(urp.posW(), prev.urp.b) + bmm(urp.negW(), prev.lrp.b) + urp.b
                lrp.W = lrp.posW() * prev.lrp.W + lrp.negW() * prev.urp.W
                urp.W = urp.posW() * prev.urp.W + urp.negW() * prev.lrp.W
            elif isinstance(prev, AffineAT):
                lrp.b = bmm(lrp.posW(), prev.rp.b) + bmm(lrp.negW(), prev.rp.b) + lrp.b
                urp.b = bmm(urp.posW(), prev.rp.b) + bmm(urp.negW(), prev.rp.b) + urp.b
                lrp.W = bmm(lrp.posW(), prev.rp.W) + bmm(lrp.negW(), prev.rp.W)
                urp.W = bmm(urp.posW(), prev.rp.W) + bmm(urp.negW(), prev.rp.W)
            else: raise NotImplementedError(f"Transform for {prev} is unclear")
            prev = prev.parent

        return lrp, urp

class AffineAT(AbstractTransformer):
    def __init__(self, weight: np.ndarray, bias: np.ndarray):
        super().__init__()
        self.W = weight
        self.b = bias
        self.rp = RelationalPolyhedra(self.W, self.b)

    @staticmethod
    def expand(weight: np.ndarray, bias: np.ndarray, batch_size: int) -> Tuple[np.ndarray]:
        return (np.tile(weight[np.newaxis,:,:], (batch_size, 1, 1)),
                np.tile(bias.reshape(1, -1, 1), (batch_size, 1, 1)))

    @classmethod
    def from_linear(cls, layer: nn.Linear, batch_size: int) -> Self:
        W, b = AffineAT.expand(layer.weight.detach().numpy(), layer.bias.detach().numpy(), batch_size)
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

        W = np.zeros(co * ho * wo, ci * hi * wi)
        b = np.zeros(co * ho * wo)
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

    def make_marginal(self, true_labels: np.ndarray, target_label: int) -> Self:
        batch_size = true_labels.shape[0]
        batch_idx = np.arange(batch_size)
        W = np.expand_dims(self.W[batch_idx, true_labels, :] - self.W[:, target_label, :], 1)
        b = np.expand_dims(self.b[batch_idx, true_labels, :] - self.b[:, target_label, :], 1)
        self.rp = RelationalPolyhedra(W, b)

class ActivationAT(AbstractTransformer):
    def __init__(self):
        super().__init__()
        self.lrp: Optional[RelationalPolyhedra] = None
        self.urp: Optional[RelationalPolyhedra] = None

    @staticmethod
    def compute_rp(bounds: Bounds) -> (RelationalPolyhedra):
        raise NotImplementedError("Subclasses must implement compute_rp method.")

    def forward(self, parent: Optional[AbstractTransformer]=None):
        self.lrp, self.urp = self.compute_rp(parent.get_bounds())

class ReLUAT(ActivationAT):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_rp(bounds: Bounds) -> (RelationalPolyhedra):
        # Initialization
        lrp = RelationalPolyhedra(np.ones_like(bounds.l), np.zeros_like(bounds.l))
        urp = lrp.copy()

        # Case 1: ubound <= 0
        below_nodes = bounds.u <= 0
        urp.W = np.where(below_nodes, 0, urp.W)

        # Case 2: lbound >= 0
        above_nodes = bounds.l >= 0
        urp.W = np.where(above_nodes, 1, urp.W)

        # Case 3: lbound <= 0 <= ubound
        uk = bounds.u / (bounds.u - bounds.l)
        ub = (1 - uk) * bounds.u
        crossing_nodes = ~(below_nodes | above_nodes)
        urp.W = np.where(crossing_nodes, uk, urp.W)
        urp.b = np.where(crossing_nodes, ub, urp.b)

        # Case a: lbound + ubound > 0
        lrp.W = np.where(bounds.l + bounds.u >= 0, 1, lrp.W)
        # Case b: lbound + ubound <= 0
        lrp.W = np.where(bounds.l + bounds.u <= 0, 0, lrp.W)

        lrp.W = lrp.W.swapaxes(-1, -2)
        urp.W = urp.W.swapaxes(-1, -2)
        return lrp, urp

class SequentialAT(nn.Sequential):
    def __init__(self, layers: List[AbstractTransformer]):
        assert isinstance(layers[0], AffineAT)
        assert isinstance(layers[-1], AffineAT)
        super().__init__(*layers)

    @staticmethod
    def parse_layers(seq_net: nn.Sequential, x: np.ndarray) -> List[AbstractTransformer]:
        layers = []
        for layer in seq_net:
            if isinstance(layer, nn.Linear):
                layers.append(AffineAT.from_linear(layer, x.shape[0]))
            elif isinstance(layer, nn.Conv2d):
                layers.append(AffineAT.from_conv2d(layer, x.shape))
            elif isinstance(layer, nn.ReLU):
                layers.append(ReLUAT())
            elif isinstance(layer, nn.Sequential):
                layers += SequentialAT.parse_layers(layer, x)
            else:
                warnings.warn(f"No Abstract Transformer for layer {layer}, skip for now", RuntimeWarning)
            x = layer(x)
        return layers

    @classmethod
    def from_sequential(cls, seq_net: nn.Sequential, x: np.ndarray) -> Self:
        return cls(SequentialAT.parse_layers(seq_net, x))

    def tail(self) -> AffineAT:
        return self[-1]

    def __call__(self, bounded_in: BoundedInput) -> AbstractTransformer:
        AbstractTransformer.bounded_in=bounded_in
        return self.forward(None)
