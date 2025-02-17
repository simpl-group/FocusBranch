import torch
import torch.nn as nn
from typing import Self, Optional, Tuple

torch.set_default_dtype(torch.float32)

class RelationalPolyhedra:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        self.W = weight  # (batch_size, output_dim, input_dim)
        self.b = bias  # (batch_size, output_dim, 1)
    
    @classmethod
    def clone(cls, other: Self) -> Self:
        return cls(other.W.clone(), other.b.clone())

class Bounds:
    def __init__(self, lbounds: torch.Tensor, ubounds: torch.Tensor):
        self.l = lbounds  # (batch_size, features, 1)
        self.u = ubounds  # (batch_size, features, 1)
    
    @classmethod
    def clone(cls, other: Self) -> Self:
        return cls(other.l.clone(), other.u.clone())

    def __add__(self, c: torch.Tensor) -> Self:
        return Bounds(self.l + c, self.u + c)

    def __radd__(self, c: torch.Tensor) -> Self:
        return self.__add__(c)

class AbstractElement:
    init_bounds = None
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
    def from_input(cls, init_bounds: Bounds) -> Self:
        cls.init_bounds = init_bounds
        batch_size, d, _ = init_bounds.l.shape
        rp = RelationalPolyhedra(torch.eye(d).expand(batch_size, -1, -1), torch.zeros_like(init_bounds.l))
        return cls(rp, rp, parent=None)
    
    def get_bounds(self) -> Bounds:
        lrp_wrt_inputs, urp_wrt_inputs = self.backsubstitution()
        return AbstractElement.compute_bounds(lrp_wrt_inputs, urp_wrt_inputs, AbstractElement.init_bounds)

    @staticmethod
    def compute_bounds(lrp: RelationalPolyhedra, urp: RelationalPolyhedra, init_bounds: Bounds) -> Bounds:
        lrp_pos = torch.where(lrp.W > 0, lrp.W, 0)
        lrp_neg = torch.where(lrp.W < 0, lrp.W, 0)
        urp_pos = torch.where(urp.W > 0, urp.W, 0)
        urp_neg = torch.where(urp.W < 0, urp.W, 0)

        return Bounds(torch.bmm(lrp_pos, init_bounds.l) + torch.bmm(lrp_neg, init_bounds.u) + lrp.b,
                      torch.bmm(urp_pos, init_bounds.u) + torch.bmm(urp_neg, init_bounds.l) + urp.b)

    def backsubstitution(self) -> Tuple[RelationalPolyhedra]:
        lrp_W, lrp_b, urp_W, urp_b = self.lrp.W, self.lrp.b, self.urp.W, self.urp.b
        prev = self.parent

        while prev is not None:
            lrp_posW = torch.where(lrp_W > 0, lrp_W, 0)
            lrp_negW = torch.where(lrp_W < 0, lrp_W, 0)
            urp_posW = torch.where(urp_W > 0, urp_W, 0)
            urp_negW = torch.where(urp_W < 0, urp_W, 0)

            lrp_b = torch.bmm(lrp_posW, prev.lrp.b) + torch.bmm(lrp_negW, prev.urp.b) + lrp_b
            urp_b = torch.bmm(urp_posW, prev.urp.b) + torch.bmm(urp_negW, prev.lrp.b) + urp_b
            lrp_W = torch.bmm(lrp_posW, prev.lrp.W) + torch.bmm(lrp_negW, prev.urp.W)
            urp_W = torch.bmm(urp_posW, prev.urp.W) + torch.bmm(urp_negW, prev.lrp.W)

            prev = prev.parent
        
        return RelationalPolyhedra(lrp_W, lrp_b), RelationalPolyhedra(urp_W, urp_b)

# Abstract Transformer
class AbstractTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, parent: AbstractElement) -> AbstractElement:
        raise NotImplementedError("Subclasses must implement forward method.")
    
    def __call__(self, ae: AbstractElement) -> AbstractElement:
        return self.forward(ae)

class AffineAT(AbstractTransformer):
    def __init__(self, layer: nn.Linear, batch_size: int):
        super().__init__()
        self.W = layer.weight.detach().unsqueeze(0).expand(batch_size, -1, -1)
        self.b = layer.bias.detach().reshape(1, -1, 1).expand(batch_size, -1, -1)
    
    @classmethod
    def from_conv2d(cls, layer: nn.Conv2d, input_shape) -> Self:
        instance = cls.__new__(cls)
        super(AffineAT, instance).__init__()
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

        instance.W = W.unsqueeze(0).expand(batch_size, -1, -1)
        instance.b = b.reshape(1, -1, 1).expand(batch_size, -1, -1)
        return instance

    def make_marginal(self, true_labels: torch.Tensor, target_label: int) -> Self:
        batch_size = true_labels.shape[0]
        batch_idx = torch.arange(batch_size, device=self.W.device)
        self.W = (self.W[batch_idx, true_labels, :] - self.W[:, target_label, :]).unsqueeze(1)
        self.b = (self.b[batch_idx, true_labels, :] - self.b[:, target_label, :]).unsqueeze(1)

    def forward(self, parent: AbstractElement) -> AbstractElement:
        rp = RelationalPolyhedra(self.W, self.b)
        return AbstractElement(rp, rp, parent)

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

    @classmethod
    def from_sequential(cls, seq_net: nn.Sequential, x: torch.Tensor, true_labels: torch.Tensor, target_label: int):
        layers = []
        for layer in seq_net:
            if isinstance(layer, nn.Linear):
                layers.append(AffineAT(layer, x.shape[0]))
            elif isinstance(layer, nn.Conv2d):
                layers.append(AffineAT.from_conv2d(layer, x.shape))
            elif isinstance(layer, nn.ReLU):
                layers.append(ReLUAT())
            elif isinstance(layer, nn.Flatten):
                pass  #
            elif isinstance(layer, nn.Sequential):  # check if this is needed
                layers.append(cls.from_sequential(layer, x))
            else:
                raise NotImplementedError(f"No DeepPoly AT for layer {layer}")
            x = layer(x)
        
        lastAT = layers[-1]
        assert isinstance(lastAT, AffineAT)
        lastAT.make_marginal(true_labels, target_label)

        return cls(layers)

    def __call__(self, ae: AbstractElement) -> AbstractElement:
        return self.forward(ae)
