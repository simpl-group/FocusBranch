import torch
import torch.nn as nn
from typing import Self, Optional

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

class AbstractElement:
    def __init__(self, bounds: Optional[Bounds]=None, parent: Optional[Self]=None,
                 lrp: Optional[RelationalPolyhedra]=None, urp: Optional[RelationalPolyhedra]=None):
        self.bounds = bounds
        # Lower and upper Relational Polyhedra
        self.lrp = lrp
        self.urp = urp
        self.parent = parent

    @classmethod
    def from_relational_polyhedra(cls, lrp: RelationalPolyhedra, urp: RelationalPolyhedra, parent: Self) -> Self:
        return cls(None, parent, lrp, urp)

    def compute_bounds(self, lrp: RelationalPolyhedra, urp: RelationalPolyhedra, init_bounds: Self):
        lrp_pos = torch.where(lrp.W > 0, lrp.W, torch.zeros_like(lrp.W))
        lrp_neg = torch.where(lrp.W < 0, lrp.W, torch.zeros_like(lrp.W))
        urp_pos = torch.where(urp.W > 0, urp.W, torch.zeros_like(urp.W))
        urp_neg = torch.where(urp.W < 0, urp.W, torch.zeros_like(urp.W))

        self.bounds = Bounds(torch.bmm(lrp_pos, init_bounds.l) + torch.bmm(lrp_neg, init_bounds.u) + lrp.b,
                             torch.bmm(urp_pos, init_bounds.u) + torch.bmm(urp_neg, init_bounds.l) + urp.b)

    def backsubstitution(self, require_rp=False):
        if self.parent is None:
            return

        lrp_W, lrp_b, urp_W, urp_b = self.lrp.W, self.lrp.b, self.urp.W, self.urp.b
        current = self.parent

        while current.parent is not None:
            lrp_posW = torch.where(lrp_W > 0, lrp_W, torch.zeros_like(lrp_W))
            lrp_negW = torch.where(lrp_W < 0, lrp_W, torch.zeros_like(lrp_W))
            urp_posW = torch.where(urp_W > 0, urp_W, torch.zeros_like(urp_W))
            urp_negW = torch.where(urp_W < 0, urp_W, torch.zeros_like(urp_W))

            lrp_b = torch.bmm(lrp_posW, current.lrp.b) + torch.bmm(lrp_negW, current.urp.b) + lrp_b
            urp_b = torch.bmm(urp_posW, current.urp.b) + torch.bmm(urp_negW, current.lrp.b) + urp_b
            lrp_W = torch.bmm(lrp_posW, current.lrp.W) + torch.bmm(lrp_negW, current.urp.W)
            urp_W = torch.bmm(urp_posW, current.urp.W) + torch.bmm(urp_negW, current.lrp.W)

            current = current.parent
        
        lrp_wrt_inputs, urp_wrt_inputs = RelationalPolyhedra(lrp_W, lrp_b), RelationalPolyhedra(urp_W, urp_b)
        if require_rp:
            return lrp_wrt_inputs, urp_wrt_inputs
        else:
            self.compute_bounds(lrp_wrt_inputs, urp_wrt_inputs, current.bounds)

# Abstract Transformer
class AffineAT(nn.Module):
    def __init__(self, layer:nn.Linear, batch_size: int):
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

    @classmethod
    def Verifier(cls, true_labels: torch.Tensor, target_label: int, num_labels=10) -> Self:
        instance = cls.__new__(cls)
        super(AffineAT, instance).__init__()
        batch_size = true_labels.shape[0]
        W = torch.zeros(batch_size, num_labels)
        W[torch.arange(batch_size), true_labels] = 1
        W[torch.arange(batch_size), target_label] = -1
        instance.W = W.unsqueeze(1)
        instance.b = torch.zeros(batch_size, 1, 1)
        return instance

    def forward(self, parent: AbstractElement) -> AbstractElement:
        rp = RelationalPolyhedra(self.W, self.b)
        return AbstractElement.from_relational_polyhedra(rp, rp, parent)
    
    def __call__(self, parent: AbstractElement) -> AbstractElement:
        return self.forward(parent)

class ActivationAT(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_rp(bounds: Bounds) -> (RelationalPolyhedra):
        raise NotImplementedError("Subclasses must implement compute_rp method.")
    
    def forward(self, ae: AbstractElement) -> AbstractElement:
        ae.backsubstitution()
        lrp, urp = self.compute_rp(ae.bounds)
        return AbstractElement.from_relational_polyhedra(lrp, urp, ae)
    
    def __call__(self, ae: AbstractElement) -> AbstractElement:
        return self.forward(ae)

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
    def __init__(self, layers):
        super().__init__(*layers)

    @classmethod
    def from_sequential(cls, network: nn.Sequential, x: torch.Tensor):
        layers = []
        for layer in network.children():
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

        return cls(layers)

    def __call__(self, ae: AbstractElement) -> AbstractElement:
        return self.forward(ae)
