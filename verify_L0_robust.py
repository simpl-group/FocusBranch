import argparse
import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from networks import get_network
import DeepPoly

DEVICE = "cpu"

def compute_x1_bounds_with_saliency(x0: torch.Tensor, W: torch.Tensor, b: torch.Tensor,
                                    eps: int, coeffiencts_x1: torch.Tensor, collactive=False) -> DeepPoly.Bounds:
    assert coeffiencts_x1.ndim == 3 # (batch, x1_neurons, 1)
    batch = W.shape[0]
    d1 = W.shape[1]
    d0 = W.shape[2]

    x0 = x0.reshape(batch, -1, 1)
    center = torch.bmm(W, x0) + b  # (batch, x1_neurons, 1)
    # (batch, x1_neurons, x0_neurons)
    X = x0.reshape(batch, 1, -1).expand(batch, d1, d0)
    WX = W * X
    negW = W < 0
    posW = W >= 0

    W_ldiff = torch.zeros_like(W)
    # assumes the input pixel range is between 0 and 1
    W_ldiff[posW] = WX[posW]
    W_ldiff[negW] = WX[negW] - W[negW]
    W_udiff = torch.zeros_like(W)
    W_udiff[posW] = W[posW] - WX[posW]
    W_udiff[negW] = - WX[negW]

    # each pixel's per-neuron contribution to the lower bound
    per_nueron_contribution = torch.where(coeffiencts_x1 > 0, W_ldiff * coeffiencts_x1, W_udiff * -coeffiencts_x1)
    if collactive:
        collactive_contribution = per_nueron_contribution.sum(1).sum(0)
        collactive_pixel_idx_set = collactive_contribution.sort(descending=True).indices[:eps]
        pixel_idx_set = collactive_pixel_idx_set.unsqueeze(0)
    else:
        contribution = per_nueron_contribution.sum(1)
        pixel_idx_set = contribution.sort(descending=True).indices[:, :eps]
    pixel_idx_set = pixel_idx_set.unsqueeze(1).expand(batch, d1, eps)

    lbounds = center - W_ldiff.gather(dim=2, index=pixel_idx_set).sum(dim=-1, keepdim=True)
    ubounds = center + W_udiff.gather(dim=2, index=pixel_idx_set).sum(dim=-1, keepdim=True)
    return DeepPoly.Bounds(lbounds, ubounds)

def compute_x1_bounds(x0: torch.Tensor, W: torch.Tensor, b: torch.Tensor, eps: int) -> DeepPoly.Bounds:
    batch = x0.shape[0]
    x0 = x0.reshape(batch, -1, 1)
    center = torch.bmm(W, x0) + b  # (batch, x1_neurons, 1)
    # W.shape[-2] number of nuerons in x1
    X = x0.reshape(batch, 1, -1).expand(-1, W.shape[-2], -1)
    WX = W * X  # (batch, x1_neurons, x0_neurons)
    negW = W < 0
    posW = W >= 0

    W_ldiff = torch.zeros_like(W)
    # assumes the input pixel range is between 0 and 1
    W_ldiff[posW] = WX[posW]
    W_ldiff[negW] = WX[negW] - W[negW]
    W_ldiff = torch.sort(W_ldiff, dim=2, descending=True).values
    lbounds = center - W_ldiff[:, :, :eps].sum(dim=-1, keepdim=True)

    W_udiff = torch.zeros_like(W)
    W_udiff[posW] = W[posW] - WX[posW]
    W_udiff[negW] = - WX[negW]
    W_udiff = torch.sort(W_udiff, dim=2, descending=True).values
    ubounds = center + W_udiff[:, :, :eps].sum(dim=-1, keepdim=True)
    return DeepPoly.Bounds(lbounds, ubounds)

def compute_x0_bounds(x0: torch.Tensor, C: torch.Tensor, eps: int) -> DeepPoly.Bounds:
    batch = C.shape[0]
    d0 = C.shape[1]
    x0 = x0.reshape(batch, -1, 1)
    pos_C = C > 0
    x0_ldiff = torch.where(pos_C, C * x0, torch.zeros_like(C))
    x0_udiff = torch.where(~pos_C, C * (1 - x0), torch.zeros_like(C))
    CPP = (x0_udiff - x0_ldiff)  # <= 0  contribution per pixel
    pixel_idx_set = CPP.sort(dim=1).indices[:, :eps, :]
    mask = torch.ones((batch, d0, 1), dtype=torch.bool).scatter_(1, pixel_idx_set, False)
    # return DeepPoly.Bounds(torch.where(mask, x0, torch.zeros_like(x0)),
    #                        torch.where(mask, x0, torch.ones_like(x0)))  # same as below
    x0_direction = torch.where(pos_C, torch.zeros_like(x0), torch.ones_like(x0))
    x0_ = torch.where(mask, x0, x0_direction)
    return DeepPoly.Bounds(x0_, x0_)

def L0_robust(net: torch.nn.Sequential, x0: torch.Tensor, eps: int,
              true_labels: torch.Tensor, target_label: int, collective=False) -> bool:
    dp_net = DeepPoly.SequentialAT.from_sequential(net, x0)
    dp_head = dp_net.pop(0)  # get the first layer
    assert isinstance(dp_head, DeepPoly.AffineAT)  # the first layer must be affine
    ae1 = dp_head(parent=None)
    ae1.bounds = compute_x1_bounds(x0, dp_head.W, dp_head.b, eps)
    dp_net.append(DeepPoly.AffineAT.Verifier(true_labels, target_label))
    output = dp_net(ae=ae1)

    lrp, urp = output.backsubstitution(require_rp=True)
    output.compute_bounds(lrp, urp, ae1.bounds)
    initial_lb = output.bounds.l.reshape(-1)  # (batch,)
    coeffiencts_x1 = lrp.W.transpose(2, 1)  # (batch, x1_neurons, 1)
    ae1_new_bounds = compute_x1_bounds_with_saliency(x0, dp_head.W, dp_head.b, eps, coeffiencts_x1)
    output.compute_bounds(lrp, urp, ae1_new_bounds)
    sound_lb = output.bounds.l.reshape(-1)

    ae0 = DeepPoly.AbstractElement()
    ae1.parent = ae0
    lrp_wrt_x0, urp_wrt_x0 = output.backsubstitution(require_rp=True)
    coeffiencts_x0 = lrp_wrt_x0.W.transpose(2, 1)  # (batch, x0_neurons, 1)
    ae0.bounds = compute_x0_bounds(x0, coeffiencts_x0, eps)
    output.compute_bounds(lrp_wrt_x0, urp_wrt_x0, ae0.bounds)
    new_lb = output.bounds.l.reshape(-1)

    ae1_ = DeepPoly.AbstractElement(ae1_new_bounds)
    output2 = dp_net(ae1_)
    output2.backsubstitution()
    unsound_lb = output2.bounds.l.reshape(-1)

    print("Initial lower bounds:\t", initial_lb.tolist())
    print("Sound lower bounds:\t", sound_lb.tolist())
    print("New sound lower bounds:\t", new_lb.tolist())
    print("Unsound lower bounds:\t", unsound_lb.tolist())
    
    return all(new_lb > 0)

def get_args():
    parser = argparse.ArgumentParser(
        description="Verufy L0 robustness of a neural network.",
    )
    parser.add_argument("--net", type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ], required=True,
        help="Neural network architecture",
    )
    parser.add_argument("-e", "--epsilon", type=int, required=True, help="L0 norm bound.")
    parser.add_argument("-t", "--target-label", type=int, required=True)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-c", "--collective", action="store_true")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    net = get_network(args.net, "mnist", f"models/mnist_{args.net}.pt").to(DEVICE)
    transform = torchvision.transforms.ToTensor()
    test_dataset = MNIST(root='./data/test', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    for images, labels in test_loader:
        pred_labels = net(images).argmax(1)
        if all(pred_labels == labels) and all(pred_labels != args.target_label):
            L0_robust(net, images, args.epsilon, pred_labels, args.target_label, args.collective)
            exit(0)
        else:
            continue
