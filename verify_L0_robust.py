import argparse
import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from networks import get_network
import DeepPoly

DEVICE = "cpu"

def compute_x1_bounds_with_saliency(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor,
                                    eps: int, lsaliency: torch.Tensor, collactive=False):
    assert lsaliency.ndim == 3 # (batch, x1_neurons, 1)
    batch = W.shape[0]
    x1_n = W.shape[1]
    x0_n = W.shape[2]

    x = x.reshape(batch, -1, 1)
    center = torch.bmm(W, x) + b  # (batch, x1_neurons, 1)
    # (batch, x1_neurons, x0_neurons)
    X = x.reshape(batch, 1, -1).expand(batch, x1_n, x0_n)
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
    per_nueron_contribution = torch.where(lsaliency > 0, W_ldiff * lsaliency, W_udiff * -lsaliency)
    if collactive:
        collactive_contribution = per_nueron_contribution.sum(1).sum(0)
        collactive_pixel_idx_set = collactive_contribution.sort(descending=True).indices[:eps]
        pixel_idx_set = collactive_pixel_idx_set.unsqueeze(0)
    else:
        contribution = per_nueron_contribution.sum(1)
        pixel_idx_set = contribution.sort(descending=True).indices[:, :eps]
    pixel_idx_set = pixel_idx_set.unsqueeze(1).expand(batch, x1_n, eps)

    lbounds = center - W_ldiff.gather(dim=2, index=pixel_idx_set).sum(dim=-1, keepdim=True)
    ubounds = center + W_udiff.gather(dim=2, index=pixel_idx_set).sum(dim=-1, keepdim=True)
    return DeepPoly.Bounds(lbounds, ubounds)

def compute_x1_bounds(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, eps: int):
    batch = x.shape[0]
    x = x.reshape(batch, -1, 1)
    center = torch.bmm(W, x) + b  # (batch, x1_neurons, 1)
    # W.shape[-2] number of nuerons in x1
    X = x.reshape(batch, 1, -1).expand(-1, W.shape[-2], -1)
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

def L0_robust(net: torch.nn.Sequential, x0: torch.Tensor, eps: int,
              true_labels: torch.Tensor, target_label: int, collective=False) -> bool:
    dp_net = DeepPoly.SequentialAT.from_sequential(net, x0)
    dp_head = dp_net.pop(0)
    assert isinstance(dp_head, DeepPoly.AffineAT)  # get the first layer to be affine
    dp_net.append(DeepPoly.AffineAT.Verifier(true_labels, target_label))
    bounds1 = compute_x1_bounds(x0, dp_head.W, dp_head.b, eps)
    ae = DeepPoly.AbstractElement(bounds1)

    output = dp_net(ae)
    lrp, _ = output.backsubstitution(require_rp=True)
    initial_lb = output.bounds.l.reshape(-1)  # (batch,)
    lsaliency = lrp.W.transpose(2, 1)  # (batch, x1_neurons, 1)
    bounds2 = compute_x1_bounds_with_saliency(x0, dp_head.W, dp_head.b, eps, lsaliency, False)
    ae2 = DeepPoly.AbstractElement(bounds2)
    output2 = dp_net(ae2)
    output2.backsubstitution()
    new_lb = output2.bounds.l.reshape(-1)
    
    bounds3 = compute_x1_bounds_with_saliency(x0, dp_head.W, dp_head.b, eps, lsaliency, True)
    ae3 = DeepPoly.AbstractElement(bounds3)
    output3 = dp_net(ae3)
    output3.backsubstitution()
    coll_lb = output3.bounds.l.reshape(-1)
    print(initial_lb)
    print(new_lb)
    print(coll_lb)
    assert all(coll_lb >= new_lb)
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
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    for images, labels in test_loader:
        pred_labels = net(images).argmax(1)
        if all(pred_labels == labels) and all(pred_labels != args.target_label):
            L0_robust(net, images, args.epsilon, pred_labels, args.target_label, args.collective)
            exit(0)
        else:
            continue
