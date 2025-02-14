import argparse
import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from networks import get_network, Normalization
import DeepPoly

DEVICE = "cpu"

def compute_x1_bounds(x0: torch.Tensor, W: torch.Tensor, b: torch.Tensor, eps: int, pixel_value_range=(0, 1)) -> DeepPoly.Bounds:
    batch = x0.shape[0]
    x0 = x0.reshape(batch, -1, 1)
    center = torch.bmm(W, x0) + b  # (batch, d1, 1)
    # W.shape[-2] number of nuerons in x1
    X = x0.reshape(batch, 1, -1).expand(-1, W.shape[-2], -1)
    L, U = pixel_value_range
    WX_L = W * (X - L)  # (batch, d1, d0)
    WX_U = W * (U - X)
    negW = W < 0
    posW = W >= 0

    W_ldiff = torch.zeros_like(W)
    W_ldiff[posW] = -WX_L[posW]
    W_ldiff[negW] = WX_U[negW]
    W_ldiff = torch.sort(W_ldiff, dim=2, descending=False).values
    lbounds = center + W_ldiff[:, :, :eps].sum(dim=-1, keepdim=True)

    W_udiff = torch.zeros_like(W)
    W_udiff[posW] = WX_U[posW]
    W_udiff[negW] = -WX_L[negW]
    W_udiff = torch.sort(W_udiff, dim=2, descending=True).values
    ubounds = center + W_udiff[:, :, :eps].sum(dim=-1, keepdim=True)
    return DeepPoly.Bounds(lbounds, ubounds)

def compute_x0_bounds(x0: torch.Tensor, C: torch.Tensor, eps: int, pixel_value_range=(0, 1)) -> DeepPoly.Bounds:
    batch = C.shape[0]
    d0 = C.shape[1]
    x0 = x0.reshape(batch, -1, 1)
    pos_C = C > 0
    L, U = pixel_value_range
    x0_ldiff = torch.where(pos_C, C * (x0 - L), torch.zeros_like(C))
    x0_udiff = torch.where(~pos_C, C * (U - x0), torch.zeros_like(C))
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
    net_head = net[0]
    if isinstance(net_head, Normalization):
        net.pop(0)
        mean = net_head.mean
        sigma = net_head.sigma
        x0 = (x0 - mean) / sigma
        L, U = (0 - mean) / sigma, (1 - mean) / sigma
    else:
        L, U = 0, 1
    dp_net = DeepPoly.SequentialAT.from_sequential(net, x0)
    dp_head = dp_net.pop(0)  # get the first layer
    assert isinstance(dp_head, DeepPoly.AffineAT)  # the first layer must be affine
    ae1 = dp_head(parent=None)
    ae1.bounds = compute_x1_bounds(x0, dp_head.W, dp_head.b, eps, (L, U))
    dp_net.append(DeepPoly.AffineAT.Marginal(true_labels, target_label))
    output = dp_net(ae=ae1)

    lrp, urp = output.backsubstitution(compute_bounds=False)
    initial_bounds = DeepPoly.AbstractElement.compute_bounds(lrp, urp, ae1.bounds)
    initial_lb = initial_bounds.l.reshape(-1)  # (batch,)

    ae0 = DeepPoly.AbstractElement()
    ae1.parent = ae0
    lrp_wrt_x0, urp_wrt_x0 = output.backsubstitution(compute_bounds=False)
    coeffiencts_x0 = lrp_wrt_x0.W.transpose(2, 1)  # (batch, d0, 1)
    ae0.bounds = compute_x0_bounds(x0, coeffiencts_x0, eps, (L, U))
    new_bounds = DeepPoly.AbstractElement.compute_bounds(lrp_wrt_x0, urp_wrt_x0, ae0.bounds)
    new_lb = new_bounds.l.reshape(-1)

    print("Initial lower bounds:\t", initial_lb.tolist())
    print("New sound lower bounds:\t", new_lb.tolist())
    
    return all(new_lb > 0)

def get_args():
    parser = argparse.ArgumentParser(
        description="Verufy L0 robustness of a neural network.",
    )
    parser.add_argument("--net", type=str,
        choices=[
            "fc1",
            "fc2",
            "fc3",
            "fc4",
            "fc5",
            "fc6",
            "fc7",
            "conv1",
            "conv2",
            "conv3",
            "conv4",
        ], required=True,
        help="Neural network architecture",
    )
    parser.add_argument("-e", "--epsilon", type=int, required=True, help="L0 norm bound.")
    parser.add_argument("-t", "--target-label", type=int, required=True)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-c", "--collective", action="store_true")
    parser.add_argument("-s", "--shuffle", action="store_true")
    parser.add_argument("-d", "--models-dir", type=str, default="models")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    net = get_network(args.net, args.models_dir)
    transform = torchvision.transforms.ToTensor()
    test_dataset = MNIST(root='./data/test', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=2)
    for images, labels in test_loader:
        pred_labels = net(images).argmax(1)
        if all(pred_labels == labels) and all(pred_labels != args.target_label):
            L0_robust(net, images, args.epsilon, pred_labels, args.target_label, args.collective)
            exit(0)
        else:
            continue
