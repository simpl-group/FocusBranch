import argparse
import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from networks import get_network, Normalization
import DeepPoly

DEVICE = "cpu"

def compute_x1_diffs(x0: torch.Tensor, W: torch.Tensor, eps: int, pixel_value_range=(0, 1)) -> DeepPoly.Bounds:
    batch = x0.shape[0]
    # W.shape[-2] number of nuerons in x1
    X = x0.reshape(batch, 1, -1).expand(-1, W.shape[-2], -1)
    L, U = pixel_value_range
    WX_L = W * (X - L)  # (batch, d1, d0)
    WX_U = W * (U - X)

    WX_ldiff = torch.where(W > 0, -WX_L, WX_U)
    WX_ldiff_sorted = WX_ldiff.sort(dim=2, descending=False).values
    ldiff = WX_ldiff_sorted[:, :, :eps].sum(dim=-1, keepdim=True)

    WX_udiff = torch.where(W < 0, -WX_L, WX_U)
    WX_udiff_sorted = WX_udiff.sort(dim=2, descending=True).values
    udiff = WX_udiff_sorted[:, :, :eps].sum(dim=-1, keepdim=True)
    return DeepPoly.Bounds(ldiff, udiff)

def x0_with_ltrigger(x0: torch.Tensor, C: torch.Tensor, eps: int, pixel_value_range=(0, 1)) -> torch.Tensor:
    batch, d0, _ = x0.shape
    L, U = pixel_value_range
    CPP = torch.where(C > 0, -C * (x0 - L), C * (U - x0))  # <= 0  contribution per pixel
    pixel_idx_set = CPP.sort(dim=1).indices[:, :eps, :]
    mask = torch.ones((batch, d0, 1), dtype=torch.bool).scatter_(1, pixel_idx_set, False)
    x0_direction = torch.where(C > 0, L, U)
    return torch.where(mask, x0, x0_direction)

def x0_with_utrigger(x0: torch.Tensor, C: torch.Tensor, eps: int, pixel_value_range=(0, 1)) -> torch.Tensor:
    batch, d0, _ = x0.shape
    L, U = pixel_value_range
    CPP = torch.where(C < 0, -C * (x0 - L), C * (U - x0))  # >= 0  contribution per pixel
    pixel_idx_set = CPP.sort(dim=1, descending=True).indices[:, :eps, :]
    mask = torch.ones((batch, d0, 1), dtype=torch.bool).scatter_(1, pixel_idx_set, False)
    x0_direction = torch.where(C < 0, L, U)  # pos->U; neg->L
    return torch.where(mask, x0, x0_direction)

def compute_x0_diffs(x0: torch.Tensor, C: torch.Tensor, eps: int, pixel_value_range=(0, 1)) -> DeepPoly.Bounds:
    L, U = pixel_value_range
    CPP_l = torch.where(C > 0, -C * (x0 - L), C * (U - x0))  # <= 0  contribution per pixel
    CPP_u = torch.where(C < 0, -C * (x0 - L), C * (U - x0))  # >= 0  contribution per pixel
    ldiff = CPP_l.sort(dim=1, descending=False).values[:, :eps, :].sum(dim=1, keepdim=True)
    udiff = CPP_u.sort(dim=1, descending=True).values[:, :eps, :].sum(dim=1, keepdim=True)
    return DeepPoly.Bounds(ldiff, udiff)

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
    dp_net = DeepPoly.SequentialAT.from_sequential(net, x0, true_labels, target_label)
    dp_head = dp_net.pop(0)  # get the first layer
    assert isinstance(dp_head, DeepPoly.AffineAT)  # the first layer must be affine
    x0_ = x0.flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
    x1_center = torch.bmm(dp_head.W, x0_) + dp_head.b  # (batch, d1, 1)
    init_x1_bounds = compute_x1_diffs(x0_, dp_head.W, eps, (L, U)) + x1_center
    ae1 = DeepPoly.AbstractElement.from_input(init_bounds=init_x1_bounds)
    output = dp_net(ae=ae1)

    initial_lb = output.get_bounds().l.reshape(-1)  # (batch,)

    ae1.parent = dp_head(ae=None)
    lrp_wrt_x0, _ = output.backsubstitution()
    coeffiencts_x0 = lrp_wrt_x0.W.transpose(2, 1)  # (batch, d0, 1)
    x0_center = torch.bmm(lrp_wrt_x0.W, x0_) + lrp_wrt_x0.b
    new_lb = compute_x0_diffs(x0_, coeffiencts_x0, eps, (L, U)) + x0_center

    print("Initial lower bounds:\t", initial_lb.tolist())
    print("x0 with ltrigger:\t", new_lb.l.reshape(-1).tolist())
    print("x0 with utrigger:\t", new_lb.u.reshape(-1).tolist())

    if not collective:
        return all(new_lb.l > 0)
    if any(new_lb.l > 0):
        print("Collectively Safe")
        return True
    if (new_lb.u > 0).sum() < 2:
        print("UNK")
        return False
    return check_unsat(x0_.squeeze(2).tolist(), coeffiencts_x0.squeeze(2).tolist(),
                       lrp_wrt_x0.b.flatten().tolist(), eps, (L, U))

def check_unsat(x0: list, coeffiencts_x0: list, b: list, eps: int, pixel_value_range=(0, 1)) -> bool:
    print("UNK")
    return False

def Linf_robust(net: torch.nn.Sequential, x0: torch.Tensor, eps: int,
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
    x0 = x0.reshape(x0.shape[0], -1, 1)
    dp_net = DeepPoly.SequentialAT.from_sequential(net, x0, true_labels, target_label)
    ae = DeepPoly.AbstractElement.from_input(init_bounds=DeepPoly.Bounds((x0-eps).clamp(L, U), (x0+eps).clamp(L, U)))
    output = dp_net(ae=ae)
    out_bounds = output.get_bounds()
    print("Initial lower bounds:\t", out_bounds.l.reshape(-1).tolist())
    return all(out_bounds.l > 0)

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
            # Linf_robust(net, images, 0.01, pred_labels, args.target_label, args.collective)
            exit(0)
        else:
            continue
