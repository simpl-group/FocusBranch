import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from networks import get_network, Normalization
from AbstractInterpreter import SequentialAT, L0BoundedInput, LinfBoundedInput
from typing import Tuple
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_args():
    parser = argparse.ArgumentParser(
        description="Verify L0 robustness of a neural network.",
    )
    parser.add_argument("--net", type=str,
        choices=["fc1", "fc2", "fc3", "fc4", "fc5", "fc6", "fc7", "fcbase",
                 "conv1", "conv2", "conv3", "conv4"], required=True,
        help="Neural network architecture",
    )
    parser.add_argument("-md", "--models-dir", type=str, default="models")
    parser.add_argument("-e", "--epsilon", type=int, default=0, help="L0 norm bound.")
    parser.add_argument("-bs", "--batch-size", type=int, required=True)
    parser.add_argument("-t", "--target-label", type=int, required=True)
    parser.add_argument("-b", "--budget", type=int, default=0)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-m", "--mute", action="store_true", help="Mute gurobi output.")
    parser.add_argument("-si", '--select-indices', type=int, nargs='*', default=[])
    parser.add_argument("-d", "--debug", action="store_true", help="Inspect a specific node.")

    args = parser.parse_args()
    return args

def preprocess(args: argparse.Namespace) -> Tuple[SequentialAT, L0BoundedInput]:
    torch.manual_seed(args.seed)

    net = get_network(args.net, args.models_dir, device=DEVICE)
    net_head = net[0]
    if isinstance(net_head, Normalization):
        net.pop(0)
        mean = net_head.mean
        sigma = net_head.sigma
        L, U = (0 - mean) / sigma, (1 - mean) / sigma
    else:
        L, U = 0, 1

    x_ = torch.zeros(args.batch_size, 1, 28, 28).to(DEVICE)
    dp_net = SequentialAT.from_sequential(net, x_)

    test_dataset = MNIST(root='./data/test', train=False, transform=ToTensor(), download=True)
    indices = [i for i, (_, label) in enumerate(test_dataset) if label != args.target_label]
    filtered_dataset = Subset(test_dataset, indices)
    test_loader = DataLoader(filtered_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    while True:
        images, labels = next(iter(test_loader))
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        pred_labels = net(images).argmax(1)
        if all(pred_labels == labels):  # No need to check target label again
            break
    if mean is not None:
        images = (images - mean) / sigma

    x0 = images.flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
    dp_net.tail().make_marginal(true_labels=pred_labels, target_label=args.target_label)
    bounded_x0 = L0BoundedInput(x0, args.epsilon, (L, U))
    return dp_net, bounded_x0

def assume_and_bound(dp_net: SequentialAT, bounded_x0: L0BoundedInput, depth: int=1):
    output = dp_net(bounded_x0)
    lrp_wrt_x0, _ = output.backsubstitution()
    lb = bounded_x0.compute_bound(lrp_wrt_x0, -1).reshape(-1).tolist()
    if depth == 0:
        print(f"({bounded_x0.num_selected()},{bounded_x0.num_ignored()}):\t", lb)

    if depth > 0 and bounded_x0.eps > 0:
        idx = bounded_x0.most_contributed_idx(lrp_wrt_x0.W)
        # assume idx is selected
        with bounded_x0.selecting(idx):
            assume_and_bound(dp_net=dp_net, bounded_x0=bounded_x0, depth=depth-1)
        # assume idx is not selected
        with bounded_x0.ignoring(idx):
            assume_and_bound(dp_net=dp_net, bounded_x0=bounded_x0, depth=depth-1)

def Linf_robust(dp_net: SequentialAT, bounded_in: LinfBoundedInput, collective=False) -> bool:
    output = dp_net(bounded_in)
    out_bounds = output.get_bounds()
    print("Initial lower bounds:\t", out_bounds.l.reshape(-1).tolist())
    if not collective:
        return all(out_bounds.l > 0)
    if any(out_bounds.l > 0):
        return True
    # future work
    return False