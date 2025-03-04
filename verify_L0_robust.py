import argparse
import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from networks import get_network, Normalization
from AbstractInterpreter import SequentialAT, L0BoundedInput, LinfBoundedInput
from solver_check import check_unsat

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def L0_robust(dp_net: SequentialAT, bounded_in: L0BoundedInput) -> bool:
    output = dp_net(bounded_in)
    lrp_wrt_x0, _ = output.backsubstitution()
    new_lb = bounded_x0.compute_bounds(lrp_wrt_x0, lrp_wrt_x0)

    print("x0 with ltrigger:\t", new_lb.l.reshape(-1).tolist())
    print("x0 with utrigger:\t", new_lb.u.reshape(-1).tolist())

    if any(new_lb.l > 0):
        print("Collectively Safe")
        return True
    if (new_lb.u > 0).sum() < 2:
        print("UNK")
        return False
    coeffiencts_x0 = lrp_wrt_x0.W.transpose(2, 1)
    return check_unsat(bounded_in.x.squeeze(2).tolist(), coeffiencts_x0.squeeze(2).tolist(),
                       lrp_wrt_x0.b.flatten().tolist(), bounded_in.eps, (bounded_in.L, bounded_in.U))

def L0_samplewise_lowerbound(dp_net: SequentialAT, bounded_in: L0BoundedInput, depth: int=1):
    output = dp_net(bounded_in)
    lrp_wrt_x0, _ = output.backsubstitution()
    lb = bounded_x0.compute_bound(lrp_wrt_x0, -1).reshape(-1).tolist()
    if depth == 0:
        print(f"({bounded_in.num_selected()},{bounded_in.num_ignored()}):\t", lb)

    if depth > 0 and bounded_in.eps > 0:
        idx = bounded_x0.most_contributed_idx(lrp_wrt_x0.W)
        # assume idx is selected
        with bounded_in.selecting(idx):
            L0_samplewise_lowerbound(dp_net=dp_net, bounded_in=bounded_in, depth=depth-1)
        # assume idx is not selected
        with bounded_in.ignoring(idx):
            L0_samplewise_lowerbound(dp_net=dp_net, bounded_in=bounded_in, depth=depth-1)

def Linf_robust(dp_net: SequentialAT, bounded_in: LinfBoundedInput, collective=False) -> bool:
    output = dp_net(bounded_in)
    out_bounds = output.get_bounds()
    print("Initial lower bounds:\t", out_bounds.l.reshape(-1).tolist())
    if not collective:
        return all(out_bounds.l > 0)
    if any(out_bounds.l > 0):
        return True
    return False

def get_args():
    parser = argparse.ArgumentParser(
        description="Verify L0 robustness of a neural network.",
    )
    parser.add_argument("--net", type=str,
        choices=["fc1", "fc2", "fc3", "fc4", "fc5", "fc6", "fc7", "fcbase",
                 "conv1", "conv2", "conv3", "conv4"], required=True,
        help="Neural network architecture",
    )
    parser.add_argument("-0e", "--l0-epsilon", type=int, default=0, help="L0 norm bound.")
    parser.add_argument("-infe", "--linf-epsilon", type=float, default=0, help="Linf norm bound.")
    parser.add_argument("-t", "--target-label", type=int, required=True)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-c", "--collective", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-d", "--depth", type=int, default=0)
    parser.add_argument("-md", "--models-dir", type=str, default="models")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    
    net = get_network(args.net, args.models_dir, device=DEVICE)
    net_head = net[0]
    mean, sigma = None, None
    if isinstance(net_head, Normalization):
        net.pop(0)
        mean, sigma = net_head.mean, net_head.sigma
        L, U = (0 - mean) / sigma, (1 - mean) / sigma
    else:
        L, U = 0, 1
    
    x_ = torch.zeros(args.batch_size, 1, 28, 28).to(DEVICE)
    dp_net = SequentialAT.from_sequential(net, x_)

    test_dataset = MNIST(root='./data/test', train=False, transform=ToTensor(), download=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if args.l0_epsilon > 0:
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            pred_labels = net(images).argmax(1)
            if all(pred_labels == labels) and all(pred_labels != args.target_label):
                if mean is not None:
                    images = (images - mean) / sigma
                x0 = images.flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
                dp_net[-1].make_marginal(true_labels=pred_labels, target_label=args.target_label)
                bounded_x0 = L0BoundedInput(x0, args.l0_epsilon, (L, U))
                if args.collective:
                    L0_robust(dp_net, bounded_x0)
                else:
                    L0_samplewise_lowerbound(dp_net, bounded_x0, depth=args.depth)
                exit(0)
            else:
                continue
    elif args.linf_epsilon > 0:
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            pred_labels = net(images).argmax(1)
            if all(pred_labels == labels) and all(pred_labels != args.target_label):
                if mean is not None:
                    images = (images - mean) / sigma
                x0 = images.flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
                dp_net[-1].make_marginal(true_labels=pred_labels, target_label=args.target_label)
                bounded_x0 = LinfBoundedInput(x0, args.linf_epsilon, (L, U))
                Linf_robust(dp_net, bounded_x0, args.collective)
                exit(0)
            else:
                continue