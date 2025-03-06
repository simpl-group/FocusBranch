import argparse
import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from networks import get_network, Normalization
from AbstractInterpreterGRB import SequentialAT, L0BoundedInput
import time
import gurobipy as gp
from gurobipy import GRB
from utils import get_args

DEVICE = "cpu"

if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)

    net = get_network(args.net, args.models_dir, device=DEVICE)
    net_head = net[0]
    mean, sigma = None, None
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
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    while True:
        images, labels = next(iter(test_loader))
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        pred_labels = net(images).argmax(1)
        if all(pred_labels == labels) and all(pred_labels != args.target_label):
            break
    if mean is not None:
        images = (images - mean) / sigma

    x0 = images.flatten(start_dim=1).unsqueeze(2).numpy()  # (batch, d1, 1)
    dp_net[-1].make_marginal(true_labels=pred_labels.numpy(), target_label=args.target_label)
    bounded_x0 = L0BoundedInput(x0, args.epsilon, gp.Model("idea2"), (L, U))
    output = dp_net(bounded_x0)
    bounded_x0.gmodel.setObjective(output.get_bounds().l.reshape(-1).item(), GRB.MINIMIZE)

    bounded_x0.gmodel.optimize()

    if bounded_x0.gmodel.Status == GRB.Status.OPTIMAL:
        print("Solution found:")
        for idx in range(bounded_x0.num_features):
            if bounded_x0.locs[idx].X > 0.5:  # If idx is within the trigger
                print(f"{idx}")
    else:
        print(f"Solver status: {bounded_x0.gmodel.Status}")