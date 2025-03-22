import torch
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset, TensorDataset
from networks import get_network, Normalization
from AbstractInterpreter import SequentialAT, L0BoundedInput, LinfBoundedInput
from torch.nn import Sequential
from typing import Iterable, Tuple, List
import numpy as np
from tqdm import tqdm
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tqdm_bar_format = "{desc}: {n_fmt}/{total_fmt} |{percentage:3.2f}% {elapsed}<{remaining} {postfix} {rate_fmt}"
num_features = 28 * 28

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

def preprocess(args: argparse.Namespace, demo: bool=True) -> Tuple[Sequential, DataLoader, float, float]:
    """Prepares data and network for verification."""
    torch.manual_seed(args.seed)

    net = get_network(args.net, args.models_dir, device=DEVICE)
    net_head = net[0]
    if isinstance(net_head, Normalization):
        net.pop(0)
        mean, sigma = net_head.mean, net_head.sigma
    else:
        mean, sigma = 0, 1

    L, U = (0 - mean) / sigma, (1 - mean) / sigma

    # Filter and collect correctly classified samples
    test_dataset = MNIST(root='./data/test', train=False, transform=Compose([ToTensor(), Normalize(mean, sigma)]), download=True)
    non_targeted_indices = (test_dataset.targets != args.target_label).nonzero().squeeze(1)
    non_targeted_subset = Subset(test_dataset, non_targeted_indices)

    correct_images = []
    correct_labels = []
    batch_size = min(256, len(non_targeted_subset))
    test_loader = DataLoader(non_targeted_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
            batch_preds = net(batch_images).argmax(1)
            correct_idx = (batch_preds == batch_labels).nonzero().squeeze(1)

            correct_images.append(batch_images[correct_idx])
            correct_labels.append(batch_preds[correct_idx])

            if demo and sum(len(imgs) for imgs in correct_images) >= args.batch_size:
                break

    correct_dataset = TensorDataset(torch.cat(correct_images), torch.cat(correct_labels))
    correct_loader = DataLoader(correct_dataset, batch_size=args.batch_size, shuffle=True)

    return net, correct_loader, L, U

def abstraction_loader(dp_net: SequentialAT, data_loader: DataLoader,
                       L: float, U: float, target_label: int, epsilon: int
                       ) -> Iterable[Tuple[SequentialAT, L0BoundedInput]]:
    for images, pred_labels in data_loader:
        x0 = images.flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
        dp_net.tail().make_marginal(true_labels=pred_labels, target_label=target_label)
        bounded_x0 = L0BoundedInput(x0, epsilon, (L, U))
        yield dp_net, bounded_x0

def mask_tensor(comb: List[int]) -> torch.Tensor:
    tensor = torch.zeros(num_features, dtype=torch.bool, device=DEVICE)
    if comb: tensor[comb] = True
    return tensor

chunk_size = 1000  # Adjust based on GPU memory availability
def split_combs(combs_to_split: List[List[int]], safe_combs: List[List[int]]) -> List[List[int]]:
    # Convert combs_to_split to a tensor
    c2s_tensor = torch.stack([mask_tensor(c) for c in combs_to_split], dim=0).to(DEVICE)

    if len(combs_to_split) < len(safe_combs): return split_combs_aux(c2s_tensor)

    # Convert safe_combs to a tensor and precompute sums
    if safe_combs:
        safe_combs_tensor = torch.stack([mask_tensor(s) for s in safe_combs], dim=0).to(DEVICE)
    else:
        safe_combs_tensor = torch.empty((0, num_features), dtype=torch.bool, device=DEVICE)
    safe_comb_sums = safe_combs_tensor.sum(dim=1, dtype=torch.float32)

    combs_to_check_tensor_list = []

    for i in tqdm(range(num_features), desc="Processing features", disable=(len(safe_combs) == 0), bar_format=tqdm_bar_format):
        # Get combs that don't include feature i (i-th column is False)
        c2s_i = c2s_tensor[~c2s_tensor[:, i]]
        num_new_icombs = c2s_i.shape[0]
        if num_new_icombs == 0: continue

        # Generate new_combs by setting i-th column to True
        new_icombs = c2s_i.clone()
        new_icombs[:, i] = True

        # Process in chunks to manage memory
        for j in range(0, num_new_icombs, chunk_size):
            new_icombs_jchunk = new_icombs[j:j + chunk_size]
            any_subset = torch.zeros(new_icombs_jchunk.shape[0], dtype=torch.bool, device=DEVICE)

            for k in range(0, safe_combs_tensor.shape[0], chunk_size):
                safe_combs_tensor_kchunk = safe_combs_tensor[k:k + chunk_size]
                safe_comb_sums_kchunk = safe_comb_sums[k:k + chunk_size]

                # check subset condition by comparing matrix product and comb_sum
                product = new_icombs_jchunk.float() @ safe_combs_tensor_kchunk.float().T
                any_subset |= (product == safe_comb_sums_kchunk.unsqueeze(0)).any(dim=1)

            # Filter out new_combs that are pruned
            valid_new_combs = new_icombs_jchunk[~any_subset]
            if valid_new_combs.shape[0] > 0:
                combs_to_check_tensor_list.append(valid_new_combs)

    # Concatenate all valid new combs and deduplicate
    if combs_to_check_tensor_list:
        combs_to_check_tensor = torch.cat(combs_to_check_tensor_list, dim=0).unique(dim=0)
    else:
        combs_to_check_tensor = torch.empty((0, num_features), dtype=torch.bool, device=DEVICE)

    # Convert tensor to List[List[int]]
    combs_to_check = []
    for comb_tensor in combs_to_check_tensor.cpu().numpy():
        indices = np.where(comb_tensor)[0].astype(int).tolist()
        combs_to_check.append(indices)

    return combs_to_check

def split_combs_aux(c2s_tensor: torch.Tensor) -> List[List[int]]:
    combs_to_check_tensor_list = []
    base_level = c2s_tensor[0].sum().item()

    for i in tqdm(range(num_features), desc="Processing features", bar_format=tqdm_bar_format):
        # Get combs that don't include feature i (i-th column is False)
        c2s_i = c2s_tensor[~c2s_tensor[:, i]]
        num_new_icombs = c2s_i.shape[0]
        if num_new_icombs == 0: continue

        # Generate new_combs by setting i-th column to True
        new_icombs = c2s_i.clone()
        new_icombs[:, i] = True

        for j in range(0, num_new_icombs, chunk_size):
            new_icombs_jchunk = new_icombs[j:j + chunk_size]
            superset_cnt = torch.zeros(new_icombs_jchunk.shape[0], dtype=torch.uint8, device=DEVICE)

            for k in range(0, c2s_tensor.shape[0], chunk_size):
                c2s_tensor_kchunk = c2s_tensor[k:k + chunk_size]

                # compute superset_cnt by sum of matrix product == base_level
                product = new_icombs_jchunk.float() @ c2s_tensor_kchunk.float().T
                superset_cnt += (product == base_level).sum(dim=1)

            # Filter out new_combs that are pruned
            valid_new_combs = new_icombs_jchunk[superset_cnt == base_level + 1]
            if valid_new_combs.shape[0] > 0:
                combs_to_check_tensor_list.append(valid_new_combs)

    # Concatenate all valid new combs and deduplicate
    if combs_to_check_tensor_list:
        combs_to_check_tensor = torch.cat(combs_to_check_tensor_list, dim=0).unique(dim=0)
    else:
        combs_to_check_tensor = torch.empty((0, num_features), dtype=torch.bool, device=DEVICE)

    # Convert tensor to List[List[int]]
    combs_to_check = []
    for comb_tensor in combs_to_check_tensor.cpu().numpy():
        indices = np.where(comb_tensor)[0].astype(int).tolist()
        combs_to_check.append(indices)

    return combs_to_check

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
