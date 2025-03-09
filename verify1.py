import time
from typing import List, Set
from utils import get_args, DEVICE
import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from networks import get_network, Normalization
from AbstractInterpreter1 import SequentialAT, L0BoundedInput
from tqdm import tqdm
import torch
MAX_BATCH_SIZE = 25000

tqdm_bar_format = "{desc}: {n_fmt}/{total_fmt} |{percentage:3.2f}% {elapsed}<{remaining} {postfix} {rate_fmt}"

def get_next_level_branches(combs_to_split: List[frozenset]) -> Set[frozenset]:
    global splits, safe_combs, level
    combs_to_check: Set[frozenset] = set()
    for comb in tqdm(combs_to_split, disable=(len(safe_combs) == 0), desc="Splitting", bar_format=tqdm_bar_format):
        for idx in range(bounded_x0.num_features):
            if idx in comb: continue
            new_comb = frozenset(comb | {idx})
            if any(safe_comb.issubset(new_comb) for safe_comb in safe_combs):
                continue  # pruning
            combs_to_check.add(new_comb)
    level = level + 1
    splits += len(combs_to_split)
    return combs_to_check

# BFS
def check_and_branch_batch(combs_to_check: Set[frozenset]) -> bool:
    num_to_check = len(combs_to_check)
    if num_to_check == 0: return True
    global safe_combs, level, branching_time, total_nodes, batch_size
    safe_combs_temp, combs_to_split = [], []
    safe_num, unk_num = 0, 0

    combs_to_check = [list(comb) for comb in combs_to_check] # [[int]]

    combs_to_check_tail = combs_to_check[-(num_to_check % batch_size):]
    combs_to_check_tail_tensor = torch.tensor(combs_to_check_tail).to(torch.long).to(DEVICE)
    tail_length = len(combs_to_check_tail_tensor)
    bounded_x0.expand(tail_length)
    dp_net.expand(tail_length)
    with bounded_x0.selecting(combs_to_check_tail_tensor):
        output = dp_net(bounded_x0)
        lrp, _ = output.backsubstitution()
        lb_range = bounded_x0.compute_bounds(lrp, lrp)

    unk_indices = lb_range.l.reshape(-1) < 0
    safe_indices = lb_range.l.reshape(-1) >= 0
    unk_num += unk_indices.sum().item()
    safe_num += safe_indices.sum().item()
    combs_to_split += combs_to_check_tail_tensor[unk_indices].tolist()
    safe_combs_temp += combs_to_check_tail_tensor[safe_indices].tolist()

    if tail_length != num_to_check:
        bounded_x0.expand(batch_size)
        dp_net.expand(batch_size)
        combs_to_check_head = [combs_to_check[i:i + batch_size]
            for i in range(0, num_to_check - batch_size, batch_size)]
        combs_to_check_head_tensor = torch.tensor(combs_to_check_head).to(torch.long).to(DEVICE)
        with tqdm(total=num_to_check, desc=f"Level {level}", bar_format=tqdm_bar_format, initial=tail_length) as pbar:
            for combs in combs_to_check_head_tensor:
                with bounded_x0.selecting(combs):
                    output = dp_net(bounded_x0)
                    lrp, _ = output.backsubstitution()
                    lb_range = bounded_x0.compute_bounds(lrp, lrp)

                unk_indices = lb_range.l.reshape(-1) < 0
                safe_indices = lb_range.l.reshape(-1) >= 0
                unk_num += unk_indices.sum().item()
                safe_num += safe_indices.sum().item()
                combs_to_split += combs[unk_indices].tolist()
                safe_combs_temp += combs[safe_indices].tolist()
                pbar.set_postfix({"Safe:UNK": f"{safe_num}:{unk_num}"})
                pbar.update(batch_size)
    else:
        print(f"Level {level} complete; Nodes: {num_to_check}; Safe:UNK={safe_num}:{unk_num}")
    total_nodes += num_to_check

    if unk_num == 0: return True
    if level == bounded_x0.eps: return False
    safe_combs += [frozenset(comb) for comb in safe_combs_temp]
    combs_to_split = [frozenset(comb) for comb in combs_to_split]
    t0 = time.time()
    next_level_branches = get_next_level_branches(combs_to_split)
    branching_time += time.time() - t0
    return check_and_branch_batch(next_level_branches)

if __name__ == "__main__":
    args = get_args()
    batch_size = args.batch_size if args.batch_size < MAX_BATCH_SIZE else MAX_BATCH_SIZE
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

    x_ = torch.zeros(1, 1, 28, 28).to(DEVICE)
    dp_net = SequentialAT.from_sequential(net, x_)

    test_dataset = MNIST(root='./data/test', train=False, transform=ToTensor(), download=True)
    indices = [i for i, (_, label) in enumerate(test_dataset) if label != args.target_label]
    filtered_dataset = Subset(test_dataset, indices)
    test_loader = DataLoader(filtered_dataset, batch_size=1, shuffle=True, num_workers=2)

    while True:
        images, labels = next(iter(test_loader))
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        pred_labels = net(images).argmax(1)
        if all(pred_labels == labels):
            break
    if mean is not None:
        images = (images - mean) / sigma

    x0 = images.flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
    dp_net.tail().make_marginal(true_labels=pred_labels.item(), target_label=args.target_label)
    bounded_x0 = L0BoundedInput(x0, args.epsilon, (L, U))

    print("Preprocessing done")

    safe_combs: List[frozenset] = []
    splits = 0
    total_nodes = 0
    branching_time = .0
    level = 0
    start = time.time()

    ret = check_and_branch_batch(set([frozenset()]))
    print(f"Total: Splits {splits}; Nodes: {total_nodes}; Time: {time.time() - start:.2f}; Branching Time: {branching_time:.2f}")
    print(f"Collectively Safe: {ret}")
