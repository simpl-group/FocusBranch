import time
from typing import List
from utils import split_combs, get_args, preprocess, tqdm_bar_format, DEVICE
import torch
from AbstractInterpreter1 import L0BoundedInput, SequentialAT
from tqdm import tqdm
import torch
MAX_BATCH_SIZE = 25000

# BFS
def check_and_branch_batch(combs_to_check: List[List[int]], safe_combs: List[List[int]]) -> bool:
    num_to_check = len(combs_to_check)
    if num_to_check == 0: return True
    global level, branching_time, total_nodes, batch_size, splits
    combs_to_split = []
    safe_num, unk_num = 0, 0

    combs_to_check_tail = combs_to_check[-(num_to_check % batch_size):]
    combs_to_check_tail_tensor = torch.tensor(combs_to_check_tail).to(torch.long).to(DEVICE)
    tail_length = len(combs_to_check_tail_tensor)
    bounded_x0.expand(tail_length)
    dp_net.expand(tail_length)
    with bounded_x0.selecting(combs_to_check_tail_tensor):
        output = dp_net(bounded_x0)
        lrp, _ = output.backsubstitution()
        lb_l = bounded_x0.compute_bound(lrp, sign=-1).reshape(-1)

    min_value = lb_l.min().item()
    min_comb = combs_to_check_tail_tensor[lb_l.argmin()]
    unk_indices = lb_l < 0
    safe_indices = lb_l >= 0
    unk_num += unk_indices.sum().item()
    safe_num += safe_indices.sum().item()
    combs_to_split += combs_to_check_tail_tensor[unk_indices].tolist()
    safe_combs += combs_to_check_tail_tensor[safe_indices].tolist()

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
                    lb_l = bounded_x0.compute_bound(lrp, sign=-1).reshape(-1)

                unk_indices = lb_l < 0
                safe_indices = lb_l >= 0
                unk_num += unk_indices.sum().item()
                safe_num += safe_indices.sum().item()
                combs_to_split += combs[unk_indices].tolist()
                safe_combs += combs[safe_indices].tolist()
                if lb_l.min() < min_value:
                    min_value = lb_l.min().item()
                    min_comb = combs[lb_l.argmin()]
                pbar.set_postfix({"Safe:UNK": f"{safe_num}:{unk_num}", "Min": f"{min_value:.2f}", "Comb": min_comb.tolist()})
                pbar.update(batch_size)
    else:
        print(f"Level {level} complete; Nodes: {num_to_check}; Safe:UNK={safe_num}:{unk_num}; Min: {min_value:.2f}; Comb: {min_comb.tolist()}")
    total_nodes += num_to_check

    if unk_num == 0: return True
    if level == bounded_x0.eps: return False
    level += 1
    t0 = time.time()
    next_level_branches = split_combs(combs_to_split, safe_combs)
    splits += unk_num
    branching_time += time.time() - t0
    return check_and_branch_batch(next_level_branches, safe_combs)

if __name__ == "__main__":
    args = get_args()
    batch_size = args.batch_size if args.batch_size < MAX_BATCH_SIZE else MAX_BATCH_SIZE
    args.batch_size = 1
    net, data_loader, L, U = preprocess(args, demo=True)
    x_ = torch.zeros(1, 1, 28, 28).to(DEVICE)
    dp_net = SequentialAT.from_sequential(net, x_)
    for images, pred_labels in data_loader:
        x0 = images.flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
        dp_net.tail().make_marginal(true_labels=pred_labels, target_label=args.target_label)
        bounded_x0 = L0BoundedInput(x0, args.epsilon, (L, U))
        break
    del net
    print("Preprocessing done")

    safe_combs: List[List[int]] = []
    splits = 0
    total_nodes = 0
    branching_time = .0
    level = 0
    start = time.time()

    ret = check_and_branch_batch([[]], safe_combs)
    print(f"Total: Splits {splits}; Nodes {total_nodes}; Time {time.time() - start:.2f}s; Branching Time {branching_time:.2f}s")
    print(f"Collectively Safe: {ret}")
