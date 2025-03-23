import time
from typing import List
from utils import split_combs, get_args, preprocess, tqdm_bar_format, DEVICE
import torch
from AbstractInterpreter import L0BoundedInput, SequentialAT
from tqdm import tqdm
import torch
MAX_BATCH_SIZE = 25000

# BFS
def check_and_branch_batch(combs_to_check: List[List[int]], safe_combs: List[List[int]]) -> bool:
    num_to_check = len(combs_to_check)
    if num_to_check == 0: return True
    global level, branching_time, total_nodes, batch_size, splits
    combs_to_split = []
    min_value = float("inf")
    min_comb = None

    combs_to_check_tensor = torch.tensor(combs_to_check, dtype=torch.long)
    del combs_to_check
    with tqdm(total=num_to_check, desc=f"Level {level}", bar_format=tqdm_bar_format) as pbar:
        for i in range(0, num_to_check, batch_size):
            combs_to_check_batch_tensor = combs_to_check_tensor[i:i + batch_size]

            current_batch_size = len(combs_to_check_batch_tensor)
            if bounded_x0.batch_size != current_batch_size:
                bounded_x0.expand(current_batch_size)
                seqAT.expand(current_batch_size)

            with bounded_x0.selecting(combs_to_check_batch_tensor.to(DEVICE)):
                output = seqAT(bounded_x0)
                lrp, _ = output.backsubstitution()
                lb_l = bounded_x0.compute_bound(lrp, sign=-1).reshape(-1).cpu()

            safe_combs += combs_to_check_batch_tensor[lb_l >= 0].tolist()
            combs_to_split += combs_to_check_batch_tensor[lb_l < 0].tolist()
            if lb_l.min() < min_value:
                min_value = lb_l.min().item()
                min_comb = combs_to_check_batch_tensor[lb_l.argmin()]
            pbar.set_postfix({"Safe:UNK": f"{len(safe_combs)}:{len(combs_to_split)}", "Min": f"{min_value:.2f}", "Comb": min_comb.tolist()})
            pbar.update(current_batch_size)

    total_nodes += num_to_check
    if len(combs_to_split) == 0: return True
    if level == bounded_x0.eps: return False
    level += 1
    del combs_to_check_batch_tensor, output, lrp, lb_l, min_comb, min_value
    torch.cuda.empty_cache()
    t0 = time.time()
    next_level_branches = split_combs(combs_to_split, safe_combs)
    branching_time += time.time() - t0
    splits += len(combs_to_split)
    del combs_to_split
    torch.cuda.empty_cache()
    return check_and_branch_batch(next_level_branches, safe_combs)

if __name__ == "__main__":
    args = get_args()
    batch_size = args.batch_size if args.batch_size < MAX_BATCH_SIZE else MAX_BATCH_SIZE
    args.batch_size = 1
    net, data_loader, L, U = preprocess(args, demo=True)
    x_ = torch.zeros(1, 1, 28, 28).to(DEVICE)
    seqAT = SequentialAT.from_sequential(net, x_)
    for images, pred_labels in data_loader:
        x0 = images.flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
        seqAT.tail().make_marginal(true_labels=pred_labels, target_label=args.target_label)
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
