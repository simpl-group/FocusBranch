import time
from typing import List
from utils import get_args, preprocess, tqdm_bar_format, DEVICE, get_pure_network, TimeoutException
import torch
from AbstractInterpreter import L0BoundedInput, SequentialAT
from tqdm import tqdm
import torch
from split_and_prune import split_and_prune
from signal import alarm
MAX_BATCH_SIZE = 25000

# BFS
def check_and_branch_batch(seqAT: SequentialAT, bounded_x0: L0BoundedInput, batch_size: int=2000, demo: bool=False) -> bool:
    combs_to_check = torch.empty((1, 0), dtype=torch.int64)
    safe_combs: List[List[int]] = []
    level = 0
    if demo:
        global branching_time, total_nodes, splits
    while level <= bounded_x0.eps:
        num_to_check = combs_to_check.shape[0]
        if num_to_check == 0: return True
        unk_combs = []
        if demo:
            min_value = float("inf")
            min_comb = None

        with tqdm(total=num_to_check, desc=f"Level {level}", bar_format=tqdm_bar_format, disable=not demo) as pbar:
            for i in range(0, num_to_check, batch_size):
                combs_to_check_batch = combs_to_check[i:i + batch_size]

                current_batch_size = len(combs_to_check_batch)
                if bounded_x0.batch_size != current_batch_size:
                    bounded_x0.expand(current_batch_size)

                with bounded_x0.selecting(combs_to_check_batch):
                    seqAT.update_activation_rp(bounded_in=bounded_x0)
                    lrp = seqAT.tail.backsubstitution_lrp()
                    lb_l = bounded_x0.compute_lbound(lrp).reshape(-1).cpu()

                safe_combs += combs_to_check_batch[lb_l >= 0].tolist()
                unk_combs += combs_to_check_batch[lb_l < 0].tolist()
                if demo:
                    if lb_l.min() < min_value:
                        min_value = lb_l.min().item()
                        min_comb = combs_to_check_batch[lb_l.argmin()]
                    pbar.set_postfix({"Safe:UNK": f"{len(safe_combs)}:{len(unk_combs)}", "Min": f"{min_value:.2f}", "Comb": min_comb.tolist()})
                    pbar.update(current_batch_size)

        if len(unk_combs) == 0: return True
        del combs_to_check, combs_to_check_batch, lb_l, lrp
        torch.cuda.empty_cache()
        t0 = time.time()
        combs_to_check = split_and_prune(unk_combs, safe_combs, level, demo)
        if demo:
            branching_time += time.time() - t0
            total_nodes += num_to_check
            splits += len(unk_combs)
        del unk_combs
        torch.cuda.empty_cache()
        level += 1
    return False

if __name__ == "__main__":
    args = get_args()
    batch_size = args.batch_size if args.batch_size < MAX_BATCH_SIZE else MAX_BATCH_SIZE
    args.batch_size = 1
    net, mean, sigma = get_pure_network(args)
    L, U = (0 - mean) / sigma, (1 - mean) / sigma
    data_loader = preprocess(args, net, mean, sigma, demo=True)
    x_ = torch.zeros(1, 1, 28, 28).to(DEVICE)
    seqAT = SequentialAT.from_sequential(net, x_)
    for images, pred_labels in data_loader:
        images = images.to(DEVICE)
        x0 = images.flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
        seqAT.tail.make_marginal(true_labels=pred_labels, target_label=args.target_label, device=DEVICE)
        bounded_x0 = L0BoundedInput(x0, args.epsilon, (L, U))
        break
    del net

    if not args.mute:
        print("Preprocessing done")
        splits = 0
        total_nodes = 0
        branching_time = .0
    start = time.time()

    ret = False
    try:
        alarm(args.timeout)
        ret = check_and_branch_batch(seqAT, bounded_x0, batch_size, demo=not args.mute)
        alarm(0)
    except TimeoutException:
        print("Timeout")

    if not args.mute:
        print(f"Total: Splits {splits}; Nodes {total_nodes}; Time {time.time() - start:.2f}s; Branching Time {branching_time:.2f}s")
    else:
        print(f"Time {time.time() - start:.2f}s")
    print(f"Safe: {ret}")
