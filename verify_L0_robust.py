import time
from solver_check import solver_check_unsat
from AbstractInterpreter import L0BoundedInput
from typing import List, Set
from utils import get_args, preprocess
from tqdm import tqdm

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
def check_and_branch(combs_to_check: Set[frozenset]) -> bool:
    global safe_combs, splits, level, solver_time_temp, solver_checks_temp
    combs_to_split = []
    safe_num, unk_num = 0, 0
    num_to_check = len(combs_to_check)
    solver_time_temp = .0
    solver_checks_temp = 0
    with tqdm(total=num_to_check, disable=(level == 0), desc=f"Level {level}", bar_format=tqdm_bar_format) as pbar:
        for comb in combs_to_check:
            if check(comb):
                safe_num += 1
                safe_combs.append(comb)
            elif splits + unk_num < args.budget and level < bounded_x0.eps:
                # verification failed, need to split the comb
                unk_num += 1
                combs_to_split.append(comb)
            else: return False
            pbar.set_postfix({"Safe:UNK": f"{safe_num}:{unk_num}", "GRB": f"{solver_checks_temp}({solver_time_temp:.2f}s)"})
            pbar.update(1)
    if unk_num == 0: return True
    next_level_branches = get_next_level_branches(combs_to_split)
    return check_and_branch(next_level_branches)

def check(comb: frozenset, debug=False, mute=True) -> bool:
    global total_checks, solver_checks, solver_checks_temp, solver_time, x0_, solver_time_temp
    total_checks += 1
    with bounded_x0.selecting(comb):
        output = dp_net(bounded_x0)
        lrp, _ = output.backsubstitution()
        lb_range = bounded_x0.compute_bounds(lrp, lrp)

    if debug and not mute:
        lb_l = lb_range.l.reshape(-1).cpu().numpy()
        lb_u = lb_range.u.reshape(-1).cpu().numpy()
        print("x0 with ltrigger:\t", lb_l)
        print("x0 with utrigger:\t", lb_u)

    # pre-check
    if (lb_range.l > 0).any(): return True
    if (lb_range.u > 0).sum() < 2: return False

    solver_checks_temp += 1
    solver_checks += 1
    coef = lrp.W.squeeze(1).cpu().numpy()
    bias = lrp.b.reshape(-1).cpu().numpy()
    if (lb_range.u < 0).any():
        keep_indices = (lb_range.u > 0).reshape(-1).cpu().numpy()
        x0 = x0_[keep_indices]
        coef = coef[keep_indices]
        bias = bias[keep_indices]
    else: x0 = x0_

    t0 = time.time()
    ret = solver_check_unsat(x0=x0, coefficients=coef,
            bias=bias, eps=bounded_x0.eps, selected_idx=comb,
            pixel_value_range=(bounded_x0.L, bounded_x0.U), mute=mute)
    solver_time += time.time() - t0
    solver_time_temp += time.time() - t0
    return ret

if __name__ == "__main__":
    args = get_args()
    dp_net, data_loader, L, U = preprocess(args, demo=True)
    for images, pred_labels in data_loader:
        x0 = images.flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
        dp_net.tail().make_marginal(true_labels=pred_labels, target_label=args.target_label)
        bounded_x0 = L0BoundedInput(x0, args.epsilon, (L, U))
        break
    x0_ = bounded_x0.x.squeeze(2).cpu().numpy()
    print("Preprocessing done")

    total_checks, solver_checks, solver_checks_temp = 0, 0, 0
    safe_combs: List[frozenset] = []
    splits = 0
    solver_time, solver_time_temp = .0, .0
    level = 0
    start = time.time()

    if args.debug:
        comb = frozenset(args.select_indices)
        ret = check(comb, debug=True, mute=args.mute)
        print(f"Collectively Safe: {ret}; Checks: {total_checks}({solver_checks})")
        exit(0)

    ret = check_and_branch(set([frozenset()]))
    print(f"Total: Splits {splits}; Time: {time.time() - start:.2f}; Nodes: {total_checks}; GRB Checks: {solver_checks}({solver_time:.2f}s)")
    print(f"Collectively Safe: {ret}")
