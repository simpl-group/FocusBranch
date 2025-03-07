import time
from solver_check import solver_check_unsat
from typing import List, Set
from utils import get_args, preprocess
from tqdm import tqdm

def get_next_level_branches(combs_to_split: List[frozenset]) -> Set[frozenset]:
    global splits, safe_combs
    combs_to_check: Set[frozenset] = set()
    for comb in combs_to_split:
        splits += 1
        for idx in range(bounded_x0.num_features):
            if idx in comb: continue
            new_comb = frozenset(comb | {idx})
            if any(safe_comb.issubset(new_comb) for safe_comb in safe_combs):
                continue  # pruning
            combs_to_check.add(new_comb)
    return combs_to_check

# BFS
def check_and_branch(combs_to_check: Set[frozenset]) -> bool:
    global safe_combs, budget_used, level
    combs_to_split = []
    safe_num, unk_num = 0, 0
    level = level + 1
    combs_to_check_tqdm = tqdm(combs_to_check)
    for comb in combs_to_check_tqdm:
        if check(comb):
            safe_num += 1
            safe_combs.append(comb)
        elif budget_used < args.budget and len(comb) < bounded_x0.eps:
            # verification failed, need to split the comb
            unk_num += 1
            budget_used += 1
            combs_to_split.append(comb)
        else: return False
        combs_to_check_tqdm.set_description(debug_message(safe_num, unk_num))
    if unk_num == 0: return True
    next_level_branches = get_next_level_branches(combs_to_split)
    return check_and_branch(next_level_branches)

def check(comb: frozenset, debug=False, mute=True) -> None:
    global total_checks, solver_checks, solver_time_used
    total_checks += 1
    with bounded_x0.selecting(comb):
        output = dp_net(bounded_x0)
        lrp, _ = output.backsubstitution()
        lb_range = bounded_x0.compute_bounds(lrp, lrp)

    lb_l = lb_range.l.reshape(-1).cpu().numpy()
    lb_u = lb_range.u.reshape(-1).cpu().numpy()
    if debug:
        print("x0 with ltrigger:\t", lb_l)
        print("x0 with utrigger:\t", lb_u)

    # pre-check
    if (lb_l > 0).any(): return True
    if (lb_u > 0).sum() < 2: return False

    solver_checks += 1
    x0 = bounded_x0.x.squeeze(2).cpu().numpy()
    coef = lrp.W.squeeze(1).cpu().numpy()
    bias = lrp.b.reshape(-1).cpu().numpy()
    if (lb_u < 0).any():
        keep_indices = lb_u > 0
        x0 = x0[keep_indices]
        coef = coef[keep_indices]
        bias = bias[keep_indices]
    start = time.time()
    ret = solver_check_unsat(x0=x0, coefficients=coef,
            bias=bias, eps=bounded_x0.eps, selected_idx=comb,
            pixel_value_range=(bounded_x0.L, bounded_x0.U), mute=mute)
    solver_time_used += time.time() - start
    return ret

def debug_message(safe_num: int, unk_num: int) -> str:
    global level, splits, solver_time_used, start, solver_checks, total_checks
    return f"Level {level}: Splits {splits}; Time {solver_time_used:.2f}/{time.time() - start:.2f}; \
Safe:UNK {safe_num}:{unk_num}; Checks: {solver_checks}/{total_checks}"

if __name__ == "__main__":
    args = get_args()
    dp_net, bounded_x0 = preprocess(args)
    print("Preprocessing done")

    total_checks, solver_checks = 0, 0
    if args.debug:
        comb = frozenset(args.select_indices)
        ret = check(comb, debug=True, mute=args.mute)
        print(f"Collectively Safe: {ret}; Checks: {solver_checks}/{total_checks}")
        exit(0)

    budget_used = 0
    safe_combs: List[frozenset] = []
    splits = 0
    solver_time_used = .0
    level = -1
    start = time.time()

    ret = check_and_branch(set([frozenset()]))
    print(f"Collectively Safe: {ret}")
