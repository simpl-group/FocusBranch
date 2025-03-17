import time
from solver_check import solver_check_unsat_all
from AbstractInterpreter import L0BoundedInput, SequentialAT
from typing import List, Set
from utils import get_args, preprocess, split_combs, tqdm_bar_format
from tqdm import tqdm

# BFS
def check_and_branch(combs_to_check: Set[frozenset], safe_combs: List[frozenset]) -> bool:
    global splits, level, solver_time_temp, solver_checks_temp
    combs_to_split: List[frozenset] = []
    safe_num, unk_num = 0, 0
    solver_time_temp = .0
    solver_checks_temp = 0
    with tqdm(total=len(combs_to_check), disable=(level == 0), desc=f"Level {level}", bar_format=tqdm_bar_format) as pbar:
        for comb in combs_to_check:
            if check_all(comb):
                safe_num += 1
                safe_combs.append(comb)
            elif splits + unk_num < args.budget:
                # verification failed, need to split the comb
                unk_num += 1
                combs_to_split.append(comb)
            else: return False
            pbar.set_postfix({"Safe:UNK": f"{safe_num}:{unk_num}", "GRB": f"{solver_checks_temp}({solver_time_temp:.2f}s)"})
            pbar.update(1)
    if unk_num == 0: return True
    if level == bounded_x0.eps: return False
    level += 1
    next_level_combs = split_combs(combs_to_split, safe_combs)
    splits += unk_num
    return check_and_branch(next_level_combs, safe_combs)

def check_all(comb: frozenset, debug=False, mute=True) -> bool:
    global total_checks, solver_checks, solver_checks_temp, solver_time, x0_, solver_time_temp, args
    total_checks += 1
    with bounded_x0.selecting(comb):
        output = dp_net(bounded_x0)
        lrp, _ = output.backsubstitution()
        lb_range = bounded_x0.compute_bounds(lrp, lrp)

    if debug and not mute:
        ...

    # pre-check
    if ((lb_range.l > 0).sum() / lb_range.l.numel()) > args.threshold: return True
    if ((lb_range.u > 0).sum() / lb_range.u.numel()) < args.threshold: return False

    solver_checks_temp += 1
    solver_checks += 1
    t0 = time.time()
    ret = solver_check_unsat_all(x0=x0_, coefficients=lrp.W.squeeze(1).cpu().numpy(),
            bias=lrp.b.reshape(-1).cpu().numpy(), eps=bounded_x0.eps, selected_idx=comb,
            pixel_value_range=(bounded_x0.L, bounded_x0.U), mute=mute, threshold=args.threshold)
    solver_time += time.time() - t0
    solver_time_temp += time.time() - t0
    return ret

if __name__ == "__main__":
    args = get_args()
    net, data_loader, L, U = preprocess(args, demo=True)
    # todo: dp_net list and bounded_x0 list, each with MAX_BATCH_SIZE
    images, pred_labels = next(iter(data_loader))
    dp_net = SequentialAT.from_sequential(net, images)
    x0 = images.flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
    dp_net.tail().make_marginal(true_labels=pred_labels, target_label=args.target_label)
    bounded_x0 = L0BoundedInput(x0, args.epsilon, (L, U))
    x0_ = bounded_x0.x.squeeze(2).cpu().numpy()
    print("Preprocessing done")

    total_checks, solver_checks, solver_checks_temp = 0, 0, 0
    safe_combs: List[frozenset] = []
    splits = 0
    solver_time, solver_time_temp = .0, .0
    level = 0
    start = time.time()

    ret = check_and_branch(set([frozenset()]), safe_combs)
    print(f"Total: Splits {splits}; Time: {time.time() - start:.2f}; Nodes: {total_checks}; GRB Checks: {solver_checks}({solver_time:.2f}s)")
    print(f"Collectively Safe: {ret}")
