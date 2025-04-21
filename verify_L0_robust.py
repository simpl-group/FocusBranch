import time
import torch
from solver_check import solver_check_unsat
from AbstractInterpreter import L0BoundedInput, SequentialAT
from typing import List
from utils import get_args, get_pure_network, tqdm_bar_format, DEVICE, TimeoutException, get_mnist_loader
from split_and_prune import split_and_prune
from tqdm import tqdm
from signal import alarm

def check_and_branch(combs_to_check: torch.Tensor, safe_combs: List[List[int]]) -> bool:
    global splits, level, solver_time_temp, solver_checks_temp
    unk_combs: List[List[int]] = []
    unk_num = 0
    solver_time_temp = .0
    solver_checks_temp = 0
    watchout = safe_combs is not []
    with tqdm(total=len(combs_to_check), disable=(level == 0), desc=f"Level {level}", bar_format=tqdm_bar_format) as pbar:
        for comb in combs_to_check:
            if check(comb):
                safe_combs.append(comb.tolist())
            else:
                # verification failed, need to split the comb
                unk_num += 1
                unk_combs.append(comb.tolist())
            pbar.set_postfix({"Safe:UNK": f"{len(safe_combs)}:{len(unk_combs)}", "GRB": f"{solver_checks_temp}({solver_time_temp:.2f}s)"})
            pbar.update(1)
    if unk_num <= level: return True
    if level == bounded_x0.eps: return False
    del combs_to_check
    torch.cuda.empty_cache()
    next_level_combs = split_and_prune(unk_combs, safe_combs, level, True, watchout)
    level += 1
    splits += unk_num
    torch.cuda.empty_cache()
    return check_and_branch(next_level_combs, safe_combs)

def check(comb: torch.Tensor, debug=False, mute=True) -> bool:
    global total_checks, solver_checks, solver_checks_temp, solver_time, x0_, solver_time_temp
    total_checks += 1
    with bounded_x0.selecting(comb):
        seqAT.update_activation_rp(bounded_in=bounded_x0)
        lrp = seqAT.tail.backsubstitution_lrp()
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
            bias=bias, eps=bounded_x0.eps, selected_idx=comb.tolist(),
            pixel_value_range=(bounded_x0.L, bounded_x0.U), mute=mute)
    solver_time += time.time() - t0
    solver_time_temp += time.time() - t0
    return ret

if __name__ == "__main__":
    args = get_args()
    assert args.target_label is not None, "Target label must be specified."
    net, mean, sigma = get_pure_network(args)
    L, U = (0 - mean) / sigma, (1 - mean) / sigma
    # data_loader = preprocess(args, net, mean, sigma, demo=True)
    data_loader = get_mnist_loader(args)
    x_ = torch.zeros(min(args.batch_size, len(data_loader)), 1, 28, 28).to(DEVICE)
    seqAT = SequentialAT.from_sequential(net, x_)
    i = 0
    for images, pred_labels in data_loader:
        i += 1
        if i <= args.seed: continue
        images = images.to(DEVICE)
        x0 = images.flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
        seqAT.tail.make_marginal(true_labels=pred_labels, target_label=args.target_label, device=DEVICE)
        bounded_x0 = L0BoundedInput(x0, args.epsilon, (L, U))
        break
    x0_ = bounded_x0.x.squeeze(2).cpu().numpy()
    del net
    print("Preprocessing done")

    total_checks, solver_checks, solver_checks_temp = 0, 0, 0
    safe_combs: List[List[int]] = []
    root_combs = torch.empty((1, 0), dtype=torch.int64)
    splits = 0
    solver_time, solver_time_temp = .0, .0
    level = 0
    start = time.time()

    if args.debug:
        comb = frozenset(args.select_indices)
        ret = check(comb, debug=True, mute=args.mute)
        print(f"Collectively Safe: {ret}; Checks: {total_checks}({solver_checks})")
        exit(0)

    ret = False
    try:
        alarm(args.timeout)
        ret = check_and_branch(root_combs, safe_combs)
        alarm(0)
    except TimeoutException:
        print("Timeout")

    print(f"Total: Splits {splits}; Time: {time.time() - start:.2f}; Nodes: {total_checks}; GRB Checks: {solver_checks}({solver_time:.2f}s)")
    print(f"Collectively Safe: {ret}")
