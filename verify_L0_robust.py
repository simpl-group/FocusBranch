from time import perf_counter
import torch
from solver_check import gurobi_check_unsat
from AbstractInterpreter import L0BoundedInput, SequentialAT
from bimm import combs2bitmasks
from utils import *
from split_and_prune import split_and_prune
from tqdm import tqdm
import signal


def check_and_branch(
    seqAT: SequentialAT,
    bounded_x0: L0BoundedInput,
    demo: bool=False,
) -> bool:
    combs_to_check = torch.empty((1, 0), dtype=torch.int64)
    num_indices = bounded_x0.num_pixels
    bitmask_length = (num_indices + 63) // 64
    safe_bitmasks: torch.Tensor = torch.empty((0, bitmask_length), dtype=torch.long)
    level = 0
    if demo:
        global splits, total_checks, branching_time, solver_checks, solver_time
    while level <= bounded_x0.eps:
        num_to_check = combs_to_check.shape[0]
        if num_to_check == 0: return True
        unk_combs_lst: list[torch.Tensor]  = []
        new_safe_bitmasks_lst: list[torch.Tensor] = [safe_bitmasks]
        if demo:
            safe_cnt, unk_cnt = 0, 0
            solver_time_temp, solver_checks_temp = 0., 0
        t0 = perf_counter()
        pbar = tqdm(total=num_to_check, desc=f"Level {level}", bar_format=TQDM_BAR_FORMAT, disable=not demo)
        with seqAT.working_on(DEVICE):
            for i in range(0, num_to_check):
                comb = combs_to_check[i].to(device=DEVICE)  # (1, level)

                with bounded_x0.selecting(comb):
                    seqAT.update_activation_rp(bounded_in=bounded_x0)
                    lrp = seqAT.tail.backsubstitution_lrp()
                    lb_range = bounded_x0.compute_bounds(lrp, lrp)

                if (lb_range.l > 0).any():
                    is_safe = True
                elif (lb_range.u > 0).sum() < 2:
                    is_safe = False
                else:
                    solver_checks_temp += 1
                    coef = lrp.W.squeeze(1).cpu().numpy()
                    bias = lrp.b.reshape(-1).cpu().numpy()
                    if (lb_range.u < 0).any():
                        keep_indices = (lb_range.u > 0).reshape(-1).cpu().numpy()
                        x0 = x0_[keep_indices]
                        coef = coef[keep_indices]
                        bias = bias[keep_indices]
                    else: x0 = x0_
                    t0 = perf_counter()
                    is_safe = gurobi_check_unsat(
                        x0=x0,
                        coefficients=coef,
                        bias=bias,
                        eps=bounded_x0.eps,
                        selected_idx=comb.cpu().numpy(),
                        pixel_value_range=(bounded_x0.L, bounded_x0.U),
                        mute=True
                    )
                    solver_time_temp += perf_counter() - t0

                if is_safe:
                    new_safe_bitmasks_lst.append(combs2bitmasks(comb.unsqueeze(0), bitmask_length).cpu())
                    safe_cnt += 1
                else:
                    unk_combs_lst.append(comb.cpu())
                    unk_cnt += 1

                if demo:
                    pbar.set_postfix({"Safe:UNK": f"{safe_cnt}:{unk_cnt}",
                        "GRB": f"{solver_checks_temp}({solver_time_temp:.2f}s)"})
                    pbar.update(1)
        pbar.close()

        if len(unk_combs_lst) == 0: return True
        if level == bounded_x0.eps: return False
        unk_combs = torch.vstack(unk_combs_lst)
        safe_bitmasks = torch.cat(new_safe_bitmasks_lst, dim=0)
        del combs_to_check
        torch.cuda.empty_cache()
        t0 = perf_counter()
        combs_to_check = split_and_prune(unk_combs, safe_bitmasks, level, num_indices, demo)
        if demo:
            branching_time += perf_counter() - t0
            total_checks += num_to_check
            splits += len(unk_combs)
            solver_checks += solver_checks_temp
            solver_time += solver_time_temp
        torch.cuda.empty_cache()
        level += 1
    return False


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, timeout_handler)
    parser = argparse.ArgumentParser(
        description="A demonstration of verifying the robustness of a neural network on a batch of images."
    )
    add_shared_arguments(parser)
    parser.add_argument("-m", "--mute", action="store_true", help="Mute gurobi output.")
    parser.add_argument("-si", '--select-indices', type=int, nargs='*', default=[])
    parser.add_argument("-d", "--debug", action="store_true", help="Inspect a specific node.")
    args = parser.parse_args()
    assert args.target_label is not None, "Target label must be specified."
    if args.demo:
        total_checks, solver_checks, solver_checks_temp = 0, 0, 0
        splits = 0
        solver_time, solver_time_temp = .0, .0
        branching_time = .0

    net, seqAT, bounded_x0, data_loader = get4(args)

    total_time = 0.
    num_verified = 0
    net.cpu()
    pbar = tqdm(total=args.test_batches, bar_format=TQDM_BAR_FORMAT, disable=args.demo)
    for images, pred_labels in data_loader:
        assert net(images).argmax(dim=1).eq(pred_labels).all()
        bounded_x0.x = images.to(DEVICE).flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
        x0_ = images.flatten(start_dim=1).numpy()
        seqAT.tail.make_marginal(true_labels=pred_labels, target_label=args.target_label, device=DEVICE)

        if args.debug:
            if len(args.select_indices) > 0:
                print(f"Checking a specific combination of indices {args.select_indices}")
            comb = torch.Tensor(args.select_indices).to(device=DEVICE, dtype=torch.long)
            with bounded_x0.selecting(comb):
                seqAT.update_activation_rp(bounded_in=bounded_x0)
                lrp = seqAT.tail.backsubstitution_lrp()
                lb_range = bounded_x0.compute_bounds(lrp, lrp)

            lb_l = lb_range.l.reshape(-1).cpu().numpy()
            lb_u = lb_range.u.reshape(-1).cpu().numpy()
            print("x0 with ltrigger:\t", lb_l)
            print("x0 with utrigger:\t", lb_u)
            if (lb_range.l > 0).any():
                is_safe = True
            elif (lb_range.u > 0).sum() < 2:
                is_safe = False
            else:
                is_safe = gurobi_check_unsat(
                    x0=x0_,
                    coefficients=lrp.W.squeeze(1).cpu().numpy(),
                    bias=lrp.b.reshape(-1).cpu().numpy(),
                    eps=bounded_x0.eps,
                    selected_idx=comb.cpu().numpy(),
                    pixel_value_range=(bounded_x0.L, bounded_x0.U),
                    mute=args.mute
                )
            print("Safe:", is_safe)
            exit(0)

        success = False
        elapsed_time = 0.
        try:
            signal.alarm(args.timeout)
            t0 = perf_counter()
            if check_and_branch(
                seqAT,
                bounded_x0,
                demo=args.demo
            ):
                elapsed_time = perf_counter() - t0
                total_time += elapsed_time
                success = True
                num_verified += 1
            signal.alarm(0)
        except TimeoutException:
            if args.demo: print("Timeout")
        except torch.OutOfMemoryError as e:
            if args.demo: raise(e)
        finally:
            bounded_x0.clear_cache()
            torch.cuda.empty_cache()
        if args.demo:
            print(f"Total: Splits {splits}; Nodes: {total_checks}; Time: {perf_counter() - t0:.2f}s; GRB Checks: {solver_checks}({solver_time:.2f}s)")
            print(f"Safe: {success}")
            break

        pbar.update(1)
        mean_time = total_time / num_verified if num_verified > 0 else 0
        pbar.set_postfix({"Verified": f"{num_verified}/{pbar.n}|{100*num_verified/pbar.n:.2f}%",
                            "Avg":f"{mean_time:.2f}s/img"})
    pbar.close()
