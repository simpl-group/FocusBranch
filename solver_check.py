from AbstractInterpreter import SequentialAT, L0BoundedInput, ActivationAT, AffineAT, ReLUAT
from utils import get4, add_shared_arguments
from utils.timeout import timeout_handler, TimeoutException
from utils.constant import TQDM_BAR_FORMAT
import argparse
import signal
import warnings
from time import perf_counter
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
from tqdm import tqdm
from json import dump

env = gp.Env(params={"OutputFlag": 0, "SolutionLimit": 1})


def gurobi_check_unsat(
    x0: np.ndarray,
    coefficients: np.ndarray,
    bias: np.ndarray,
    eps: int,
    selected_idx: np.ndarray,
    pixel_value_range=(0, 1),
    mute=False
) -> bool:
    num_batch, num_pixel = x0.shape
    L, U = pixel_value_range

    model = gp.Model("check_unsat", env=env)

    controllers = model.addMVar((num_pixel,), vtype=GRB.BINARY, name="c")
    triggers = model.addMVar((num_pixel,), vtype=GRB.CONTINUOUS, lb=L, ub=U, name="t")

    # l0=eps constraint
    model.addConstr(controllers.sum() == eps)
    if len(selected_idx) > 0:
        assert len(selected_idx) <= eps
        model.addConstr(controllers[selected_idx] == 1)

    for i in range(num_batch):
        img = controllers * triggers + (1 - controllers) * x0[i]
        model.addConstr(coefficients[i] @ img + bias[i] <= 0)

    model.optimize()

    if model.Status == GRB.Status.INFEASIBLE: return True
    if mute: return False
    if model.Status == GRB.Status.OPTIMAL:
        print("Solution found:")
        for idx in range(num_pixel):
            if controllers[idx].X > 0.5:  # If idx is within the trigger
                print(f"{idx}: {triggers[idx].X}")
    else:
        print(f"Solver status: {model.Status}")
    return False


def gurobi_complete_check(seqAT: SequentialAT, bounded_x0: L0BoundedInput,
                          pre_activation_bounds_list: list[tuple[np.ndarray, np.ndarray]],
                          demo: bool = True, ground_truth_label: int | None = None, ) -> bool:
    model = gp.Model("check_unsat", env=env)
    img = bounded_x0.x[0, ..., 0].numpy()  # input: [784]
    delta = model.addMVar(img.shape, vtype=GRB.CONTINUOUS, name="delta")  # delta: [784]
    for i in range(img.shape[0]):
        model.addConstr(delta[i] <= bounded_x0.U - img[i])
        model.addConstr(bounded_x0.L - img[i] <= delta[i])
    x = img + delta  # input

    epsilon = bounded_x0.eps
    l0_loss = model.addVar(lb=epsilon, ub=epsilon)
    model.addGenConstrNorm(l0_loss, delta, 0.0)

    # forward
    relu_layer_idx = 0
    for layer_idx, AT in enumerate(seqAT.ATs):
        if isinstance(AT, AffineAT):  # FC, ConV2D
            weight, bias = AT.rp.W[0, ...].numpy(), AT.rp.b[0, ..., 0].numpy()
            x = weight @ x + bias
        elif isinstance(AT, ReLUAT):  # ReLU
            lbound, ubound = pre_activation_bounds_list[relu_layer_idx]

            y = model.addMVar(x.shape, vtype=GRB.CONTINUOUS, name=f"ReLU{layer_idx}_y")
            for idx, (lb, ub) in enumerate(zip(lbound, ubound)):
                if ub <= 0:
                    # Case 1: ubound <= 0
                    model.addConstr(y[idx] == 0)
                elif lb >= 0:
                    # Case 2: lbound >= 0
                    model.addConstr(y[idx] == x[idx])
                else:
                    # Case 3: lbound <= 0 <= ubound
                    a = model.addVar(vtype=GRB.BINARY)  # ReLU auxiliary var
                    model.addConstr(y[idx] <= x[idx] - lb * (1 - a))
                    model.addConstr(y[idx] >= x[idx])
                    model.addConstr(y[idx] <= ub * a)
                    model.addConstr(y[idx] >= 0)

            x = y
            relu_layer_idx += 1
        else:
            raise NotImplementedError(type(AT))

    if ground_truth_label is None:
        model.addConstr(x[0] <= 0)
    else:
        any_not_safe = [model.addVar(vtype=GRB.BINARY) for _ in range(10)]
        for i in range(10):
            if i == ground_truth_label:
                model.addConstr(any_not_safe[i] == 0)
                continue
            model.addGenConstrIndicator(any_not_safe[i], True, x[i] <= 0)

        res = model.addVar(vtype=GRB.BINARY, lb=1, ub=1)  # True constant
        model.addGenConstrOr(res, any_not_safe)

    model.optimize()

    if model.Status == GRB.Status.INFEASIBLE:
        return True
    if model.Status in {GRB.Status.OPTIMAL, GRB.Status.SOLUTION_LIMIT}:
        if demo:
            print(f"Solution found")
            delta_values = np.asarray(delta.X)
            xs = np.nonzero(delta_values)
            print(f"Delta indices: {xs}, value: {delta_values[xs]}")
            # check model input
            print(img[xs])
            print(img[xs] + delta.X[xs])
    elif model.Status == GRB.Status.INF_OR_UNBD:  # unsolvable
        warnings.warn("Model is infeasible or unbounded")
    else:
        warnings.warn(f"Solver status: {model.Status}")
    return False


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, timeout_handler)
    parser = argparse.ArgumentParser(
        description="Complete verification of L0 robustness using Gurobi."
    )
    add_shared_arguments(parser)
    args = parser.parse_args()
    if args.demo:
        args.test_batches = 1

    net, seqAT, bounded_x0, data_loader = get4(args)

    total_time = 0.
    num_verified = 0
    record: list[tuple[bool, float]] = []
    net.cpu()
    pbar = tqdm(total=args.test_batches, bar_format=TQDM_BAR_FORMAT, disable=args.demo)
    for images, pred_labels in data_loader:
        assert net(images).argmax(dim=1).eq(pred_labels).all()
        bounded_x0.x = images.flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
        if args.target_label is not None:
            seqAT.tail.make_marginal(true_labels=pred_labels, target_label=args.target_label, device="cpu")
        else:
            seqAT.tail.make_marginals(true_labels=pred_labels, device="cpu")

        pre_activation_bounds_list = []
        for AT in seqAT.ATs:
            if isinstance(AT, ActivationAT):
                lrp_wrt_inputs, urp_wrt_inputs = AT.parent.backsubstitution()
                pre_activation_bounds = bounded_x0.compute_bounds(lrp_wrt_inputs, urp_wrt_inputs)
                AT.lrp, AT.urp = AT.compute_rp(pre_activation_bounds)
                # convert to np.ndarray
                # delete batch dimension
                pre_activation_bounds_list.append((
                    pre_activation_bounds.l[0, ..., 0].numpy(),  # lower bound
                    pre_activation_bounds.u[0, ..., 0].numpy(),  # upper bound
                ))

        timeout = True
        safe = False
        elapsed_time = 0.
        try:
            signal.alarm(args.timeout)
            t0 = perf_counter()
            safe = gurobi_complete_check(seqAT, bounded_x0, pre_activation_bounds_list, demo=args.demo,
                                         ground_truth_label=pred_labels.item() if args.target_label is None else None)
            timeout = False
            elapsed_time = perf_counter() - t0
            total_time += elapsed_time
            num_verified += 1
            signal.alarm(0)
        except TimeoutException:
            if args.demo: print("Timeout")
        except torch.OutOfMemoryError as e:
            if args.demo: print(e)
        finally:
            bounded_x0.clear_cache()

        if args.demo:
            print(f"Safe: {safe}")
            break

        pbar.update(1)
        mean_time = total_time / num_verified if num_verified > 0 else 0
        pbar.set_postfix({"Verified": f"{num_verified}/{pbar.n}|{100 * num_verified / pbar.n:.2f}%",
                          "Avg": f"{mean_time:.2f}s/img"})

        record.append((safe, elapsed_time, timeout))
        with open(args.output, "w") as f:
            dump(record, f, indent=2)
    pbar.close()
