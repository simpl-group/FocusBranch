import gurobipy as gp
from gurobipy import GRB
import numpy as np
from AbstractInterpreter import SequentialAT, L0BoundedInput, ActivationAT, Bounds, AffineAT, ReLUAT
from utils import *
from models import get_network
from data.prepare import get_loader
from data import DATASETS
import argparse
import torch
import signal

env = gp.Env(params={"OutputFlag": 0, "SolutionLimit": 1})


def gurobi_check_unsat(x0: np.ndarray, coefficients: np.ndarray, bias: np.ndarray,
                eps: int, selected_idx: list[int]=[], pixel_value_range=(0, 1),
                mute=False) -> bool:
    num_batch, num_pixel = len(x0), len(x0[0])
    L, U = pixel_value_range

    model = gp.Model("check_unsat", env=env)

    controllers = model.addMVar((num_pixel,), vtype=GRB.BINARY, name="c")
    triggers = model.addMVar((num_pixel,), vtype=GRB.CONTINUOUS, lb=L, ub=U, name="t")

    # l0=eps constraint
    model.addConstr(controllers.sum() == eps)
    if len(selected_idx) > 0:
        selected_idx = np.array(list(selected_idx))
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


def gurobi_complete_check(bounded_x0: L0BoundedInput, seqAT: SequentialAT,
        pre_activation_bounds_list: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    model = gp.Model("check_unsat", env=env)
    img = bounded_x0.x[0, ..., 0].numpy()  # input: [784]
    delta = model.addMVar(img.shape, vtype=GRB.CONTINUOUS, name="delta")  # delta: [784]
    x = img + delta  # input
    # 0 <= img + delta <= 1
    model.addConstr(bounded_x0.L <= x, name="input_lower_bound")
    model.addConstr(x <= bounded_x0.U, name="input_upper_bound")

    # L0 norm
    epsilon = bounded_x0.eps
    l0_loss = model.addVar(lb=epsilon, ub=epsilon)
    model.addGenConstrNorm(l0_loss, delta, 0., name="L0_norm")

    # forward
    relu_layer_idx = 0
    for layer_idx, AT in enumerate(seqAT.ATs):
        if isinstance(AT, AffineAT):  # FC, ConV2D
            weight, bias = AT.rp.W[0, ...].numpy(), AT.rp.b[0, ..., 0].numpy()
            # x = x @ weight.T + bias
            x = weight @ x + bias
        elif isinstance(AT, ReLUAT):  # ReLU
            lbound, ubound = pre_activation_bounds_list[relu_layer_idx]

            # x:MLinExpr -> intermediate_x:MVar
            intermediate_x = model.addMVar(x.shape, vtype=GRB.CONTINUOUS, name=f"ReLU{layer_idx}_ix")
            model.addConstr(x == intermediate_x)

            y = model.addMVar(x.shape, vtype=GRB.CONTINUOUS, name=f"ReLU{layer_idx}_y")
            for idx, (lb, ub) in enumerate(zip(lbound, ubound)):
                if ub <= 0:
                    # Case 1: ubound <= 0
                    model.addConstr(y[idx] == 0)
                elif lb >= 0:
                    # Case 2: lbound >= 0
                    model.addConstr(y[idx] == intermediate_x[idx])
                else:
                    # Case 3: lbound <= 0 <= ubound
                    model.addGenConstrPWL(intermediate_x[idx], y[idx], [lb, 0, ub], [0, 0, ub],
                                          name=f"ReLU{layer_idx}_appr")
            x = y
            relu_layer_idx += 1
        else:
            raise NotImplementedError(type(AT))

    model.addConstr(x <= 0., name="output_greater_than_0")
    model.optimize()

    if model.Status == GRB.Status.INFEASIBLE: return True
    if model.Status == GRB.Status.OPTIMAL:
        print("Solution found:")
    else:
        print(f"Solver status: {model.Status}")
    return False


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, timeout_handler)
    parser = argparse.ArgumentParser(
        description="Complete verification of L0 robustness using Gurobi."
    )
    add_shared_arguments(parser)
    args = parser.parse_args()
    net, spec = get_network(args.net)
    data_loader = get_loader(spec.dataset_name, batch_size=1, target_label=args.target_label,
                             num_batches=args.start_idx + args.test_batches)
    x_ = torch.zeros(1, *DATASETS[spec.dataset_name].image_shape)
    seqAT = SequentialAT.from_sequential(net, x_)
    i = 0
    for images, pred_labels in data_loader:
        if i < args.start_idx:
            i += 1
            continue
        assert net(images).argmax(dim=1).eq(pred_labels).all()
        x0 = images.flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
        bounded_x0 = L0BoundedInput(x0, args.epsilon)
        seqAT.tail.make_marginal(true_labels=pred_labels, target_label=args.target_label, device="cpu")
        break

    pre_activation_bounds_list = []
    for AT in seqAT.ATs:
        if isinstance(AT, ActivationAT):
            lrp_wrt_inputs, urp_wrt_inputs = AT.parent.backsubstitution()
            pre_activation_bounds = bounded_x0.compute_bounds(lrp_wrt_inputs, urp_wrt_inputs)
            AT.lrp, AT.urp = AT.compute_rp(pre_activation_bounds)
            # convert to np.ndarray
            # delete batch dimension
            pre_activation_bounds_list.append([
                pre_activation_bounds.l[0, ..., 0].numpy(),  # lower bound
                pre_activation_bounds.u[0, ..., 0].numpy(),  # upper bound
            ])

    res = gurobi_complete_check(bounded_x0, seqAT, pre_activation_bounds_list)
    print(res)
