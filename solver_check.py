import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import List

env = gp.Env(params={"OutputFlag": 0, "SolutionLimit": 1})

def solver_check_unsat(x0: np.ndarray, coefficients: np.ndarray, bias: np.ndarray,
                eps: int, selected_idx: List[int]=[], pixel_value_range=(0, 1),
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
