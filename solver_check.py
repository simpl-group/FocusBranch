import gurobipy as gp
from gurobipy import GRB
import numpy as np

def check_unsat(x0: np.ndarray, coefficients_x0: np.ndarray, b: np.ndarray,
                eps: int, selected_idx: np.ndarray=None, pixel_value_range=(0, 1)) -> bool:
    num_batch, num_pixel = len(x0), len(x0[0])
    L, U = pixel_value_range

    model = gp.Model("check_unsat")

    controllers = model.addMVar((num_pixel,), vtype=GRB.BINARY, name="c")
    triggers = model.addMVar((num_pixel,), vtype=GRB.CONTINUOUS, lb=L, ub=U, name="t")

    # l0=eps constraint
    model.addConstr(controllers.sum() == eps)
    if selected_idx is not None:
        assert selected_idx.ndim == 1 and len(selected_idx) <= eps
        model.addConstr(controllers[selected_idx] == 1)

    for i in range(num_batch):
        img = controllers * triggers + (1 - controllers) * x0[i]
        model.addConstr(coefficients_x0[i] @ img + b[i] <= 0)

    model.optimize()

    if model.Status == GRB.Status.INFEASIBLE:
        print("Collectively Safe")
        return True
    elif model.Status == GRB.Status.OPTIMAL:
        print("Solution found:")
        for idx in range(num_pixel):
            if controllers[idx].X > 0.5:  # If idx is within the trigger
                print(f"{idx}: {triggers[idx].X}")
        return False
    else:
        print(f"Solver status: {model.Status}")
        return False
