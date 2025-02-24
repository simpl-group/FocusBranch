import gurobipy as gp
from gurobipy import GRB

def check_unsat(x0: list, coefficients_x0: list, b: list, eps: int, pixel_value_range=(0, 1)) -> bool:
    num_batch, num_pixel = len(x0), len(x0[0])
    L, U = pixel_value_range

    model = gp.Model("check_unsat")

    controllers = model.addVars(num_pixel, vtype=GRB.BINARY, name="c")
    triggers = model.addVars(num_pixel, vtype=GRB.CONTINUOUS, lb=L, ub=U, name="t")

    # l0=eps constraint
    model.addConstr(gp.quicksum(controllers[idx] for idx in range(num_pixel)) == eps)

    for i in range(num_batch):
        img = [
            controllers[idx] * triggers[idx] + (1 - controllers[idx]) * x0[i][idx]
            for idx in range(num_pixel)
        ]
        model.addConstr(
            gp.quicksum(coefficients_x0[i][idx] * img[idx] for idx in range(num_pixel))
            + b[i] <= 0
        )

    model.optimize()

    if model.status == GRB.INFEASIBLE:
        print("Collectively Safe")
        return True
    elif model.status == GRB.OPTIMAL:
        print("Solution found:")
        for idx in range(num_pixel):
            if controllers[idx].x > 0.5:  # If idx is within the trigger
                print(f"{idx}: {triggers[idx].x}")
        return False
    else:
        print(f"Solver status: {model.status}")
        return False
