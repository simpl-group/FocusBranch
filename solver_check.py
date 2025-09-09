from AbstractInterpreter import SequentialAT, L0BoundedInput, AffineAT, ReLUAT, Bounds, LinfBoundedInput
from utils import timeout_handler, TimeoutException, TQDM_BAR_FORMAT
from utils.shared import get4, add_shared_arguments
from models import get_network_and_dataset_utils
import argparse
import signal
import warnings
import threading
import queue
from multiprocessing import Process
from multiprocessing.connection import Client
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

def add_forward_constraints(model: gp.Model, seqAT: SequentialAT, bounded_x0, x: gp.MVar) -> gp.MVar:
    pre_activation_bounds_list = seqAT.get_preactivation_bounds(bounded_in=bounded_x0)
    relu_layer_idx = 0
    for layer_idx, AT in enumerate(seqAT.ATs):
        if isinstance(AT, AffineAT):  # FC, ConV2D
            weight, bias = AT.rp.W[0, ...].cpu().numpy(), AT.rp.b[0, ..., 0].cpu().numpy()
            x = weight @ x + bias
        elif isinstance(AT, ReLUAT):  # ReLU
            _bound = pre_activation_bounds_list[relu_layer_idx]
            lbound, ubound = _bound.l[0,..., 0].cpu().numpy(), _bound.u[0, ..., 0].cpu().numpy()

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
    return x

def gurobi_complete_check(seqAT: SequentialAT, bounded_x0: L0BoundedInput,
    demo: bool = True, ground_truth_label: int | None = None
) -> bool:
    model = gp.Model("check_unsat", env=env)
    img = bounded_x0.x[0, ..., 0].cpu().numpy()  # input: [784]
    delta = model.addMVar(img.shape, vtype=GRB.CONTINUOUS, name="delta")  # delta: [784]
    for i in range(img.shape[0]):
        model.addConstr(delta[i] <= bounded_x0.U - img[i])
        model.addConstr(bounded_x0.L - img[i] <= delta[i])
    x = img + delta  # input

    epsilon = bounded_x0.eps
    l0_loss = model.addVar(lb=epsilon, ub=epsilon)
    model.addGenConstrNorm(l0_loss, delta, 0.0)

    x = add_forward_constraints(model, seqAT, bounded_x0, x)

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

class GurobiWorker:
    def __init__(self, seqAT: SequentialAT, channel_size: int):
        self.__seqAT = seqAT
        self.__ub = 1. # (1. - self.__means) / self.__stds
        self.__lb = 0. # (0. - self.__means) / self.__stds
        self.__channel_size = channel_size

    def work(self, port: int):
        with Client(('localhost', port)) as self.conn:
            # Every iteration of this loop is one image
            message = self.conn.recv()
            while message != 'terminate':
                image, label = message
                self.__handle_image(image, label)
                message = self.conn.recv()

    def __handle_image(self, image, label):
        job_queue = queue.Queue()
        stop_event = threading.Event()

        def worker():
            while not stop_event.is_set() or not job_queue.empty():
                try:
                    pixel_indices = job_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                verified, adv_image, adv_label = self.milp_gurobi(image, label, pixel_indices)
                self.conn.send((pixel_indices, verified, adv_image, adv_label))
                job_queue.task_done()

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        while True:
            message = self.conn.recv()
            if message == 'stop':
                stop_event.set()
                break
            job_queue.put(message)
        worker_thread.join()
        self.conn.send('stopped')

    def milp_gurobi(self, image: np.ndarray, label: int, pixel_indices: list[int]):
        pixel_indices = self.update_indices_if_needed(pixel_indices)

        # initialize
        torch_image = torch.from_numpy(image).float().reshape(1, -1, 1)
        bounds = Bounds(l=torch_image.clone(), u=torch_image.clone())
        for pixel in pixel_indices:
            bounds.l[0, pixel, 0] = self.__lb
            bounds.u[0, pixel, 0] = self.__ub
        bounded_x0 = LinfBoundedInput(x=torch_image, eps=None, bounds=bounds)
        self.__seqAT.tail.make_marginals(true_labels=torch.Tensor([label]).long(), device="cpu")

        # constraint
        model = gp.Model("check_unsat", env=env)
        img = model.addMVar(image.shape, vtype=GRB.CONTINUOUS, name="x")
        for i in range(image.shape[0]):
            if i not in pixel_indices:
                model.addConstr(img[i] == image[i])
            else:
                model.addConstr(img[i] <= self.__ub)
                model.addConstr(self.__lb <= img[i])

        # forward
        x = add_forward_constraints(model, self.__seqAT, bounded_x0, img)

        label_size = self.__seqAT.tail.W.shape[1]
        any_not_safe = [model.addVar(vtype=GRB.BINARY) for _ in range(label_size)]
        for i in range(label_size):
            if i == label:
                model.addConstr(any_not_safe[i] == 0)
                continue
            model.addGenConstrIndicator(any_not_safe[i], True, x[i] <= 0)

        res = model.addVar(vtype=GRB.BINARY, lb=1, ub=1)  # True constant
        model.addGenConstrOr(res, any_not_safe)

        model.optimize()

        # return args
        if model.Status == GRB.Status.INFEASIBLE:
            return True, None, None
        if model.Status in {GRB.Status.OPTIMAL, GRB.Status.SOLUTION_LIMIT}:
            adv_image = img.X.tolist()
            for i in range(label_size):
                if any_not_safe[i].X > 0.5:
                    return False, adv_image, i
        raise NotImplementedError(f"Unknown state: {model.Status}.")

    def update_indices_if_needed(self, pixel_indices):
        if self.__channel_size == 1:
            return pixel_indices
        indices = []
        for pixel in pixel_indices:
            for i in range(self.__channel_size):
                indices.append(pixel * self.__channel_size + i)
        return indices

def launch_gurobi_worker(netname: str, port: int) -> Process:
    def init_worker():
        net, dataset_utils = get_network_and_dataset_utils(netname, device="cpu")
        x_ = torch.zeros(1, *dataset_utils.image_shape).to("cpu")
        seqAT = SequentialAT.from_sequential(net, x_)
        gurobi_worker = GurobiWorker(seqAT=seqAT, channel_size=dataset_utils.channel_size)
        gurobi_worker.work(port=port)
    process = Process(target=init_worker, daemon=True)
    process.start()
    return process

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
        bounded_x0.update_x(images.flatten(start_dim=1).unsqueeze(2))  # (batch, d1, 1)
        if args.target_label is not None:
            seqAT.tail.make_marginal(true_labels=pred_labels, target_label=args.target_label, device="cpu")
        else:
            seqAT.tail.make_marginals(true_labels=pred_labels, device="cpu")

        timeout = True
        safe = False
        elapsed_time = 0.
        try:
            signal.alarm(args.timeout)
            t0 = perf_counter()
            safe = gurobi_complete_check(seqAT, bounded_x0, demo=args.demo,
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
