from AbstractInterpreter import SequentialAT, L0BoundedInput, AffineAT, ReLUAT, Bounds, LinfBoundedInput
from utils import timeout_handler, TQDM_BAR_FORMAT
from utils.shared import get4, add_shared_arguments
from models import get_network_and_dataset_utils
from data.db import insert_milp, milp_should_skip, milp_config_consistent, sigma0_min_eps
import os
import argparse
import warnings
import threading
import queue
from multiprocessing.connection import Client
from time import perf_counter, sleep
import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
from tqdm import tqdm

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

def gurobi_complete_check(seqAT: SequentialAT, bounded_x0: L0BoundedInput, demo: bool = True
) -> tuple[bool,bool,bool]: # (verified, timed_out, found_adv)
    model = gp.Model("check_unsat", env=env)
    img = bounded_x0.x[0, ..., 0].cpu().numpy()  # input: [784] or [3072]
    delta = model.addMVar((bounded_x0.num_features,), vtype=GRB.CONTINUOUS, name="delta")
    for i in range(img.shape[0]):
        model.addConstr(delta[i] <= bounded_x0.U - img[i])
        model.addConstr(bounded_x0.L - img[i] <= delta[i])
    x = img + delta  # input

    if bounded_x0.channel_size == 1:
        model.addGenConstrNorm(model.addVar(lb=0.0, ub=bounded_x0.eps), delta, 0.0)
    else:
        # For RGB images, we want to count the number of pixels that are changed, not the number of features that are changed. For example, if one pixel is changed from (0, 0, 0) to (1, 1, 1), we want to count it as 1 change, not 3 changes.
        # We can use an auxiliary binary variable for each pixel to indicate whether it is changed or not.
        pixel_changed = model.addMVar(bounded_x0.num_pixels, vtype=GRB.BINARY, name="pixel_changed")
        # inputs are flattened from (C, H, W): all pixels of channel 0, then channel 1, ..., so the features of pixel i are at i + c * num_pixels
        M = bounded_x0.U - bounded_x0.L  # big-M constant
        feature_changed = model.addMVar(bounded_x0.num_features, vtype=GRB.BINARY, name="feature_changed")
        for j in range(bounded_x0.num_features):
            # feature_changed[j] == 1 iff delta[j] != 0, using big-M formulation
            model.addConstr(delta[j] <= M * feature_changed[j])
            model.addConstr(delta[j] >= -M * feature_changed[j])
        for i in range(bounded_x0.num_pixels):
            # If any of the features of pixel i is changed, then pixel_changed[i] should be 1
            feature_vars = [feature_changed[i + c * bounded_x0.num_pixels] for c in range(bounded_x0.channel_size)]
            model.addGenConstrOr(pixel_changed[i], feature_vars)
        model.addConstr(pixel_changed.sum() <= bounded_x0.eps)

    x = add_forward_constraints(model, seqAT, bounded_x0, x)

    # x: the margins of the true label over the other labels; adversarial iff one of them is <= 0
    any_not_safe = [model.addVar(vtype=GRB.BINARY) for _ in range(x.shape[0])]
    for i in range(x.shape[0]):
        model.addGenConstrIndicator(any_not_safe[i], True, x[i] <= 0)

    res = model.addVar(vtype=GRB.BINARY, lb=1, ub=1)  # True constant
    model.addGenConstrOr(res, any_not_safe)

    model.optimize()

    if model.Status == GRB.Status.INFEASIBLE:
        return True, False, False
    if model.Status in {GRB.Status.OPTIMAL, GRB.Status.SOLUTION_LIMIT}:
        if demo:
            print(f"Solution found")
            delta_values = np.asarray(delta.X)
            xs = np.nonzero(delta_values)
            print(f"Delta indices: {xs}, value: {delta_values[xs]}")
            print("Original input at delta indices:")
            print(img[xs])
            print("Modified input at delta indices:")
            print(img[xs] + delta.X[xs])
        return False, False, True
    if model.Status == GRB.Status.TIME_LIMIT:
        if demo: warnings.warn("Gurobi solver timed out")
        return False, True, False
    if model.Status == GRB.Status.INTERRUPTED:
        warnings.warn("Gurobi solver interrupted")
    else:
        warnings.warn(f"Solver status: {model.Status}")
    return False, False, False

def gurobi_env_waiting(params: dict) -> gp.Env:
    """gp.Env(params), waiting while the license server has no token free ("use limit exceeded"): a cluster's token
    pool can be used up for hours, and a MILP worker that gives up here leaves its run hanging at start-up.
    Stops waiting (exit status 1) once the parent process is gone, e.g. a killed `python .` run."""
    parent, waited = os.getppid(), False
    while True:
        try:
            return gp.Env(params=params)
        except gp.GurobiError as e:
            if 'use limit' not in str(e):
                raise
            if not waited:
                print(f'Gurobi: {e}; waiting for a free license token', file=sys.stderr, flush=True)
                waited = True
            sleep(10)
            if os.getppid() != parent:  # reparented: nobody is waiting for this process any more
                print('Gurobi: parent process is gone, no longer waiting for a license token', file=sys.stderr, flush=True)
                sys.exit(1)


class MILPWorker:
    def __init__(self, net: torch.nn.Sequential, x_: torch.Tensor, milp_solver: str="gurobi"):
        _, self.__channel_size, __H, __W = x_.shape
        self.__ub = 1. # (1. - self.__means) / self.__stds
        self.__lb = 0. # (0. - self.__means) / self.__stds
        self.__num_pixels = __H * __W
        if milp_solver == "gurobi":
            self.milp = self.milp_gurobi
            self.__seqAT = SequentialAT.from_sequential(net, x_)
            self.__model_lock = threading.Lock()
            self.__current_model = None
            self.__env = gurobi_env_waiting({"OutputFlag": 0, "SolutionLimit": 1})
        elif milp_solver == "abcrown":
            from abcrown import ABCrownSolver, IOConstraints, input_vars, output_vars
            import contextlib, logging
            logging.getLogger("auto_LiRPA").setLevel(logging.WARNING)
            self.milp = self.milp_abcrown
            self.__net = net
            self.__img_shape = (self.__channel_size, __H, __W)
        else:
            assert False

    def work(self, port: int):
        try:
            with Client(('localhost', port)) as self.conn:
                # Every iteration of this loop is one image
                message = self.conn.recv()
                while message != 'terminate':
                    image, label = message
                    self.__handle_image(image, label)
                    message = self.conn.recv()
        except (EOFError, ConnectionError):
            os._exit(1)  # parent is gone: exit immediately, don't tear down Gurobi under a running solve

    def __handle_image(self, image, label):
        job_queue = queue.Queue()
        stop_event = threading.Event()

        def terminate_current_model():
            if self.milp == self.milp_abcrown: return # TODO
            with self.__model_lock:
                if self.__current_model is not None:
                    self.__current_model.terminate()

        def worker():
            while not stop_event.is_set():
                try:
                    pixel_indices = job_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                verified, adv_image, adv_label = self.milp(image, label, self.update_indices_if_needed(pixel_indices))
                self.conn.send((pixel_indices, verified, adv_image, adv_label))
                job_queue.task_done()

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        while True:
            message = self.conn.recv()
            if message == 'stop':
                stop_event.set()
                terminate_current_model()
                break
            job_queue.put(message)
        worker_thread.join()
        self.conn.send('stopped')

    def milp_abcrown(self, image: np.ndarray, label: int, pixel_indices: list[int]):
        # marginal_tail = make_marginals_tail(label)
        torch_image = torch.from_numpy(image).float()#
        upper_bound = torch_image.clone()
        lower_bound = torch_image.clone()
        for pixel in pixel_indices:
            lower_bound[pixel] = self.__lb
            upper_bound[pixel] = self.__ub
        lower_bound = lower_bound.reshape(1, *self.__img_shape)
        upper_bound = upper_bound.reshape(1, *self.__img_shape)
        x = input_vars((1, *self.__img_shape))
        y = output_vars((1, 10))
        output_constraint = None
        for i in list(range(10)):
            if i == label: continue
            if output_constraint is None:
                output_constraint = (y[0, label] > y[0, i])
            else:
                output_constraint &= (y[0, label] > y[0, i])

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            result = ABCrownSolver(self.__net, x, y).verify(
                constraints=IOConstraints(
                    input_vars=x,
                    output_vars=y,
                    input_constraint=(x >= lower_bound) & (x <= upper_bound),
                    output_constraint=output_constraint,
                )
            )
        if result.success:
            return True, None, None
        raise NotImplementedError(f"ABCrown solver returned status {result.status} and success {result.success}.")

    def milp_gurobi(self, image: np.ndarray, label: int, pixel_indices: list[int]):
        torch_image = torch.from_numpy(image).float().reshape(1, -1, 1)
        bounds = Bounds(l=torch_image.clone(), u=torch_image.clone())
        for pixel in pixel_indices:
            bounds.l[0, pixel, 0] = self.__lb
            bounds.u[0, pixel, 0] = self.__ub
        bounded_x0 = LinfBoundedInput(x=torch_image, eps=None, bounds=bounds)
        self.__seqAT.tail.make_marginals(true_label=label, device="cpu")

        # constraint
        model = gp.Model("check_unsat", env=self.__env)
        with self.__model_lock:
            self.__current_model = model
        img = model.addMVar(image.shape, vtype=GRB.CONTINUOUS, name="x")
        for i in range(image.shape[0]):
            if i not in pixel_indices:
                model.addConstr(img[i] == image[i])
            else:
                model.addConstr(img[i] <= self.__ub)
                model.addConstr(self.__lb <= img[i])

        # forward
        x = add_forward_constraints(model, self.__seqAT, bounded_x0, img)

        # x: the margins of the true label over the other labels; adversarial iff one of them is <= 0
        any_not_safe = [model.addVar(vtype=GRB.BINARY) for _ in range(x.shape[0])]
        for i in range(x.shape[0]):
            model.addGenConstrIndicator(any_not_safe[i], True, x[i] <= 0)

        res = model.addVar(vtype=GRB.BINARY, lb=1, ub=1)  # True constant
        model.addGenConstrOr(res, any_not_safe)

        model.optimize()
        with self.__model_lock:
            if self.__current_model is model:
                self.__current_model = None

        # return args
        if model.Status == GRB.Status.INFEASIBLE:
            return True, None, None
        if model.Status in {GRB.Status.OPTIMAL, GRB.Status.SOLUTION_LIMIT}:
            adv_image = img.X.tolist()
            for i in range(len(any_not_safe)):
                if any_not_safe[i].X > 0.5:
                    return False, adv_image, i if i < label else i + 1  # row i: the margin over label i, or i + 1 past the true label
        if model.Status in {GRB.Status.TIME_LIMIT, GRB.Status.INTERRUPTED}:
            return False, None, None
        raise NotImplementedError(f"Unknown state: {model.Status}.")

    def update_indices_if_needed(self, pixel_indices: list[int]) -> list[int]:
        if self.__channel_size == 1:
            return pixel_indices
        indices = []
        for pixel in pixel_indices:
            for i in range(self.__channel_size):
                indices.append(pixel + i * self.__num_pixels)
        return indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete verification of L0 robustness using Gurobi.")
    add_shared_arguments(parser)
    parser.add_argument("--port", type=int, default=-1)
    args = parser.parse_args()
    if args.port != -1:
        net, dataset_utils = get_network_and_dataset_utils(args.net, device="cpu")
        x_ = torch.zeros(1, *dataset_utils.image_shape).to("cpu")
        MILPWorker(net=net, x_=x_, milp_solver=args.solver).work(port=args.port)
        exit(0)

    if not milp_config_consistent(net=args.net, img=args.start_idx+args.test_batches-1, eps=args.epsilon, timeout=args.timeout):
        warnings.warn(f"Configuration inconsistent: net={args.net}, img={args.start_idx+args.test_batches-1}, eps={args.epsilon}, timeout={args.timeout}.")
        if not args.force_store and not args.demo:
            exit(1)

    import signal
    signal.signal(signal.SIGALRM, timeout_handler)
    env = gurobi_env_waiting({"OutputFlag": 0, "SolutionLimit": 1, "TimeLimit": args.timeout})

    if args.demo:
        args.test_batches = 1

    net, seqAT, bounded_x0, data_loader = get4(args)

    total_time = 0.
    num_verified = 0
    net.cpu()
    pbar = tqdm(total=args.test_batches, bar_format=TQDM_BAR_FORMAT, disable=args.demo, file=sys.stdout)
    for _i, (images, pred_labels) in enumerate(data_loader):
        i = _i + args.start_idx
        adv_min_eps = sigma0_min_eps(net=args.net, img=i)
        if adv_min_eps is not None and adv_min_eps <= args.epsilon:
            print(f"Image {i} already has a known adversarial example with eps={adv_min_eps} <= epsilon={args.epsilon}.")
            skip = True
            insert_milp(net=args.net, img=i, eps=args.epsilon, found_adv=True)
        else: skip = milp_should_skip(net=args.net, img=i, eps=args.epsilon)
        if skip: print(f"Image {i} for net={args.net}, eps={args.epsilon} should be skipped.")
        if not skip or args.force_store or args.demo:
            if net(images).argmax(dim=1).eq(pred_labels).all():
                bounded_x0.update_x(images.flatten(start_dim=1).unsqueeze(2))  # (batch, d1, 1)
                seqAT.tail.make_marginals(true_label=pred_labels.item(), device="cpu")

                elapsed_time = 0.
                t0 = perf_counter()
                safe, timed_out, found_adv = gurobi_complete_check(
                    seqAT, bounded_x0, demo=args.demo
                )
                num_verified += 1 if safe else 0
                bounded_x0.clear_cache()

                elapsed_time = perf_counter() - t0
                total_time += elapsed_time

                if args.demo:
                    print(f"Verified safe: {safe}, Timedout:{timed_out}, found_adv:{found_adv}, Time: {elapsed_time:.2f}s, result NOT stored")
                    break
                insert_milp(
                    net=args.net, img=i, eps=args.epsilon,
                    verified=safe, timed_out=timed_out, found_adv=found_adv,
                    total_duration=round(elapsed_time, 3)
                )
            else:
                if args.demo:
                    print(f"Misclassified! Result NOT stored")
                    break
                insert_milp(net=args.net, img=i, eps=args.epsilon, cc=False)

        pbar.update(1)
        mean_time = total_time / num_verified if num_verified > 0 else 0
        pbar.set_postfix({"Verified": f"{num_verified}/{pbar.n}|{100 * num_verified / pbar.n:.2f}%", "Avg": f"{mean_time:.2f}s/img"})
    pbar.close()
