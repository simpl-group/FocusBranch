from AbstractInterpreter import SequentialAT, Bounds, LinfBoundedInput
from multiprocessing.connection import Client
from solver_check import add_forward_constraints
import threading
import queue
import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB

env = gp.Env(params={"OutputFlag": 0, "SolutionLimit": 1})

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
                self.conn.send('stopped')
                message = self.conn.recv()

    def __handle_image(self, image, label):
        job_queue = queue.Queue()
        stop_event = threading.Event()

        def job_receiver():
            while True:
                message = self.conn.recv()
                if message == 'stop':
                    stop_event.set()
                    break
                job_queue.put(message)

        def worker():
            while not stop_event.is_set() or not job_queue.empty():
                try:
                    pixel_indices = job_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                verified, adv_image, adv_label = self.milp_gurobi(image, label, pixel_indices)
                self.conn.send((pixel_indices, verified, adv_image, adv_label))
                job_queue.task_done()

        job_receiver_thread = threading.Thread(target=job_receiver, daemon=True)
        job_receiver_thread.start()

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        job_receiver_thread.join()
        worker_thread.join()

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


if __name__ == '__main__':
    from utils.shared import add_shared_arguments
    from models import get_network_and_dataset_utils
    from AbstractInterpreter import SequentialAT
    import argparse

    parser = argparse.ArgumentParser(description='MILP Gurobi (CPU) Worker', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', type=int, default=None, help='the port main process is listening on')
    add_shared_arguments(parser)
    args = parser.parse_args()
    net, dataset_utils = get_network_and_dataset_utils(args.net, device="cpu")
    x_ = torch.zeros(args.grain_size, *dataset_utils.image_shape).to("cpu")
    seqAT = SequentialAT.from_sequential(net, x_)

    l_zero_cpu_worker = GurobiWorker(seqAT=seqAT, channel_size=dataset_utils.channel_size)
    l_zero_cpu_worker.work(port=args.port)
