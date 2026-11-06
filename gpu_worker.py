"""
GPU worker process: bounds nodes on one GPU on behalf of the verifier (see `verifier.py`).

In super branching (SCSaP), every worker verifies its share of the covering verification design: every num_workers-th
block, streamed by a covering generator process (`covering_generator`). Nodes wait in per-size pools, a full pool is
bounded in batches, and the unknown super nodes are refined with the refinement coverings.
"""
from multiprocessing.connection import Client, Connection, Listener
from time import perf_counter
import math
import threading
import numpy as np
import torch
from covering_generator import launch_covering_generator, CVD_MIN_T
from coverd_finite_geometries import Geometry
from data import load_covering, Strategy
from split_and_prune import prepare4pruning, prune
from utils import protocol as P
from bounding_engine import BoundingEngine, batched_randsample, selected_indices_sample


def refinement_subsets(focus: list[int], covering: list[list[int]]) -> list[list[int]]:
    """The children of a super node: the subsets of its focus set given by the blocks of a covering of its positions."""
    return [[focus[i] for i in block] for block in covering]


class SuperBranching:
    """Branch-and-bound over super nodes (Algorithm 5, lines 5-11) for one image on one worker."""

    def __init__(self, worker: "GpuWorker", geometry: Geometry, chosen_indices: list[int], refinement_strategy: dict[int, int | None], stop_event: threading.Event):
        self.worker, self.engine = worker, worker.engine
        self.geometry, self.chosen_indices = geometry, chosen_indices
        self.level = geometry.t
        self.eps = self.engine.eps
        self.stop_event = stop_event
        # if no sampled super node was safe at this level, only concrete nodes are worth bounding in batches
        self.max_size_to_batch = self.engine.max_sizes_to_batch.get(self.level, self.level)
        self.pool_size = self.engine.pool_size
        # in descending order of size: the remaining nodes are bounded largest first, so their children can still be pooled
        self.pools: dict[int, list[list[int]]] = {k: [] for k in range(self.max_size_to_batch, self.level - 1, -1)}
        self.refinement_coverings = {v: load_covering(v, k, self.level) for v, k in refinement_strategy.items() if k is not None}
        self.generator = None

    def stop(self):
        if self.generator is not None and self.generator.poll() is None:  # still running
            self.generator.terminate()

    def __handle_supernodes(self, size: int):
        if self.stop_event.is_set(): return
        start = perf_counter()
        nodes = torch.tensor(self.pools[size])
        self.pools[size].clear()
        safes = self.engine.bound_supernodes(nodes, self.level)
        self.worker.send((P.STATS, size, nodes.shape[0], safes.sum().item(), perf_counter() - start))
        if not safes.all():  # need to refine
            if size == self.level:  # concrete nodes, cannot be refined here
                self.worker.send((P.SUSPECTS if self.level == self.eps else P.UNKNOWN, nodes[~safes].numpy()))
            else:
                children = []
                for focus in nodes[~safes].tolist():
                    children.extend(refinement_subsets(focus, self.refinement_coverings[size]))
                for child in children:
                    self.__handle_supernode(child)

    def __handle_supernode(self, focus: list[int]):
        if self.stop_event.is_set(): return
        size = len(focus)
        # stats: (size, checked, verified, seconds)
        if size < self.level:  # no level-subset inside, nothing to verify
            self.worker.send((P.STATS, size, 1, 1, 0.))
        elif size <= self.max_size_to_batch:  # bounded in batches
            self.pools[size].append(focus)
            if len(self.pools[size]) == self.pool_size:
                self.__handle_supernodes(size)
        else:  # not worth bounding (default to unknown), refine right away
            self.worker.send((P.STATS, size, 1, 0, 0.))
            for child in refinement_subsets(focus, self.refinement_coverings[size]):
                self.__handle_supernode(child)

    def __verify_blocks(self):
        num_workers, worker_id = self.worker.num_workers, self.worker.worker_id
        num_blocks = math.floor(self.geometry.number_of_blocks() / num_workers)
        if worker_id < self.geometry.number_of_blocks() % num_workers:
            num_blocks += 1
        with Listener(('localhost', 0)) as blocks_listener:
            self.generator = launch_covering_generator(
                q=self.geometry.q, m=self.geometry.m, t=self.geometry.t,
                port=int(blocks_listener.address[1]),
                chosen_indices=self.chosen_indices,
                num_workers=num_workers,
                worker_id=worker_id
            )
            with blocks_listener.accept() as blocks_conn:
                for _ in range(num_blocks):
                    if self.stop_event.is_set(): return
                    self.__handle_supernode(blocks_conn.recv())  # a sorted list[int]
                assert blocks_conn.recv() == 'done'
                blocks_conn.send('done')
        for size in self.pools:  # the remaining nodes, largest first
            if len(self.pools[size]) > 0:
                self.__handle_supernodes(size)
        self.worker.send((P.DONE,))

    def run(self):
        try:
            self.__verify_blocks()
        except EOFError:
            pass  # the generator was terminated early


class GpuWorker:
    def __init__(self, engine: BoundingEngine, num_workers: int, worker_id: int):
        self.engine = engine
        self.num_workers, self.worker_id = num_workers, worker_id
        self.__send_lock = threading.Lock()

    def send(self, message):
        with self.__send_lock:  # the job thread and the message loop both talk to the verifier
            self.conn.send(message)

    def serve(self, port: int):
        with Client(('localhost', port)) as self.conn:
            self.send(self.worker_id)
            # every iteration of this loop is one image
            message = self.conn.recv()
            while not (isinstance(message, str) and message == P.TERMINATE):
                seed, image, label = message
                self.__run_image(seed, image, label)
                message = self.conn.recv()

    def concrete_sample_all_levels(self) -> tuple[list[int], list[int]]:
        engine = self.engine
        assert engine.bounded_x0.level is None, "bounded_x0.level must be None since we are using engine.bounded_x0.selecting"
        sample_counts, sample_successes = [0] * (engine.eps + 1), [0] * (engine.eps + 1)
        if not engine.strategy.require_supercheck():
            start_level = 2
        elif engine.strategy == Strategy.ForceCVD:
            start_level = CVD_MIN_T
        else: assert False, f"Unsupported strategy {engine.strategy}"
        for level in range(start_level, engine.eps + 1):
            focus_sets = selected_indices_sample(engine.num_pixels, level, engine.rep_each_level, num_workers=self.num_workers, worker_id=self.worker_id)
            sample_counts[level] = focus_sets.shape[0]
            sample_successes[level] = int(engine.bound_nodes(focus_sets).sum().item())
            if sample_successes[level] == sample_counts[level]:
                # fake the rest
                sample_counts[level+1:engine.eps + 1] = [min(engine.rep_each_level, math.comb(engine.num_pixels, l)) for l in range(level + 1, engine.eps + 1)]
                sample_successes[level+1:engine.eps + 1] = sample_counts[level+1:engine.eps + 1]
                break  # All nodes are safe, no need to go deeper
        return sample_counts, sample_successes

    def super_sample_all_k(self, level: int) -> tuple[list[int], list[int], list[float]]:
        engine = self.engine
        sampling_counts, sampling_successes , sampling_time = [], [], []
        rep_each_k = engine.rep_each_k
        for k in range(level, engine.max_k + 1):
            k_sized_supersets = batched_randsample(engine.num_pixels, k, rep_each_k)
            batched_supercheck_start = perf_counter()
            sampling_successes.append(int(engine.bound_supernodes(k_sized_supersets, level).sum().item()))
            sampling_time.append(perf_counter() - batched_supercheck_start)
            sampling_counts.append(k_sized_supersets.shape[0])
            if sampling_successes[-1] == 0:
                rep_each_k = max(1, engine.rep_each_k // 20)  # if no safe supersets, use smaller batch size to sample more
            else:
                engine.max_sizes_to_batch[level] = k
        return sampling_counts, sampling_successes, sampling_time

    def __plan(self):
        """Answer the verifier's probing and sampling requests; returns the first message of the verification stage."""
        engine = self.engine
        if engine.strategy != Strategy.toptCoVerD:
            if self.worker_id == 0:  # the root node
                self.send(1. if bool(engine.bound_nodes(torch.empty((1, 0), dtype=torch.int64)).item()) else -1.)
            message = self.conn.recv()
            while message[0] == P.BOUND_CONCRETE:
                self.send(engine.min_margin_lbs(torch.from_numpy(message[1])).numpy())
                message = self.conn.recv()
            assert message[0] == P.PROBE_DONE, f"Expected {P.PROBE_DONE}, but got {message}"
            if message[1] > 1:
                self.send(self.concrete_sample_all_levels())
        message = self.conn.recv()
        while message[0] == P.SAMPLE_SUPER:
            self.send(self.super_sample_all_k(int(message[1])))
            message = self.conn.recv()
        return message

    def __run_image(self, seed: int, image: np.ndarray, label: int):
        engine = self.engine
        torch.manual_seed(seed)
        engine.adapt_target_images(images=torch.tensor(image, dtype=torch.float32).reshape(1, -1, 1),
                              true_label=label, check_first=False)
        message = self.__plan()
        if message[0] == P.SKIP:
            return
        stop_event = threading.Event()
        super_branching: SuperBranching | None = None
        if message[0] == P.START_CONCRETE:
            level = int(message[1])
            if level == 1:  # level-1 nodes were all bounded while probing
                job = threading.Thread(target=lambda: None, daemon=True)
            else:
                all_combs = torch.combinations(torch.arange(engine.num_pixels), level)
                share = all_combs.shape[0] // self.num_workers
                end = all_combs.shape[0] if self.worker_id == self.num_workers - 1 else (self.worker_id + 1) * share
                job = threading.Thread(target=self.__check_concrete, args=(all_combs[self.worker_id * share:end], stop_event), daemon=True)
                del all_combs
        elif message[0] == P.START_SUPER:
            _, geometry, chosen_indices, refinement_strategy = message
            super_branching = SuperBranching(self, geometry, chosen_indices, refinement_strategy, stop_event)
            level = geometry.t
            job = threading.Thread(target=super_branching.run, daemon=True)
        else: assert False, f"Unknown message {message}"
        job.start()
        while True:
            if not self.conn.poll():
                continue  # polls like the paper's version: a blocking recv() here changes the timings of super branching
            message = self.conn.recv()
            if isinstance(message, str) and message == P.STOP:
                stop_event.set()
                if super_branching is not None: super_branching.stop()
                break
            job.join()  # the previous level is finished before the verifier sends the next one
            super_branching = None
            if message[0] == P.SPLIT_PRUNE_CHECK:
                new_combs, reference_bitmasks, target_s_cnt = prepare4pruning(torch.from_numpy(message[1]), engine.num_pixels)
                # striding over the children balances the pruned workload among workers
                new_combs = new_combs.to(engine.device)[self.worker_id::self.num_workers].contiguous()
                combs = prune(new_combs, reference_bitmasks, target_s_cnt, level)
                self.send(combs.shape[0])
            elif message[0] == P.BOUND_CONCRETE:
                combs = torch.from_numpy(message[1])
            else: assert False, f"Unknown message {message}"
            level += 1
            job = threading.Thread(target=self.__check_concrete, args=(combs, stop_event), daemon=True)
            job.start()
        job.join()
        self.send(P.STOPPED)

    def __check_concrete(self, combs: torch.Tensor, stop_event: threading.Event):
        self.send(self.engine.unknown_concrete(combs, stop_event))


class GpuWorkerPool:
    """The connections to the GPU workers, ordered by worker id."""

    def __init__(self, workers: list[Connection]):
        self.workers = workers

    def __len__(self) -> int:
        return len(self.workers)

    def broadcast(self, message):
        for worker in self.workers:
            worker.send(message)

    def gather(self) -> list:
        return [worker.recv() for worker in self.workers]

    def scatter(self, tag: str, rows: np.ndarray | torch.Tensor):
        """Split rows evenly (the remainder goes to the last worker) and send (tag, rows_i) to worker i."""
        rows = rows.numpy() if isinstance(rows, torch.Tensor) else rows
        each = rows.shape[0] // len(self.workers)
        for i, worker in enumerate(self.workers):
            worker.send((tag, rows[i * each:] if i == len(self.workers) - 1 else rows[i * each:(i + 1) * each]))

    def gather_rows(self) -> np.ndarray:
        return np.vstack(self.gather())

    def wait_stopped(self):
        for worker in self.workers:
            while not (isinstance(message := worker.recv(), str) and message == P.STOPPED):
                pass  # results of abandoned work


if __name__ == '__main__':
    import argparse
    from utils.shared import add_shared_arguments, get_engine
    parser = argparse.ArgumentParser(description='L0 Robustness Verifier: (GPU) Worker',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_shared_arguments(parser)
    parser.add_argument('--worker_id', type=int, default=0, help='The index of the worker')
    parser.add_argument('--num-workers', type=int, default=1, help='Total number of workers')
    parser.add_argument('--port', type=int, default=0, help='The port main verification process is listening on')
    args = parser.parse_args()

    GpuWorker(get_engine(args), num_workers=args.num_workers, worker_id=args.worker_id).serve(port=args.port)
