"""
Dispatches unbranchable unknown nodes (level-eps concrete nodes) to the MILP workers and collects their verdicts
(Algorithm 1, lines 11-13). Runs in a background thread of the analyzer.
"""
from multiprocessing.connection import Connection, wait
import queue
import sys
import threading
import time
import numpy as np
import torch
from tqdm import tqdm
from bimm import combs2bitmasks, bitmasks2combs, superset_sum
from utils import TQDM_BAR_FORMAT
from utils import protocol as P

JOBS_PER_WORKER = 3  # in-flight jobs per MILP worker


class MilpDispatcher:
    def __init__(self, cpu_workers: list[Connection], num_pixels: int, channel_size: int, eps: int, timeout: float):
        self.cpu_workers = cpu_workers
        self.num_pixels, self.channel_size, self.eps, self.timeout = num_pixels, channel_size, eps, timeout

    def update_image(self, image: np.ndarray, label: int):
        for cpu_worker in self.cpu_workers:
            cpu_worker.send((image, label))

    def start(self, start_time: float, deduplicate: bool):
        """Start dispatching jobs for the current image. Duplicates are removed once all jobs are known if `deduplicate`."""
        self.start_time = start_time
        self.jobs: queue.Queue = queue.Queue()
        self.settled = threading.Event()
        self.found_adversarial = threading.Event()
        self.waiting: set[tuple[int, ...]] = set()
        self.innocent: set[tuple[int, ...]] = set()
        self.adversarial_results = {'adversarial_pixels': None, 'adversarial_example': None, 'adversarial_label': None}
        self.__deduplicated = not deduplicate
        self.__thread = threading.Thread(target=self.__run)
        self.__thread.start()

    def submit(self, combs: np.ndarray | list[list[int]]):
        for comb in (combs.tolist() if isinstance(combs, np.ndarray) else combs):
            self.jobs.put(comb)

    def settle(self):
        """No more jobs will be submitted."""
        self.settled.set()

    def join(self):
        self.__thread.join()

    @property
    def unfinished(self) -> int:
        return self.jobs.unfinished_tasks

    def send_stop(self):
        """Ask every MILP worker to abandon the current image (see `wait_stopped`)."""
        for cpu_worker in self.cpu_workers:
            cpu_worker.send(P.STOP)

    def wait_stopped(self):
        for cpu_worker in self.cpu_workers:
            while not (isinstance(message := cpu_worker.recv(), str) and message == P.STOPPED):
                pass  # verdicts of abandoned jobs

    def stop(self):
        self.send_stop()
        self.wait_stopped()

    def done(self) -> bool:
        return self.found_adversarial.is_set()

    def __timed_out(self) -> bool:
        return time.time() - self.start_time >= self.timeout

    def __deduplicate(self):
        """Drop duplicated jobs and jobs already sent to a worker before showing the progress bar."""
        queued: list[list[int]] = []
        while True:
            try:
                queued.append(self.jobs.get_nowait())
                self.jobs.task_done()
            except queue.Empty:
                break
        if queued:
            unique = torch.tensor(queued, device="cuda").unique(dim=0)
            known = self.innocent | self.waiting
            if known:
                words = (self.num_pixels + 63) // 64
                unique_bm = combs2bitmasks(unique, words)
                unique = bitmasks2combs(unique_bm[superset_sum(unique_bm, combs2bitmasks(torch.tensor(list(known), device="cuda"), words)) == 0], self.eps)
            unique = unique.tolist()  # plain int lists: suspects are tracked as hashable tuples and sent to the MILP workers
            for job in unique:
                self.jobs.put(job)
            if len(queued) > len(unique):
                assert self.jobs.unfinished_tasks == len(unique) + len(self.waiting), \
                    f"Unfinished tasks in queue {self.jobs.unfinished_tasks} doesn't match expected {len(unique) + len(self.waiting)} after deduplication."
                print(f"Deduplicated MILP jobs queue, removed {len(queued) - len(unique)} duplicates, remaining jobs: {self.jobs.unfinished_tasks}")
        self.__deduplicated = True

    def __run(self):
        jobs_per_worker = np.zeros(len(self.cpu_workers), dtype=int)
        pbar = None
        try:
            while (not self.settled.is_set() or self.jobs.unfinished_tasks > 0) and not self.__timed_out():
                if pbar is None and self.settled.is_set():
                    if not self.__deduplicated:
                        self.__deduplicate()
                    pbar = tqdm(total=self.jobs.unfinished_tasks, desc="MILP jobs", bar_format=TQDM_BAR_FORMAT, mininterval=1.0, maxinterval=2.0, file=sys.stdout)
                while jobs_per_worker.min() < JOBS_PER_WORKER:  # the least busy worker takes the next job
                    try:
                        pixel_indices = self.jobs.get_nowait()
                    except queue.Empty:
                        break
                    suspect = tuple(pixel_indices)
                    if not self.__deduplicated and (suspect in self.innocent or suspect in self.waiting):
                        self.jobs.task_done()
                        continue
                    worker_id = int(jobs_per_worker.argmin())
                    self.waiting.add(suspect)
                    self.cpu_workers[worker_id].send(pixel_indices)
                    jobs_per_worker[worker_id] += 1
                for conn in wait(self.cpu_workers, timeout=0.01):
                    worker_id = self.cpu_workers.index(conn)
                    pixel_indices, verified, adversarial_example, adversarial_label = conn.recv()
                    self.jobs.task_done()
                    jobs_per_worker[worker_id] -= 1
                    if pbar is not None: pbar.update(1)
                    if verified:
                        self.waiting.remove(tuple(pixel_indices))
                        self.innocent.add(tuple(pixel_indices))
                    else:
                        self.__report_adversarial(pixel_indices, adversarial_example, adversarial_label)
                        return
        finally:
            if pbar is not None: pbar.close()

    def __report_adversarial(self, pixel_indices: list[int], adversarial_example: list[float], adversarial_label: int):
        self.found_adversarial.set()
        self.adversarial_results['adversarial_pixels'] = str(pixel_indices)
        features = [pixel + c * self.num_pixels for pixel in pixel_indices for c in range(self.channel_size)]
        delta = {i: adversarial_example[i] for i in features}
        print(f'\nAdversarial example found, changing label to "{adversarial_label}" while perturbing pixels: {delta}')
        self.adversarial_results['adversarial_pixel_values'] = str(delta)
        self.adversarial_results['adversarial_example'] = str(adversarial_example)
        self.adversarial_results['adversarial_label'] = adversarial_label
