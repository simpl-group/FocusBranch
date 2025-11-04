# Reference: https://github.com/YuvShap/CoVerD/blob/main/tf_verify/calzone_robustness_analyzer.py
import torch
from data import load_covering_sizes
from coverd_finite_geometries import get_possible_geometries, Geometry
from split_and_prune import split_and_prune_by_unk
import threading
import queue
import math
import time
import random
from itertools import chain
from collections import defaultdict
from multiprocessing.connection import Connection
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import norm

def decreasing_exp(x, a, b):
    return 1 - np.exp(a*(x-b))

class LZeroRobustnessAnalyzer:
    def __init__(self, gpu_workers: list[Connection], cpu_workers: list[Connection], t: int, timeout: int, max_k: int, num_pixels: int):
        self.__gpu_workers = gpu_workers
        self.__cpu_workers = cpu_workers
        self.__t = t
        self.__timeout = timeout
        self.__number_of_pixels = num_pixels
        self.__refinement_db_max_v = max_k
        self.covering_sizes = load_covering_sizes(self.__t, clipping_size=500000) if 3 <= t <= 6 else dict()

    def update_image(self, image: np.ndarray, label: int):
        for gpu_worker in self.__gpu_workers:
            worker_seed = random.randint(0, 2 ** 32 - 1)
            gpu_worker.send((worker_seed, image, label))
        for cpu_worker in self.__cpu_workers:
            cpu_worker.send((image, label))

    def __finalize_first_safe_level(self) -> int:
        fsls = []
        for gpu_worker in self.__gpu_workers:
            fsls.append(gpu_worker.recv())
        self.__level = max(fsls)
        for gpu_worker in self.__gpu_workers:
            gpu_worker.send(self.__level)
        return self.__level

    def analyze(self):
        print('Planning stage started')
        analyzer_start_time = time.time()
        self.__finalize_first_safe_level()
        print(f'First safe level determined: {self.__level}', end='; ')
        # Planning stage
        if self.__level < 2 and self.__level < self.__t:
            for cpu_worker in self.__cpu_workers:
                cpu_worker.send('stop')
            print('Early termination triggered')
            return {'verified': True, 'timed_out': False, 'end_level': self.__level}

        self.unk_combs_settled = threading.Event()
        self.waiting_suspects: set[tuple[int]] = set()
        self.innocent_suspects: set[tuple[int]] = set()
        self.milp_job_queue = queue.Queue()
        self.milp_found_adversarial = threading.Event()
        self.adversarial_results = {'adversarial_pixels': None, 'adversarial_example': None, 'adversarial_label': None}
        cpu_handler_thread = threading.Thread(target=self.__handle_cpu_workers_messages, args=(analyzer_start_time,))
        cpu_handler_thread.start()
        # l=1, t=1 or l=2, t=2
        if self.__level == self.__t <= 2:
            planning_duration = round(time.time() - analyzer_start_time, 3)
            print(f"Planning stage took {planning_duration} s")
            self.__level = 1
            # collect all unknown combinations
            unk_combs_lst: list[np.ndarray] = []
            for gpu_worker in self.__gpu_workers:
                unk_combs_lst.append(gpu_worker.recv())  # wait for gpu workers to finish
            unk_combs = np.vstack(unk_combs_lst)
            print(f'Level {self.__level} finished, unk_combs/all: {unk_combs.shape[0]}/{math.comb(self.__number_of_pixels, self.__level)}'
                f' using {round(time.time() - analyzer_start_time, 3)} s')
            results = {'verified': True, 'timed_out': False}
        else:
            self.__level = max(self.__level, 3)
            print(f'Operating on level {self.__level}')
            success_rate, mean_time = self.__estimate_sr_and_mt()
            fraction_robust = self.__compute_fraction_robust(success_rate)
            choosing_strategy_start = time.time()
            VT, refinement_strategy = self.__choose_strategy(fraction_robust, mean_time, success_rate)
            chosen_geometry, geometries_data = self.__choose_geometry(VT)
            estimated_verification_time = round(geometries_data[0]["estimated_summed_duration"] / len(self.__gpu_workers), 3)
            choosing_strategy_duration = round(time.time() - choosing_strategy_start, 3)
            print(
                f'Chosen {geometries_data[0]}\n'
                f'Estimated verification time {estimated_verification_time} s, '
                f'choosing strategy duration: {choosing_strategy_duration} s'
            )
            planning_duration = round(time.time() - analyzer_start_time, 3)
            print(f"Planning stage took {planning_duration} s")
            refinement_sizes = {v: self.covering_sizes[(v, k)] for v, k in refinement_strategy.items() if k is not None}
            fr = {v: round(fraction_robust[v][k], 5) for v, k in refinement_strategy.items() if k is not None}

            chosen_indices = random.sample(range(chosen_geometry.number_of_points()), self.__number_of_pixels)
            random.shuffle(chosen_indices)
            for gpu_worker in self.__gpu_workers:
                gpu_worker.send((chosen_geometry, chosen_indices, refinement_strategy))
            blocks_stats_by_size = defaultdict(lambda: {'runs': 0, 'successes': 0, 'total_duration': 0.})

            iterating_covering_start_time = time.time()
            unk_combs = self.__handle_gpu_workers_messages(analyzer_start_time, blocks_stats_by_size, chosen_geometry.number_of_blocks(), refinement_strategy, refinement_sizes)
            iterating_covering_duration = round(time.time() - iterating_covering_start_time, 3)
            summed_duration = round(sum(stats['total_duration'] for stats in blocks_stats_by_size.values()), 3)
            print(f'GPU iterating coverings took {iterating_covering_duration} s, Summed duration (all gpu workers): {summed_duration} s')
            results = {
                'verified': True, 'timed_out': False,
                'choosing_strategy_duration': choosing_strategy_duration,
                'planning_duration': planning_duration,
                'candidate geometries': geometries_data[1:],
                'chosen_geometry': geometries_data[0],
                'estimated_verification_time': estimated_verification_time,
                'iterating_covering_duration': iterating_covering_duration,
                'summed_duration': summed_duration,
                'exec_details': {
                    "Success Rate": {k:round(v, 5) for k, v in success_rate.items()},
                    "Mean Time": {k:round(v, 6) for k, v in mean_time.items()},
                    "VT": {k:round(v, 6) for k, v in VT.items()},
                    "R": refinement_strategy,
                    "refinement_sizes": refinement_sizes,
                    "fraction_robust": fr,
                    "runs": {size: stats['runs'] for size, stats in blocks_stats_by_size.items()},
                    "successes": {size: stats['successes'] for size, stats in blocks_stats_by_size.items()},
                    "success_rate": {size: round(stats['successes'] / stats['runs'], 5) if stats['runs'] > 0 else 0 for size, stats in blocks_stats_by_size.items()},
                    "total_duration": {size: round(stats['total_duration'], 6) for size, stats in blocks_stats_by_size.items()},
                    "mean_duration": {size: round(stats['total_duration'] / stats['runs'], 6) if stats['runs'] > 0 else 0 for size, stats in blocks_stats_by_size.items()},
                }
            }

        handle_unk_combs_start = time.time()
        self.__handle_unk_combs(analyzer_start_time, unk_combs)
        handle_unk_combs_duration = round(time.time() - handle_unk_combs_start, 3)
        waiting_milp_start = time.time()
        cpu_handler_thread.join()
        waiting_milp_duration = round(time.time() - waiting_milp_start, 3)

        if handle_unk_combs_duration > 0.1:
            print(f'Handling unk_combs duration: {handle_unk_combs_duration} s.')
            results['handle_unk_combs_duration'] = handle_unk_combs_duration
        if waiting_milp_duration > 0.1:
            print(f'Waiting for MILP duration: {waiting_milp_duration} s.')
            results['waiting_milp_duration'] = waiting_milp_duration
        if time.time() - analyzer_start_time >= self.__timeout:
            results['verified'] = False
            results['timed_out'] = True
        if self.milp_found_adversarial.is_set():
            results['verified'] = False
            results['adversarial_results'] = self.adversarial_results
        if len(self.innocent_suspects) + len(self.waiting_suspects) > 0:
            results['proved_with_milp'] = len(self.innocent_suspects)
            results['waiting_for_milp'] = len(self.waiting_suspects)
        results['end_level'] = self.__level
        self.__stop_workers()
        return results

    def __estimate_sr_and_mt(self) -> tuple[dict[int, float], dict[int, float]]:
        sampling_lower_bound = self.__t
        sampling_upper_bound = self.__refinement_db_max_v

        sampling_counts_vector = np.zeros(sampling_upper_bound - sampling_lower_bound + 1)
        sampling_successes_vector = np.zeros(sampling_upper_bound - sampling_lower_bound + 1)
        sampling_time_vector = np.zeros(sampling_upper_bound - sampling_lower_bound + 1)
        for gpu_worker in self.__gpu_workers:
            sampling_counts, sampling_successes, sampling_time = gpu_worker.recv()
            sampling_counts_vector += np.array(sampling_counts)
            sampling_successes_vector += np.array(sampling_successes)
            sampling_time_vector += np.array(sampling_time)

        success_ratio_vector = sampling_successes_vector / sampling_counts_vector
        mean_time_vector = sampling_time_vector / sampling_counts_vector

        sr_vector = savgol_filter(success_ratio_vector, 15, 2).clip(0, 1)

        return {k + self.__t:v for k, v in enumerate(sr_vector)}, {k + self.__t:v for k, v in enumerate(mean_time_vector)}

    def __compute_fraction_robust(self, success_rate: dict[int, float]) -> dict[int, dict[int, float]]:
        fraction_robust: dict[int, dict[int, float]] = dict()
        for v in range(self.__t, self.__refinement_db_max_v + 1):
            fraction_robust[v] = dict()
            for k in range(self.__t, v):
                if 1 - success_rate[v] < 1e-7:
                    fraction_robust[v][k] = 1.0
                else:
                    estimated_value = (success_rate[k] - success_rate[v]) / (1 - success_rate[v])
                    fraction_robust[v][k] = min(1, max(estimated_value, 0))

        return fraction_robust

    def __choose_strategy(self,
        fraction_robust: dict[int, dict[int, float]],
        mean_time: dict[int, float],
        success_rate: dict[int, float],
        sbap_VT: dict[int,float]|None=None
    ) -> tuple[dict[int,float], dict[int,int|None]]:
        RT: dict[int,float] = {self.__t: 0.}  # estimated refinement time
        VT: dict[int,float] = {self.__t: mean_time[self.__t]}  # estimated verification time
        refinement_strategy: dict[int,int|None] = {self.__t: None}  # best next k or None
        for v in range(self.__t + 1, self.__refinement_db_max_v + 1):
            best_next_k = None
            min_refine_time = float('inf')
            for k in range(self.__t, min(v, self.__refinement_db_max_v + 1)):
                if (v, k) not in self.covering_sizes:
                    continue
                refine_time = self.covering_sizes[(v, k)] * (mean_time[k] + (1 - fraction_robust[v][k]) * RT[k])
                if refine_time < min_refine_time:
                    best_next_k = k
                    min_refine_time = refine_time
            RT[v] = min_refine_time
            refinement_strategy[v] = best_next_k
            VT[v] = mean_time[v] + min_refine_time * (1 - success_rate[v])
            if sbap_VT is not None and v in sbap_VT and sbap_VT[v] * 0.95 <= VT[v]:  # tunable coefficient
                VT[v] = sbap_VT[v]
                RT[v] = 0.
                refinement_strategy[v] = None
        return VT, refinement_strategy

    def __choose_geometry(self, VT: dict[int,float]) -> tuple[Geometry, list[tuple[str, float, int, float]]]:
        best_geometry = None
        best_time = float('inf')
        geometries = []
        for geometry in get_possible_geometries(self.__number_of_pixels, self.__level):
            estimated_time = 0
            pseudo_k = geometry.block_size() * self.__number_of_pixels / geometry.number_of_points()  # mean block size
            std_dev = math.sqrt(pseudo_k * (1 - pseudo_k + (geometry.block_size() - 1) * (self.__number_of_pixels - 1) / (geometry.number_of_points() - 1)))
            total_number_of_blocks = geometry.number_of_blocks()

            if (1 - norm.cdf(self.__refinement_db_max_v, pseudo_k, std_dev)) * total_number_of_blocks > 1e-2:
                continue

            cdf_before = norm.cdf(self.__t - 0.5, pseudo_k, std_dev)
            for block_size in range(self.__t, self.__refinement_db_max_v + 1):
                cdf_after = norm.cdf(block_size + 0.5, pseudo_k, std_dev)
                block_size_probability = cdf_after - cdf_before
                cdf_before = cdf_after
                expected_number_of_blocks_of_current_size = block_size_probability * total_number_of_blocks
                number_of_blocks_of_current_size = math.floor(expected_number_of_blocks_of_current_size)
                if random.random() < expected_number_of_blocks_of_current_size - number_of_blocks_of_current_size:
                    number_of_blocks_of_current_size += 1
                estimated_time += number_of_blocks_of_current_size * VT[block_size]

            geometries.append({str(geometry): f"{geometry.number_of_blocks()} blocks, mean size {round(pseudo_k, 3)}",
                               "estimated_summed_duration": round(float(estimated_time), 3)})
            if estimated_time < best_time:
                best_time = estimated_time
                best_geometry = geometry
        return best_geometry, sorted(geometries, key=lambda x: x["estimated_summed_duration"])[:3]  # return top 3 geometries sorted by estimated time

    def __handle_gpu_workers_messages(
        self,
        analyzer_start_time: float,
        statistics_by_size: dict,
        initial_covering_size: int,
        refinement_strategy: dict[int,int|None],
        refinement_sizes: dict[int,int]
    ):
        num_finished = 0
        done_count = 0
        total_covering_size = initial_covering_size
        last_print_time = time.time()
        unk_combs_lst: list[list[int]] = []
        while done_count < len(self.__gpu_workers) \
            and not self.milp_found_adversarial.is_set() \
            and time.time() - analyzer_start_time < self.__timeout:
            last_print_time = self.__print_progress(total_covering_size, last_print_time, num_finished)

            for gpu_worker in self.__gpu_workers:
                if gpu_worker.poll():
                    message = gpu_worker.recv()
                    if message == 'adversarial-example-suspect':
                        self.milp_job_queue.put(gpu_worker.recv())
                    elif message == 'concrete-combs':
                        unk_combs_lst.append(gpu_worker.recv())
                    elif message == 'done':
                        done_count += 1
                    else:  # update statistics
                        num_indices, runs, num_verified, duration = message
                        size_statistics = statistics_by_size[num_indices]
                        if num_indices < self.__t:
                            num_finished += num_verified
                        elif refinement_strategy[num_indices] is not None:  # could potentially be refined
                            num_finished += runs
                            size_statistics['runs'] += runs
                            size_statistics['successes'] += num_verified
                            size_statistics['total_duration'] += duration
                            num_failed = runs - num_verified
                            if num_failed > 0:  # to be refined later, add to total covering size
                                total_covering_size += refinement_sizes[num_indices] * num_failed
                        else:  # handled by sbap verifier, could be in a batch
                            num_finished += runs
                            size_statistics['runs'] += runs
                            size_statistics['total_duration'] += duration
                            size_statistics['successes'] += num_verified
        print(f"num_finished: {num_finished}, total_covering_size: {total_covering_size}, remaining: {total_covering_size - num_finished}")
        return np.array(unk_combs_lst) if len(unk_combs_lst) > 0 else np.empty((0, self.__number_of_pixels), dtype=int)

    def __handle_unk_combs(self, analyzer_start_time: float, unk_combs: np.ndarray):
        while unk_combs.shape[0] > 0 and self.__level < self.__t and time.time() - analyzer_start_time < self.__timeout:
            new_combs = split_and_prune_by_unk(torch.from_numpy(unk_combs).long(), self.__level, self.__number_of_pixels).numpy()
            self.__level += 1
            print(f'Current level: {self.__level}, new_combs size: {new_combs.shape[0]}')
            for i, gpu_worker in enumerate(self.__gpu_workers):
                num_each = new_combs.shape[0] // len(self.__gpu_workers)
                if i == len(self.__gpu_workers) - 1:
                    gpu_worker.send(new_combs[i * num_each : ])
                else:
                    gpu_worker.send(new_combs[i * num_each : (i + 1) * num_each])
            unk_combs_lst: list[np.ndarray] = []
            for gpu_worker in self.__gpu_workers:
                unk_combs_lst.append(gpu_worker.recv())  # wait for gpu workers to finish
            unk_combs = np.unique(np.vstack(unk_combs_lst), axis=0)
        if self.__level == self.__t and unk_combs.shape[0] > 0:
            print(f'Checked deepest level {self.__t}, remaining unk_combs size: {unk_combs.shape[0]}')
            for combs in unk_combs.tolist():
                self.milp_job_queue.put(combs)
        self.unk_combs_settled.set()

    def __print_progress(self, total_covering_size, last_print_time, num_finished):
        current = time.time()
        if current - last_print_time < 2:
            return last_print_time
        print(f'progress: {num_finished}/{int(total_covering_size)}=' \
            f'{100 * num_finished / total_covering_size:.3f}%', sep='; ', end='\r')
        return current

    def __handle_cpu_workers_messages(self, analyzer_start_time: float):
        next_cpu_worker_id = 0
        while not self.unk_combs_settled.is_set() or self.milp_job_queue.unfinished_tasks > 0:
            try:
                pixel_indices = self.milp_job_queue.get(timeout=0.1)
                suspect = tuple(pixel_indices)
                if suspect in self.innocent_suspects or suspect in self.waiting_suspects:
                    self.milp_job_queue.task_done()
                else:
                    self.waiting_suspects.add(suspect)
                    self.__cpu_workers[next_cpu_worker_id].send(pixel_indices)
                    next_cpu_worker_id = (next_cpu_worker_id + 1) % len(self.__cpu_workers)
            except queue.Empty:
                ...
            if time.time() - analyzer_start_time >= self.__timeout:
                return

            for cpu_worker in self.__cpu_workers:
                if cpu_worker.poll():
                    pixel_indices, verified, adversarial_example, adversarial_label = cpu_worker.recv()
                    self.milp_job_queue.task_done()
                    if verified:
                        self.waiting_suspects.remove(tuple(pixel_indices))
                        self.innocent_suspects.add(tuple(pixel_indices))
                    else:
                        self.milp_found_adversarial.set()
                        print(f'\nAdversarial example found, changing label to "{adversarial_label}" while perturbing pixels: {pixel_indices}')
                        self.adversarial_results['adversarial_pixels'] = pixel_indices
                        self.adversarial_results['adversarial_example'] = str(adversarial_example)
                        self.adversarial_results['adversarial_label'] = adversarial_label
                        return

    def __stop_workers(self):
        for worker in chain(self.__gpu_workers, self.__cpu_workers):
            worker.send('stop')
        for worker in chain(self.__gpu_workers, self.__cpu_workers):
            message = worker.recv()
            while message != 'stopped':
                message = worker.recv()
