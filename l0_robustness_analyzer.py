# Reference: https://github.com/YuvShap/CoVerD/blob/main/tf_verify/calzone_robustness_analyzer.py
from utils import TQDM_BAR_FORMAT
from data import load_covering_sizes
from coverd_finite_geometries import get_possible_geometries, Geometry
from covering_generator.estimator import estimate_generator_speed, estimate_extra_time
from split_and_prune import estimate_SaP_speed, estimate_SaP_time, prepare4pruning, prune
from bimm import combs2bitmasks, bitmasks2combs, superset_sum
from verifier import Verifier
import threading
import queue
import math
import time
import random
import sys
from itertools import chain
from enum import Enum, auto
from collections import defaultdict
from multiprocessing.connection import Connection
import torch
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import norm
from tqdm import tqdm

CVD_MIN_T = 3
CVD_MAX_T = 6

class Strategy(Enum):
    CheSaP = auto()
    CCSaP  = auto()
    SCSaP  = auto()
    CoVerD = auto()

    def __str__(self) -> str:
        return self.name

class LZeroRobustnessAnalyzer:
    def __init__(self, verifier: Verifier, gpu_workers: list[Connection], cpu_workers: list[Connection], plan_only: bool, strategy: Strategy, until_timedout: bool, force_reestimate: bool):
        self.verifier = verifier
        self.__gpu_workers = gpu_workers
        self.__num_gpu_workers = len(gpu_workers)
        self.__cpu_workers = cpu_workers
        self.__num_cpu_workers = len(cpu_workers)
        self.__plan_only = plan_only
        self.__strategy = strategy
        self.__until_timedout = until_timedout
        self.__e, self.__v, self.__timeout, self.__channel_size = verifier.eps, verifier.num_pixels, verifier.timeout, verifier.bounded_x0.channel_size
        self.__refinement_db_max_v = verifier.max_k

        self.covering_sizes = {t:load_covering_sizes(t, clipping_size=500000) for t in range(CVD_MIN_T, min(self.__e + 1, CVD_MAX_T + 1))}
        generator_launch_time, skipping_iter_speed, normal_iter_speed = estimate_generator_speed(num_workers=self.__num_gpu_workers, num_pixels=self.__v, force_reestimate=force_reestimate)
        self.cvd_extra_time = lambda geometry: generator_launch_time + estimate_extra_time(geometry, num_workers=self.__num_gpu_workers, num_pixels=self.__v, pool_size=verifier.pool_size, skipping_iter_speed=skipping_iter_speed, iter_speed=normal_iter_speed)
        splitting_speed, pruning_speed = estimate_SaP_speed(epsilon=self.__e, num_indices=self.__v, force_reestimate=force_reestimate)
        self.estimate_SaP_time = lambda num_unk, level: estimate_SaP_time(num_unk, level, self.__v, splitting_speed[level], pruning_speed[level])
        self.estimate_pruning_time = lambda num_new_combs, num_references, level: num_new_combs * num_references / pruning_speed[level]
        self.mean_concretecheck_time: list[float]|None = None

    def update_image(self, image: np.ndarray, label: int):
        self.__deduplicated = False
        for gpu_worker in self.__gpu_workers:
            worker_seed = random.randint(0, 2 ** 32 - 1)
            gpu_worker.send((worker_seed, image, label))
        for cpu_worker in self.__cpu_workers:
            cpu_worker.send((image, label))

    def timed_out(self) -> bool:
        return time.time() - self.analyzer_start_time > self.__timeout

    def analyze(self):
        assert self.mean_concretecheck_time is not None, "mean_concretecheck_time should have been set by now."
        print('Planning stage started')
        self.analyzer_start_time = time.time()
        super_success_rate, mean_supercheck_time = {}, {}
        # Sample concrete nodes at all levels
        sample_counts, sample_successes = [0] * (self.__e + 1), [0] * (self.__e + 1)
        for gpu_worker in self.__gpu_workers:
            sc, ss = gpu_worker.recv()
            sample_counts = [a + b for a, b in zip(sample_counts, sc)]
            sample_successes = [a + b for a, b in zip(sample_successes, ss)]
        concrete_safe_rate = [s / c if c > 0 else 0.0 for s, c in zip(sample_successes, sample_counts)]
        if all(sample_successes[i] != math.comb(self.__v, i) * self.__num_gpu_workers for i in range(self.__e + 1)):
            # Sample super-nodes at levels with high concrete safe rate
            for l in range(self.__e + 1):
                if CVD_MIN_T <= l <= CVD_MAX_T and (concrete_safe_rate[l] > 0.99 or l == self.__e):
                    print(f'Sampling super-node on level {l}')
                    for gpu_worker in self.__gpu_workers:
                        gpu_worker.send(('sample-all-k', l))
                    self.__level = l
                    super_success_rate[l], mean_supercheck_time[l] = self.__estimate_sr_and_mt()
        estimated_CC_time_per_l = [0.0] * (self.__e + 1)
        estimated_SaP_time_per_l = [0.0] * (self.__e + 1)  # only estimate SaP time for levels >= 2, since level 0 and 1 are usually very fast and don't need pruning
        first_any_safe_level = 0
        # TODO: Add logging for estimated CC&SaP time
        for l in range(self.__e + 1):
            if concrete_safe_rate[l] == 0.0:
                first_any_safe_level += 1  # not to start on this level
                continue
            if l < 2:  # avoid estimating SaP time for level 0 and 1
                estimated_nodes_this_level = math.comb(self.__v, l)
            else:
                estimated_nodes_this_level = math.ceil(math.comb(self.__v, l) * (1 - concrete_safe_rate[l - 1]) ** 2)
                # print(f"Estimated concretes  on level {l}: {estimated_nodes_this_level} with concrete safe rate {concrete_safe_rate[l]:.4f}")
                if l < self.__e:
                    estimated_unk_nodes_this_level = estimated_nodes_this_level * (1 - concrete_safe_rate[l])
                    estimated_SaP_time_per_l[l] = self.estimate_SaP_time(estimated_unk_nodes_this_level, l)
                    # print(f"Estimated unknowns   on level {l}: {estimated_unk_nodes_this_level:.2f}")
                    # print(f"Estimated SaP   time on level {l}: {estimated_SaP_time_per_l[l]:.2f} seconds")
            estimated_CC_time_per_l[l] = self.mean_concretecheck_time[l] * estimated_nodes_this_level
            # print(f"Estimated check time on level {l}: {estimated_CC_time_per_l[l]:.2f} seconds")
            if concrete_safe_rate[l] == 1.0:
                break  # this level is estimated to be 100% robust and verified, no need to go deeper

        candidate_strategies: list[tuple[float, str, int]] = []
        candidate_geometries: list[dict[str, object]] = []
        # The keys are levels
        fraction_robusts: dict[int, dict[int, dict[int, float]]] = {}
        VTs: dict[int, dict[int, float]] = {}
        refinement_strategies: dict[int, dict[int, int|None]] = {}
        chosen_geometries: dict[int, Geometry] = {}
        chosen_geometries_info: dict[int, dict[str, object]] = {}

        choosing_strategy_start = time.time()
        for l in range(first_any_safe_level, self.__e + 1):
            if sample_successes[l] == math.comb(self.__v, l) * self.__num_gpu_workers:
                candidate_strategies.append((estimated_CC_time_per_l[l], 'skip', l))
                break
            tail_CC_time = sum(estimated_CC_time_per_l[l + 1:])
            tail_SaP_time = sum(estimated_SaP_time_per_l[l:])
            if tail_SaP_time > 1.0: tail_SaP_time /= self.__num_gpu_workers
            CCSaP_time = (self.mean_concretecheck_time[l] * math.comb(self.__v, l) + tail_CC_time) / self.__num_gpu_workers + tail_SaP_time
            candidate_strategies.append((CCSaP_time, 'CCSaP', l))
            if l in super_success_rate:
                self.__level = l
                fraction_robusts[l] = self.__compute_fraction_robust(super_success_rate[l])
                VTs[l], refinement_strategies[l] = self.__choose_strategy(fraction_robusts[l], mean_supercheck_time[l], super_success_rate[l])
                chosen_geometries[l], geometries_data = self.__choose_geometry(VTs[l])
                chosen_geometries_info[l] = geometries_data[0]
                candidate_geometries.extend(geometries_data)
                SCSaP_time: float = (chosen_geometries_info[l]["estimated_summed_duration"] + tail_CC_time) / self.__num_gpu_workers + self.cvd_extra_time(chosen_geometries[l]) + tail_SaP_time
                candidate_strategies.append((round(SCSaP_time, 3), 'SCSaP', l))
        choosing_strategy_duration = round(time.time() - choosing_strategy_start, 3)
        planning_duration = round(time.time() - self.analyzer_start_time, 3)
        candidate_strategies = sorted(map(lambda s: (round(s[0] + planning_duration, 3), s[1], s[2]), candidate_strategies), key=lambda s: s[0])
        if self.__strategy == Strategy.CCSaP:
            candidate_strategies = list(filter(lambda s: s[1] == 'CCSaP', candidate_strategies))
        elif self.__strategy == Strategy.SCSaP:
            candidate_strategies = list(filter(lambda s: s[1] == 'SCSaP', candidate_strategies))
        elif self.__strategy == Strategy.CoVerD:
            candidate_strategies = list(filter(lambda s: s[1] == 'SCSaP' and s[2] == self.__e, candidate_strategies))
        best_strategy = candidate_strategies[0]
        candidate_strategies = list(filter(lambda s: s[0] <= self.__timeout * 10, candidate_strategies))  # filter out noise strategies
        if len(candidate_strategies) == 0: candidate_strategies.append(best_strategy)  # at least have one strategy to show, even if it's an outlier
        self.__level = best_strategy[2]
        print(f"Planning stage took {planning_duration} s, choosing strategy duration: {choosing_strategy_duration} s, best strategy: {best_strategy}")
        print(f"Candidate strategies: {candidate_strategies}")

        results = {
            'verified': best_strategy[1] == 'skip',
            'timed_out': self.timed_out(),
            'choosing_strategy_duration': choosing_strategy_duration,
            'best_strategy': best_strategy,
            'estimated_verification_time': best_strategy[0],
            'starting_level': self.__level,
            'planning_duration': planning_duration,
            'candidate_strategies': candidate_strategies
        }
        if len(chosen_geometries) > 0:
            results['chosen_geometries'] = {l:str(g) for l,g in chosen_geometries.items()}
            results['candidate_geometries'] = candidate_geometries

        # to be deleted later if needed
        if best_strategy[1] == 'CCSaP' and best_strategy[2] >= 4 or best_strategy[0] > self.__timeout * 2 and not self.__until_timedout:
            results['timed_out'] = True
            print(f'{best_strategy[1]} at level {best_strategy[2]} estimated to take too long, skipping this image.')

        if results['verified'] or results['timed_out'] or self.__plan_only:
            results['running_time'] = round(time.time() - self.analyzer_start_time, 3)
            self.__skip_this_image()
            print('Early termination triggered')
            if results['verified']: results['end_level'] = results['starting_level']
            return results

        self.unk_combs_settled = threading.Event()
        self.waiting_suspects: set[tuple[int]] = set()
        self.innocent_suspects: set[tuple[int]] = set()
        self.milp_job_queue = queue.Queue()
        self.milp_found_adversarial = threading.Event()
        self.adversarial_results = {'adversarial_pixels': None, 'adversarial_example': None, 'adversarial_label': None}
        cpu_handler_thread = threading.Thread(target=self.__handle_cpu_workers_messages, args=(self.analyzer_start_time,))
        cpu_handler_thread.start()

        if best_strategy[1] == 'CCSaP':
            self.__deduplicated = True  # SaP won't create duplicates
            for gpu_worker in self.__gpu_workers:
                gpu_worker.send(('CCSaP', self.__level))  # no geometry, no refinement strategy
            current_level_start_time = time.time()
            # collect all unknown combinations
            unk_combs_lst: list[np.ndarray] = []
            for gpu_worker in self.__gpu_workers:
                unk_combs_lst.append(gpu_worker.recv())  # wait for gpu workers to finish
            unk_combs = np.vstack(unk_combs_lst)
            print(f'Level {self.__level} finished, unk_combs/all: {unk_combs.shape[0]}/{math.comb(self.__v, self.__level)}'
                f' using {round(time.time() - current_level_start_time, 3)} s')
        elif best_strategy[1] == 'SCSaP':
            self.__deduplicated = self.__level < self.__e  # SaP won't be used on the last level, so we might have duplicates on the last level
            chosen_geometry, refinement_strategy = chosen_geometries[self.__level], refinement_strategies[self.__level]
            assert self.__level == min(refinement_strategy.keys()), f"Level mismatch: {self.__level} vs {min(refinement_strategy.keys())}"
            chosen_indices = random.sample(range(chosen_geometry.number_of_points()), self.__v)
            random.shuffle(chosen_indices)
            for gpu_worker in self.__gpu_workers:
                gpu_worker.send(('SCSaP', chosen_geometry, chosen_indices, refinement_strategy))

            print(
                f'Chosen {chosen_geometries_info[self.__level]}; Operating on level: {self.__level}'
                # f"\nCandidate geometries: {candidate_geometries}"
            )
            refinement_sizes = {v: self.covering_sizes[self.__level][(v, k)] for v, k in refinement_strategy.items() if k is not None}
            supercheck_stats_by_size = defaultdict(lambda: {'num_checked': 0, 'successes': 0, 'total_duration': 0.})

            iterating_covering_start_time = time.time()
            unk_combs = self.__handle_gpu_workers_messages(supercheck_stats_by_size, chosen_geometry.number_of_blocks(), refinement_strategy, refinement_sizes)
            iterating_covering_duration = round(time.time() - iterating_covering_start_time, 3)
            summed_duration = round(sum(stats['total_duration'] for stats in supercheck_stats_by_size.values()), 3)
            print(f'GPU iterating coverings took {iterating_covering_duration} s, Summed duration (all gpu workers): {summed_duration} s', flush=True)
            if self.milp_job_queue.unfinished_tasks > 0:
                print(f'Waiting for MILP jobs to finish, remaining: {self.milp_job_queue.unfinished_tasks}, finished: {len(self.innocent_suspects)}', flush=True)
            results['SCSaP'] = {
                'chosen_geometry': chosen_geometries_info[self.__level],
                'iterating_covering_duration': iterating_covering_duration,
                'summed_duration': summed_duration
            }
            results['exec_details'] = {
                "Success Rate": {k:round(v, 5) for k, v in super_success_rate[self.__level].items()},
                "Mean Time": {k:round(v, 6) for k, v in mean_supercheck_time[self.__level].items()},
                "VT": {k:round(v, 6) for k, v in VTs[self.__level].items()},
                "R": refinement_strategy,
                "refinement_sizes": refinement_sizes,
                "fraction_robust": {v: round(fraction_robusts[self.__level][v][k], 5) for v, k in refinement_strategy.items() if k is not None},
                "num_checked": {size: stats['num_checked'] for size, stats in supercheck_stats_by_size.items()},
                "successes": {size: stats['successes'] for size, stats in supercheck_stats_by_size.items()},
                "success_rate": {size: round(stats['successes'] / stats['num_checked'], 5) if stats['num_checked'] > 0 else 0 for size, stats in supercheck_stats_by_size.items()},
                "total_duration": {size: round(stats['total_duration'], 6) for size, stats in supercheck_stats_by_size.items()},
                "mean_duration": {size: round(stats['total_duration'] / stats['num_checked'], 6) if stats['num_checked'] > 0 else 0 for size, stats in supercheck_stats_by_size.items()},
            }
        else: assert False, f'Unknown best strategy: {best_strategy}'

        handle_unk_combs_start = time.time()
        self.__handle_unk_combs(unk_combs, deduplicate=best_strategy[1] == 'SCSaP')
        self.unk_combs_settled.set()
        handle_unk_combs_duration = round(time.time() - handle_unk_combs_start, 3)
        if handle_unk_combs_duration > 0.01:
            print(f'Handling unk_combs duration: {handle_unk_combs_duration} s.')
            results['handle_unk_combs_duration'] = handle_unk_combs_duration

        waiting_milp_start = time.time()
        cpu_handler_thread.join()
        waiting_milp_duration = round(time.time() - waiting_milp_start, 3)
        results['timed_out'] = self.timed_out()
        results['running_time'] = round(time.time() - self.analyzer_start_time, 3)
        if self.milp_job_queue.unfinished_tasks > 0:
            print(f'MILP jobs unfinished: {self.milp_job_queue.unfinished_tasks}, finished: {len(self.innocent_suspects)}')
        if waiting_milp_duration > 0.1:
            print(f'Waiting for MILP duration: {waiting_milp_duration} s.')
            results['waiting_milp_duration'] = waiting_milp_duration
        if self.milp_found_adversarial.is_set():
            results['adversarial_results'] = self.adversarial_results
        elif not results['timed_out']:
            results['verified'] = True
        if len(self.innocent_suspects) + len(self.waiting_suspects) > 0:
            results['proved_with_milp'] = len(self.innocent_suspects)
            results['waiting_for_milp'] = len(self.waiting_suspects)
        results['end_level'] = self.__level
        self.__stop_workers()
        return results

    def __estimate_sr_and_mt(self) -> tuple[dict[int, float], dict[int, float]]:
        sampling_lower_bound = self.__level
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

        return {k + self.__level:v for k, v in enumerate(sr_vector)}, {k + self.__level:v for k, v in enumerate(mean_time_vector)}

    def __compute_fraction_robust(self, success_rate: dict[int, float]) -> dict[int, dict[int, float]]:
        fraction_robust: dict[int, dict[int, float]] = dict()
        for v in range(self.__level, self.__refinement_db_max_v + 1):
            fraction_robust[v] = dict()
            for k in range(self.__level, v):
                if 1 - success_rate[v] < 1e-7:
                    fraction_robust[v][k] = 1.0
                else:
                    estimated_value = (success_rate[k] - success_rate[v]) / (1 - success_rate[v])
                    fraction_robust[v][k] = min(1, max(estimated_value, 0))

        return fraction_robust

    def __choose_strategy(self,
        fraction_robust: dict[int, dict[int, float]],
        mean_time: dict[int, float],
        success_rate: dict[int, float]
    ) -> tuple[dict[int,float], dict[int,int|None]]:
        RT: dict[int,float] = {self.__level: 0.}  # estimated refinement time
        VT: dict[int,float] = {self.__level: mean_time[self.__level]}  # estimated verification time
        refinement_strategy: dict[int,int|None] = {self.__level: None}  # best next k or None
        for v in range(self.__level + 1, self.__refinement_db_max_v + 1):
            best_next_k = None
            min_refine_time = float('inf')
            for k in range(self.__level, min(v, self.__refinement_db_max_v + 1)):
                if (v, k) not in self.covering_sizes[self.__level]:
                    continue
                refine_time = self.covering_sizes[self.__level][(v, k)] * (mean_time[k] + (1 - fraction_robust[v][k]) * RT[k])
                if refine_time < min_refine_time:
                    best_next_k = k
                    min_refine_time = refine_time
            RT[v] = min_refine_time
            refinement_strategy[v] = best_next_k
            VT[v] = mean_time[v] + min_refine_time * (1 - success_rate[v])
        return VT, refinement_strategy

    def __choose_geometry(self, VT: dict[int,float]) -> tuple[Geometry, list[dict[str, object]]]:
        best_geometry: Geometry = None
        best_time = float('inf')
        geometries = []
        for geometry in get_possible_geometries(self.__v, self.__level):
            estimated_time = 0
            pseudo_k = geometry.block_size() * self.__v / geometry.number_of_points()  # mean block size
            std_dev = math.sqrt(pseudo_k * (1 - pseudo_k + (geometry.block_size() - 1) * (self.__v - 1) / (geometry.number_of_points() - 1)))
            total_number_of_blocks = geometry.number_of_blocks()

            if (1 - norm.cdf(self.__refinement_db_max_v, pseudo_k, std_dev)) * total_number_of_blocks > 1e-2:
                continue

            cdf_before = norm.cdf(self.__level - 0.5, pseudo_k, std_dev)
            for block_size in range(self.__level, self.__refinement_db_max_v + 1):
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
        statistics_by_size: dict,
        initial_covering_size: int,
        refinement_strategy: dict[int,int|None],
        refinement_sizes: dict[int,int]
    ) -> np.ndarray:
        num_finished = 0
        done_count = 0
        total_covering_size = initial_covering_size
        pbar = tqdm(total=initial_covering_size, desc="SuperChecking", bar_format=TQDM_BAR_FORMAT, file=sys.stdout)
        last_print_time = time.time()
        unk_combs_lst: list[list[int]] = []
        while done_count < self.__num_gpu_workers \
            and not self.milp_found_adversarial.is_set() \
            and not self.timed_out():
            last_print_time = self.__print_progress(pbar, total_covering_size, num_finished, last_print_time)

            for gpu_worker in self.__gpu_workers:
                if gpu_worker.poll():
                    message = gpu_worker.recv()
                    if isinstance(message, str) and message == 'adversarial-example-suspects':
                        for suspect in gpu_worker.recv().tolist():
                            self.milp_job_queue.put(suspect)
                    elif isinstance(message, str) and message == 'unknown-combs':
                        unk_combs_lst.append(gpu_worker.recv())
                    elif isinstance(message, str) and message == 'done':
                        done_count += 1
                    else:  # update statistics
                        num_indices, num_checked, num_verified, duration = message
                        size_statistics = statistics_by_size[num_indices]
                        if num_indices < self.__level:
                            num_finished += num_verified
                        elif refinement_strategy[num_indices] is not None:  # could potentially be refined
                            num_finished += num_checked
                            size_statistics['num_checked'] += num_checked
                            size_statistics['successes'] += num_verified
                            size_statistics['total_duration'] += duration
                            num_failed = num_checked - num_verified
                            if num_failed > 0:  # to be refined later, add to total covering size
                                total_covering_size += refinement_sizes[num_indices] * num_failed
                        else:  # handled by CCSaP verifier, could be in a batch
                            num_finished += num_checked
                            size_statistics['num_checked'] += num_checked
                            size_statistics['total_duration'] += duration
                            size_statistics['successes'] += num_verified
        pbar.close()
        print(f"num_finished: {num_finished}, total_covering_size: {total_covering_size}, remaining: {total_covering_size - num_finished}")
        return np.vstack(unk_combs_lst) if len(unk_combs_lst) > 0 else np.empty((0, self.__v), dtype=int)

    def __handle_unk_combs(self, unk_combs: np.ndarray, deduplicate: bool = False):
        if unk_combs.shape[0] == 0 or self.timed_out(): return
        unk_combs_tensor: torch.Tensor = torch.from_numpy(unk_combs)
        if deduplicate:
            print(f"Level {self.__level} Before deduplication, unk_combs size: {unk_combs.shape[0]}", end="; ")
            unk_combs_tensor = torch.unique(unk_combs_tensor.cuda(), dim=0).cpu()
            print(f'After deduplication, unk_combs size: {unk_combs_tensor.shape[0]}')
        while unk_combs_tensor.shape[0] > 0 and self.__level < self.__e and not self.timed_out():
            new_level_start_time = time.time()
            if self.__level > 1 and self.estimate_SaP_time(unk_combs_tensor.shape[0], self.__level) > 1.0:
                print(f'Level {self.__level} estimated SaP time: {self.estimate_SaP_time(unk_combs_tensor.shape[0], self.__level):3f} s. Sharing SaP workload across all GPU workers...')
                for i, gpu_worker in enumerate(self.__gpu_workers):
                    gpu_worker.send(('SaP&CC', unk_combs_tensor.numpy()))
                num_new_combs_per_worker = [0] * self.__num_gpu_workers
                for i, gpu_worker in enumerate(self.__gpu_workers):
                    num_new_combs_per_worker[i] = gpu_worker.recv()
                concretecheck_start_time = time.time()
                SaP_duration = round(concretecheck_start_time - new_level_start_time, 3)
                self.__level += 1
                num_pruned = sum(num_new_combs_per_worker)
                print(f'Current level: {self.__level}, SaP duration: {SaP_duration}s, num_pruned: {num_pruned}, each worker got: {num_new_combs_per_worker}')
            else:
                new_combs_tensor, reference_bitmasks, target_s_cnt = prepare4pruning(unk_combs_tensor, self.__v)
                if new_combs_tensor.shape[0] == 0: return
                pruned_combs = prune(new_combs_tensor, reference_bitmasks, target_s_cnt, self.__level).numpy()
                if pruned_combs.shape[0] == 0: return
                concretecheck_start_time = time.time()
                SaP_duration = round(concretecheck_start_time - new_level_start_time, 3)
                self.__level += 1
                num_pruned = pruned_combs.shape[0]
                print(f'Current level: {self.__level}, SaP duration: {SaP_duration}s, num_pruned: {num_pruned}')
                for i, gpu_worker in enumerate(self.__gpu_workers):
                    num_each = pruned_combs.shape[0] // self.__num_gpu_workers
                    if i == self.__num_gpu_workers - 1:
                        gpu_worker.send(('CC', pruned_combs[i * num_each :]))
                    else:
                        gpu_worker.send(('CC', pruned_combs[i * num_each : (i + 1) * num_each]))
            unk_combs_lst: list[np.ndarray] = []
            for gpu_worker in self.__gpu_workers:
                unk_combs_lst.append(gpu_worker.recv())  # wait for gpu workers to finish
            unk_combs_tensor = torch.from_numpy(np.vstack(unk_combs_lst))
            print(f'Level {self.__level} finished, unk_combs/all: {unk_combs_tensor.shape[0]}/{num_pruned}'
                f' using {round(time.time() - new_level_start_time, 3)} s, checking duration: {round(time.time() - concretecheck_start_time, 3)} s')
        if self.__level == self.__e and unk_combs_tensor.shape[0] > 0:
            print(f'Checked deepest level {self.__e}, remaining unk_combs size: {unk_combs_tensor.shape[0]}')
            for combs in unk_combs_tensor.tolist():
                self.milp_job_queue.put(combs)

    def __print_progress(self, pbar: tqdm, total_covering_size: int, num_finished: int, last_print_time: float) -> float:
        current = time.time()
        if current - last_print_time < 2:
            return last_print_time
        pbar.n = pbar.total - total_covering_size + num_finished
        pbar.set_postfix({'FNS/AGG': f'{num_finished}/{int(total_covering_size)}'})
        return current

    def __deduplicate_milp_jobs(self):
        # Deduplicate suspects in the jobs queue before showing progress bar.
        queued_jobs: list[list[int]] = []
        while True:
            try:
                queued_jobs.append(self.milp_job_queue.get_nowait())
                self.milp_job_queue.task_done()
            except queue.Empty:
                break
        if len(queued_jobs) > 0:
            known_suspects = self.innocent_suspects.union(self.waiting_suspects)
            unique_jobs = torch.tensor(queued_jobs, device="cuda").unique(dim=0)
            if len(known_suspects) > 0:
                bitmask_length = (self.__v + 63) // 64
                known_suspects_bm = combs2bitmasks(torch.tensor(list(known_suspects), device="cuda"), bitmask_length)
                unique_jobs_bm = combs2bitmasks(unique_jobs, bitmask_length)
                unique_jobs = bitmasks2combs(unique_jobs_bm[superset_sum(unique_jobs_bm, known_suspects_bm)==0], self.__e).tolist()
            removed = len(queued_jobs) - len(unique_jobs)
            for job in unique_jobs:
                self.milp_job_queue.put(job)
            if removed > 0:
                assert self.milp_job_queue.unfinished_tasks == len(unique_jobs) + len(self.waiting_suspects), f"Unfinished tasks in queue {self.milp_job_queue.unfinished_tasks} doesn't match expected {len(unique_jobs) + len(self.waiting_suspects)} after deduplication."
                print(f"Deduplicated MILP jobs queue, removed {removed} duplicates, remaining jobs: {self.milp_job_queue.unfinished_tasks}")
        self.__deduplicated = True

    def __handle_cpu_workers_messages(self, analyzer_start_time: float):
        jobs_limit = 3
        jobs_each_worker = np.zeros(self.__num_cpu_workers, dtype=int)
        def get_first_avail_worker_id():
            min_jobs = int(jobs_each_worker.min())
            if min_jobs >= jobs_limit:
                return None
            return int(jobs_each_worker.argmin())
        pbar = None
        while not self.unk_combs_settled.is_set() or self.milp_job_queue.unfinished_tasks > 0:
            if pbar is None and self.unk_combs_settled.is_set():
                if not self.__deduplicated:
                    self.__deduplicate_milp_jobs()
                pbar = tqdm(total=self.milp_job_queue.unfinished_tasks, desc="MILP jobs", bar_format=TQDM_BAR_FORMAT, mininterval=1.0, maxinterval=2.0, file=sys.stdout)
            first_avail_worker_id = get_first_avail_worker_id()
            if first_avail_worker_id is not None:
                try:
                    pixel_indices = self.milp_job_queue.get(timeout=0.0001)
                    suspect = tuple(pixel_indices)
                    if not self.__deduplicated and (suspect in self.innocent_suspects or suspect in self.waiting_suspects):
                        self.milp_job_queue.task_done()
                    else:
                        self.waiting_suspects.add(suspect)
                        self.__cpu_workers[first_avail_worker_id].send(pixel_indices)
                        jobs_each_worker[first_avail_worker_id] += 1
                except queue.Empty:
                    ...
            if time.time() - analyzer_start_time >= self.__timeout:
                if pbar is not None: pbar.close()
                return

            for i, cpu_worker in enumerate(self.__cpu_workers):
                if cpu_worker.poll():
                    pixel_indices, verified, adversarial_example, adversarial_label = cpu_worker.recv()
                    self.milp_job_queue.task_done()
                    jobs_each_worker[i] -= 1
                    if pbar is not None: pbar.update(1)
                    if verified:
                        self.waiting_suspects.remove(tuple(pixel_indices))
                        self.innocent_suspects.add(tuple(pixel_indices))
                    else:
                        self.milp_found_adversarial.set()
                        self.adversarial_results['adversarial_pixels'] = str(pixel_indices)
                        # adversarial_example: list[int]
                        # pixel_indices: list[int]
                        if self.__channel_size > 1:
                            indices = []
                            for pixel in pixel_indices:
                                for i in range(self.__channel_size):
                                    indices.append(pixel * self.__channel_size + i)
                            pixel_indices = indices
                        delta_dict = dict((i, adversarial_example[i]) for i in pixel_indices)
                        print(f'\nAdversarial example found, changing label to "{adversarial_label}" while perturbing pixels: {delta_dict}')
                        self.adversarial_results['adversarial_pixel_values'] = str(delta_dict)
                        self.adversarial_results['adversarial_example'] = str(adversarial_example)
                        self.adversarial_results['adversarial_label'] = adversarial_label
                        if pbar is not None: pbar.close()
                        return

    def __stop_workers(self):
        for worker in chain(self.__gpu_workers, self.__cpu_workers):
            worker.send('stop')
        for worker in chain(self.__gpu_workers, self.__cpu_workers):
            message = worker.recv()
            while not isinstance(message, str) or message != 'stopped':
                message = worker.recv()

    def __skip_this_image(self):
        for gpu_worker in self.__gpu_workers:
            gpu_worker.send(('skip', 0))
        for cpu_worker in self.__cpu_workers:
            cpu_worker.send('stop')
        for cpu_worker in self.__cpu_workers:
            message = cpu_worker.recv()
            while not isinstance(message, str) or message != 'stopped':
                message = cpu_worker.recv()