# Reference: https://github.com/YuvShap/CoVerD/blob/main/tf_verify/calzone_robustness_analyzer.py
from utils import TQDM_BAR_FORMAT
from data import load_covering_sizes, Strategy
from coverd_finite_geometries import Geometry
from covering_generator import CVD_MIN_T, CVD_MAX_T
from covering_generator.estimator import estimate_generator_speed, estimate_extra_time
from split_and_prune import estimate_SaP_speed, estimate_SaP_time, prepare4pruning, prune, canonical_split
from milp_dispatcher import MilpDispatcher
from verifier import Verifier
import planner
import math
import time
import random
import sys
from collections import defaultdict
from multiprocessing.connection import Connection
import torch
import numpy as np
from tqdm import tqdm

SR_THRES_TO_SAMPLE_SUPER = [0., 0., 0.7, 0.97, 0.9997,0.99999,0.999999,0.9999999,0.99999999,0.999999999]  # if the concrete safe rate is higher than this threshold, we will sample super-nodes on this level. The thresholds are determined by experiments and can be tuned.

MAX_MEAN_SIZE_NOT_ALL_SAFE = 15

class LZeroRobustnessAnalyzer:
    def __init__(self, verifier: Verifier, gpu_workers: list[Connection], cpu_workers: list[Connection], plan_only: bool, strategy: Strategy, probe_thickness: int, until_timedout: bool, force_reestimate: bool):
        self.verifier = verifier
        self.__gpu_workers = gpu_workers
        self.__num_gpu_workers = len(gpu_workers)
        self.__plan_only = plan_only
        self.__strategy = strategy
        self.__probe_thickness = probe_thickness
        self.__until_timedout = until_timedout
        self.__e, self.__v, self.__timeout, self.__channel_size = verifier.eps, verifier.num_pixels, verifier.timeout, verifier.bounded_x0.channel_size
        self.__refinement_db_max_v = verifier.max_k
        self.__milp = MilpDispatcher(cpu_workers, self.__v, self.__channel_size, self.__e, self.__timeout)

        self.covering_sizes = {t:load_covering_sizes(t, clipping_size=500000) for t in range(CVD_MIN_T, min(self.__e + 1, CVD_MAX_T + 1))}
        generator_launch_time, skipping_iter_speed, normal_iter_speed = estimate_generator_speed(num_workers=self.__num_gpu_workers, num_pixels=self.__v, force_reestimate=force_reestimate)
        self.cvd_extra_time = lambda geometry: generator_launch_time + estimate_extra_time(geometry, num_workers=self.__num_gpu_workers, num_pixels=self.__v, pool_size=verifier.pool_size, skipping_iter_speed=skipping_iter_speed, iter_speed=normal_iter_speed)
        splitting_speed, pruning_speed = estimate_SaP_speed(epsilon=self.__e, num_indices=self.__v, force_reestimate=force_reestimate)
        self.estimate_SaP_time = lambda num_unk, level: estimate_SaP_time(num_unk, level, self.__v, splitting_speed[level], pruning_speed[level])
        self.estimate_pruning_time = lambda num_new_combs, num_references, level: num_new_combs * num_references / pruning_speed[level]
        self.mean_concretecheck_time: list[float]|None = None

    def update_image(self, image: np.ndarray, label: int):
        for gpu_worker in self.__gpu_workers:
            worker_seed = random.randint(0, 2 ** 32 - 1)
            gpu_worker.send((worker_seed, image, label))
        self.__milp.update_image(image, label)

    def timed_out(self, estimated_more: float=0.0) -> bool:
        return time.time() - self.analyzer_start_time + estimated_more > self.__timeout

    def __gpu_workers_send_all(self, message):
        for gpu_worker in self.__gpu_workers:
            gpu_worker.send(message)

    def __probe(self) -> tuple[int, float, torch.Tensor]:
        level = 0
        unk_combs1 = torch.empty((0, 1), dtype=torch.int64)
        min_lb_l = self.__gpu_workers[0].recv()  # get the result from the first gpu worker
        combs2check = torch.arange(self.__v).reshape(-1, 1)
        while min_lb_l <= 0 or level == self.__e:  # a node is unknown iff its bound is <= 0
            lb_l_min_values = torch.empty((combs2check.shape[0],), dtype=torch.float32)
            level += 1
            num_each = combs2check.shape[0] // self.__num_gpu_workers
            for i, gpu_worker in enumerate(self.__gpu_workers):
                if i == self.__num_gpu_workers - 1:
                    gpu_worker.send(('CC', combs2check[i * num_each :].numpy()))
                else:
                    gpu_worker.send(('CC', combs2check[i * num_each : (i + 1) * num_each].numpy()))
            for i, gpu_worker in enumerate(self.__gpu_workers):
                if i == self.__num_gpu_workers - 1:
                    lb_l_min_values[i * num_each :] = torch.from_numpy(gpu_worker.recv())
                else:
                    lb_l_min_values[i * num_each : (i + 1) * num_each] = torch.from_numpy(gpu_worker.recv())
            if level == 1:
                unk_combs1 = combs2check[lb_l_min_values <= 0]
            min_lb_l = lb_l_min_values.min().item()
            # print(f"Level {level} min lb_l: {min_lb_l:.4f}, num combs: {combs2check.shape[0]}")
            if min_lb_l > 0 or level == self.__e: break
            num_lt_0 = int(lb_l_min_values.le(0).sum().item())
            unk_combs = combs2check[lb_l_min_values.topk(min(self.__probe_thickness, num_lt_0), largest=False, sorted=False).indices]
            combs2check = canonical_split(unk_combs, self.__v)
        if min_lb_l <= 0: level += 1
        return level, min_lb_l, unk_combs1

    def analyze(self):
        assert self.mean_concretecheck_time is not None, "mean_concretecheck_time should have been set by now."
        print('Planning stage started')
        self.analyzer_start_time = time.time()
        sample_counts, sample_successes = [0] * (self.__e + 1), [0] * (self.__e + 1)
        super_success_rate, mean_supercheck_time = {}, {}
        mean_size_limits = {l: self.__refinement_db_max_v for l in range(CVD_MIN_T, min(self.__e + 1, CVD_MAX_T + 1))}
        if not self.__strategy == Strategy.toptCoVerD:
            first_all_safe_level, min_lb_l, unk_combs1 = self.__probe()
            print(f'Finished probing, first_all_safe_level: {first_all_safe_level}, min_lb_l: {min_lb_l:.4f}, took {round(time.time() - self.analyzer_start_time, 3)} s')
            self.__gpu_workers_send_all(('finished probing', first_all_safe_level))
            if first_all_safe_level <= 1:
                sample_counts[first_all_safe_level:] = [math.comb(self.__v, l) for l in range(first_all_safe_level, self.__e + 1)]
                sample_successes = sample_counts.copy()
            else:
                for l in mean_size_limits:
                    if l < first_all_safe_level:
                        mean_size_limits[l] = MAX_MEAN_SIZE_NOT_ALL_SAFE
                sample_counts[:2] = [1, self.__v]
                sample_successes[:2] = [0, 0 if self.__strategy.require_supercheck() else self.__v - unk_combs1.shape[0]]
                for gpu_worker in self.__gpu_workers:
                    sc, ss = gpu_worker.recv()
                    sample_counts = [a + b for a, b in zip(sample_counts, sc)]
                    sample_successes = [a + b for a, b in zip(sample_successes, ss)]
        concrete_safe_rate = [s / c if c > 0 else 0. for s, c in zip(sample_successes, sample_counts)]
        print(f'Sampled concrete safe rates: {concrete_safe_rate}')
        if self.__strategy.require_supercheck()\
            or self.__strategy == Strategy.FocusBranch and sum(concrete_safe_rate[:2]) == 0:  # if level 0 safe or any level 1 safe, go CCSaP directly
            print(f'Sampling super-nodes on level', end='', flush=True)
            # Sample super-nodes at levels with high concrete safe rate
            sample_all_k_end_level = min(self.__e, CVD_MAX_T)
            for l in range(CVD_MIN_T, sample_all_k_end_level + 1):
                if self.__strategy != Strategy.toptCoVerD and concrete_safe_rate[l] >= SR_THRES_TO_SAMPLE_SUPER[l] or l == sample_all_k_end_level:
                    print(f' {l}', end='\n' if l == sample_all_k_end_level else '', flush=True)
                    for gpu_worker in self.__gpu_workers:
                        gpu_worker.send(('sample-all-k', l))
                    super_success_rate[l], mean_supercheck_time[l] = planner.aggregate_super_samples([w.recv() for w in self.__gpu_workers], l)
        estimated_CC_time_per_l = [0.0] * (self.__e + 1)
        estimated_SaP_time_per_l = [0.0] * (self.__e + 1)  # only estimate SaP time for levels >= 2, since level 0 and 1 are usually very fast and don't need pruning
        first_any_safe_level = 0
        # TODO: Add logging for estimated CC&SaP time
        for l in range(self.__e + 1):
            if concrete_safe_rate[l] == 0.0: # not to start on this level
                first_any_safe_level = min(first_any_safe_level+1, self.__e)
                continue
            if l < 2:  # avoid estimating SaP time for level 0 and 1
                estimated_nodes_this_level = math.comb(self.__v, l)
            else:
                estimated_nodes_this_level = math.ceil(math.comb(self.__v, l) * (1 - concrete_safe_rate[l - 1]) ** 2)
                # print(f"Estimated concretes  on level {l}: {estimated_nodes_this_level} with concrete safe rate {concrete_safe_rate[l]:.4f}")
                if l < self.__e:
                    estimated_unk_nodes_this_level = math.comb(self.__v, l) * (1 - concrete_safe_rate[l])
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
            if sample_successes[l] == math.comb(self.__v, l) and not self.__strategy.require_supercheck():  # all sampled nodes are safe, skip this level
                candidate_strategies.append((estimated_CC_time_per_l[l], 'skip', l))
                break
            tail_CC_time = sum(estimated_CC_time_per_l[l + 1:])
            tail_SaP_time = sum(estimated_SaP_time_per_l[l:])
            if tail_SaP_time > 1.0: tail_SaP_time /= self.__num_gpu_workers
            if not self.__strategy.require_supercheck():
                CCSaP_time = (self.mean_concretecheck_time[l] * math.comb(self.__v, l) + tail_CC_time) / self.__num_gpu_workers + tail_SaP_time
                candidate_strategies.append((CCSaP_time, 'CCSaP', l))
            if l in super_success_rate:
                fraction_robusts[l] = planner.fraction_robust(super_success_rate[l], l, self.__refinement_db_max_v)
                VTs[l], refinement_strategies[l] = planner.compute_super_branching_strategy(fraction_robusts[l], mean_supercheck_time[l], super_success_rate[l], self.covering_sizes[l], l, self.__refinement_db_max_v)
                geometry, geometries_data = planner.choose_geometry(VTs[l], self.__v, l, self.__refinement_db_max_v, mean_size_limits[l])
                chosen_geometries[l], chosen_geometries_info[l] = geometry, geometries_data[0]
                candidate_geometries.extend(geometries_data)
                SCSaP_time: float = (chosen_geometries_info[l]["estimated_summed_duration"] + tail_CC_time) / self.__num_gpu_workers + self.cvd_extra_time(geometry) + tail_SaP_time
                candidate_strategies.append((round(SCSaP_time, 3), 'SCSaP', l))
        choosing_strategy_duration = round(time.time() - choosing_strategy_start, 3)
        planning_duration = round(time.time() - self.analyzer_start_time, 3)
        candidate_strategies = sorted(map(lambda s: (round(s[0] + planning_duration, 3), s[1], s[2]), candidate_strategies), key=lambda s: s[0])
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
        if best_strategy[1] == 'CCSaP' and best_strategy[2] >= 4 or \
            best_strategy[0] > self.__timeout * 1.3 and best_strategy[2] == self.__e and not self.__until_timedout or \
            best_strategy[0] > self.__timeout * 2.0 and best_strategy[2]  < self.__e and not self.__until_timedout:  # SaP estimation might be inaccurate
            results['timed_out'] = True
            print(f'{best_strategy[1]} at level {best_strategy[2]} estimated to take too long, skipping this image.')

        if results['verified'] or results['timed_out'] or self.__plan_only:
            results['running_time'] = round(time.time() - self.analyzer_start_time, 3)
            self.__skip_this_image()
            print('Early termination triggered')
            if results['verified']: results['end_level'] = results['starting_level']
            return results

        # the level-eps unknown nodes of super branching may repeat, so deduplicate them once they are all known
        self.__milp.start(self.analyzer_start_time, deduplicate=best_strategy[1] == 'SCSaP' and self.__level == self.__e)

        if best_strategy[1] == 'CCSaP':
            self.__gpu_workers_send_all(('CCSaP', self.__level))  # no geometry, no refinement strategy
            current_level_start_time = time.time()
            if self.__level == 1:
                unk_combs = unk_combs1
            else:
                unk_combs_lst: list[np.ndarray] = []
                for gpu_worker in self.__gpu_workers:
                    unk_combs_lst.append(gpu_worker.recv())  # wait for gpu workers to finish
                unk_combs = torch.from_numpy(np.vstack(unk_combs_lst))
            print(f'Level {self.__level} finished, unk_combs/all: {unk_combs.shape[0]}/{math.comb(self.__v, self.__level)}'
                f' using {round(time.time() - current_level_start_time, 3)} s')
        elif best_strategy[1] == 'SCSaP':
            chosen_geometry, refinement_strategy = chosen_geometries[self.__level], refinement_strategies[self.__level]
            mean_size = int(chosen_geometry.block_size() * self.__v / chosen_geometry.number_of_points()) + 1
            step_to_level = 0
            current_size = mean_size
            while current_size > self.__level:
                step_to_level += 1
                current_size = refinement_strategy[current_size]
            print(f"{step_to_level} steps to go from mean size {mean_size} to size {self.__level}")
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
            if self.__milp.unfinished > 0:
                print(f'Waiting for MILP jobs to finish, remaining: {self.__milp.unfinished}, finished: {len(self.__milp.innocent)}', flush=True)
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

        if self.timed_out() or self.__milp.done():
            results['timed_out'] = self.timed_out()
        else:
            handle_unk_combs_start = time.time()
            results['timed_out'] = self.__handle_unk_combs(unk_combs, deduplicate=best_strategy[1] == 'SCSaP')
            handle_unk_combs_duration = round(time.time() - handle_unk_combs_start, 3)
            if handle_unk_combs_duration > 0.01:
                print(f'Handling unk_combs duration: {handle_unk_combs_duration} s.')
                results['handle_unk_combs_duration'] = handle_unk_combs_duration
        self.__milp.settle()

        waiting_milp_start = time.time()
        self.__milp.join()
        waiting_milp_duration = round(time.time() - waiting_milp_start, 3)
        if not results['timed_out']: results['timed_out'] = self.timed_out()
        results['running_time'] = round(time.time() - self.analyzer_start_time, 3)
        if self.__milp.unfinished > 0:
            print(f'MILP jobs unfinished: {self.__milp.unfinished}, finished: {len(self.__milp.innocent)}')
        if waiting_milp_duration > 0.1:
            print(f'Waiting for MILP duration: {waiting_milp_duration} s.')
            results['waiting_milp_duration'] = waiting_milp_duration
        if self.__milp.found_adversarial.is_set():
            results['adversarial_results'] = self.__milp.adversarial_results
        elif not results['timed_out']:
            results['verified'] = True
        if len(self.__milp.innocent) + len(self.__milp.waiting) > 0:
            results['proved_with_milp'] = len(self.__milp.innocent)
            results['waiting_for_milp'] = len(self.__milp.waiting)
        results['end_level'] = self.__level
        self.__stop_workers()
        return results

    def __handle_gpu_workers_messages(
        self,
        statistics_by_size: dict,
        initial_covering_size: int,
        refinement_strategy: dict[int,int|None],
        refinement_sizes: dict[int,int]
    ) -> torch.Tensor:
        num_finished = 0
        done_count = 0
        total_covering_size = initial_covering_size
        pbar = tqdm(total=initial_covering_size, desc="SuperChecking", bar_format=TQDM_BAR_FORMAT, file=sys.stdout)
        last_print_time = time.time()
        unk_combs_lst: list[list[int]] = []
        while done_count < self.__num_gpu_workers \
            and not self.__milp.done() \
            and not self.timed_out():
            last_print_time = self.__print_progress(pbar, total_covering_size, num_finished, last_print_time)

            for gpu_worker in self.__gpu_workers:
                if gpu_worker.poll():
                    message = gpu_worker.recv()
                    if isinstance(message, str) and message == 'adversarial-example-suspects':
                        self.__milp.submit(gpu_worker.recv())
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
        return torch.from_numpy(np.vstack(unk_combs_lst)) if len(unk_combs_lst) > 0 else torch.empty((0, self.__level), dtype=torch.int64)

    def __handle_unk_combs(self, unk_combs: torch.Tensor, deduplicate: bool = False) -> bool:  # return True if we should terminate early due to timeout
        if unk_combs.shape[0] == 0: return self.timed_out()  # no need to handle
        if deduplicate:
            print(f"Level {self.__level} Before deduplication, unk_combs size: {unk_combs.shape[0]}", end="; ")
            unk_combs = torch.unique(unk_combs.cuda(), dim=0).cpu()
            print(f'After deduplication, unk_combs size: {unk_combs.shape[0]}')
        while unk_combs.shape[0] > 0 and self.__level < self.__e:
            estimated_SaP_time = self.estimate_SaP_time(unk_combs.shape[0], self.__level) if self.__level >= 2 else 0.0
            if self.timed_out(estimated_more=estimated_SaP_time/self.__num_gpu_workers):
                print(f'Level {self.__level} estimated SaP time: {estimated_SaP_time:3f} s, which is too long, terminating early.')
                return True
            new_level_start_time = time.time()
            if estimated_SaP_time > 1.0:  # local split, parallel prune
                print(f'Level {self.__level} estimated SaP time: {estimated_SaP_time:3f} s. Sharing SaP workload across all GPU workers...')
                for i, gpu_worker in enumerate(self.__gpu_workers):
                    gpu_worker.send(('SaP&CC', unk_combs.numpy()))
                num_new_combs_per_worker = [0] * self.__num_gpu_workers
                for i, gpu_worker in enumerate(self.__gpu_workers):
                    num_new_combs_per_worker[i] = gpu_worker.recv()
                concretecheck_start_time = time.time()
                SaP_duration = round(concretecheck_start_time - new_level_start_time, 3)
                self.__level += 1
                num_after_prune = sum(num_new_combs_per_worker)
                print(f'Current level: {self.__level}, SaP duration: {SaP_duration}s, Total After Pruning: {num_after_prune}, each worker got: {num_new_combs_per_worker}')
            else:    # local split and prune
                new_combs_tensor, reference_bitmasks, target_s_cnt = prepare4pruning(unk_combs, self.__v)
                if new_combs_tensor.shape[0] == 0: return self.timed_out()
                pruned_combs = prune(new_combs_tensor, reference_bitmasks, target_s_cnt, self.__level).numpy()
                if pruned_combs.shape[0] == 0: return self.timed_out()
                concretecheck_start_time = time.time()
                SaP_duration = round(concretecheck_start_time - new_level_start_time, 3)
                self.__level += 1
                num_after_prune = pruned_combs.shape[0]
                print(f'Current level: {self.__level}, SaP duration: {SaP_duration}s, Total After Pruning: {num_after_prune}')
                for i, gpu_worker in enumerate(self.__gpu_workers):
                    num_each = pruned_combs.shape[0] // self.__num_gpu_workers
                    if i == self.__num_gpu_workers - 1:
                        gpu_worker.send(('CC', pruned_combs[i * num_each :]))
                    else:
                        gpu_worker.send(('CC', pruned_combs[i * num_each : (i + 1) * num_each]))
            if self.timed_out(estimated_more=(num_after_prune / self.__num_gpu_workers) * self.mean_concretecheck_time[self.__level]):
                print(f'Level {self.__level} estimated concrete check time: {(num_after_prune / self.__num_gpu_workers) * self.mean_concretecheck_time[self.__level]:.2f} s, which is too long, terminating early.')
                return True
            unk_combs_lst: list[np.ndarray] = []
            for gpu_worker in self.__gpu_workers:
                unk_combs_lst.append(gpu_worker.recv())  # wait for gpu workers to finish
            unk_combs = torch.from_numpy(np.vstack(unk_combs_lst))
            print(f'Level {self.__level} finished, unk_combs/all: {unk_combs.shape[0]}/{num_after_prune}'
                f' using {round(time.time() - new_level_start_time, 3)} s, checking duration: {round(time.time() - concretecheck_start_time, 3)} s')
        if self.__level == self.__e and unk_combs.shape[0] > 0:
            print(f'Checked deepest level {self.__e}, remaining unk_combs size: {unk_combs.shape[0]}')
            self.__milp.submit(unk_combs.numpy())
        return self.timed_out()

    def __print_progress(self, pbar: tqdm, total_covering_size: int, num_finished: int, last_print_time: float) -> float:
        current = time.time()
        if current - last_print_time < 2:
            return last_print_time
        pbar.n = pbar.total - total_covering_size + num_finished
        pbar.set_postfix({'FNS/AGG': f'{num_finished}/{int(total_covering_size)}'})
        return current

    def __stop_workers(self):
        """Abandon the current image on every worker, and wait until they all acknowledged it."""
        for gpu_worker in self.__gpu_workers:
            gpu_worker.send('stop')
        self.__milp.send_stop()
        for gpu_worker in self.__gpu_workers:
            message = gpu_worker.recv()
            while not isinstance(message, str) or message != 'stopped':
                message = gpu_worker.recv()
        self.__milp.wait_stopped()

    def __skip_this_image(self):
        for gpu_worker in self.__gpu_workers:
            gpu_worker.send(('skip', 0))
        self.__milp.stop()
