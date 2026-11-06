# Reference: https://github.com/YuvShap/CoVerD/blob/main/tf_verify/calzone_robustness_analyzer.py
from utils import TQDM_BAR_FORMAT
from utils import protocol as P
from data import load_covering_sizes, Strategy
from coverd_finite_geometries import Geometry
from covering_generator import CVD_MIN_T, CVD_MAX_T
from covering_generator.estimator import estimate_generator_speed, estimate_extra_time
from split_and_prune import estimate_SaP_speed, estimate_SaP_time, prepare4pruning, prune, canonical_split
from gpu_worker import WorkerError, check_worker_error, GpuWorkerPool
from milp_dispatcher import MilpDispatcher
from bounding_engine import BoundingEngine
import planner
import math
import time
import random
import sys
from collections import defaultdict
from multiprocessing.connection import Connection, wait
import torch
import numpy as np
from tqdm import tqdm

SR_THRES_TO_SAMPLE_SUPER = [0., 0., 0.7, 0.97, 0.9997,0.99999,0.999999,0.9999999,0.99999999,0.999999999]  # if the concrete safe rate is higher than this threshold, we will sample super-nodes on this level. The thresholds are determined by experiments and can be tuned.

MAX_MEAN_SIZE_NOT_ALL_SAFE = 15


class Verifier:
    def __init__(self, engine: BoundingEngine, gpu_workers: list[Connection], cpu_workers: list[Connection], plan_only: bool, strategy: Strategy, probe_thickness: int, until_timedout: bool, force_reestimate: bool):
        self.engine = engine
        self.__gpus = GpuWorkerPool(gpu_workers)
        self.__num_gpu_workers = len(gpu_workers)
        self.__plan_only = plan_only
        self.__strategy = strategy
        self.__probe_thickness = probe_thickness
        self.__until_timedout = until_timedout
        self.__e, self.__v, self.__timeout, self.__channel_size = engine.eps, engine.num_pixels, engine.timeout, engine.bounded_x0.channel_size
        self.__refinement_db_max_v = engine.max_k
        self.__milp = MilpDispatcher(cpu_workers, self.__v, self.__channel_size, self.__e, self.__timeout)

        self.covering_sizes = {t:load_covering_sizes(t, clipping_size=500000) for t in range(CVD_MIN_T, min(self.__e + 1, CVD_MAX_T + 1))}
        generator_launch_time, skipping_iter_speed, normal_iter_speed = estimate_generator_speed(num_workers=self.__num_gpu_workers, num_pixels=self.__v, force_reestimate=force_reestimate)
        self.cvd_extra_time = lambda geometry: generator_launch_time + estimate_extra_time(geometry, num_workers=self.__num_gpu_workers, num_pixels=self.__v, pool_size=engine.pool_size, skipping_iter_speed=skipping_iter_speed, iter_speed=normal_iter_speed)
        splitting_speed, pruning_speed = estimate_SaP_speed(epsilon=self.__e, num_indices=self.__v, force_reestimate=force_reestimate)
        self.estimate_SaP_time = lambda num_unk, level: estimate_SaP_time(num_unk, level, self.__v, splitting_speed[level], pruning_speed[level])
        self.mean_boundconcrete_time: list[float]|None = None

    def update_image(self, image: np.ndarray, label: int):
        for gpu_worker in self.__gpus.workers:
            worker_seed = random.randint(0, 2 ** 32 - 1)
            gpu_worker.send((worker_seed, image, label))
        self.__milp.update_image(image, label)

    def timed_out(self, estimated_more: float=0.0) -> bool:
        return time.time() - self.verifier_start_time + estimated_more > self.__timeout

    # ------------------------------------------------------------------ planning
    def __probe(self) -> tuple[int, float, torch.Tensor]:
        """Bound level by level, following the probe_thickness least safe unknown nodes (Appendix F.5).
        Returns the first level whose probed nodes are all safe, the smallest bound seen last and the unknown level-1 nodes."""
        level = 0
        unk_combs1 = torch.empty((0, 1), dtype=torch.int64)
        min_lb_l = self.__gpus.workers[0].recv()  # the root node, bounded by the first gpu worker
        combs2check = torch.arange(self.__v).reshape(-1, 1)
        while min_lb_l <= 0 or level == self.__e:  # a node is unknown iff its bound is <= 0
            level += 1
            self.__gpus.scatter(P.BOUND_CONCRETE, combs2check)
            lb_l_min_values = torch.from_numpy(np.concatenate(self.__gpus.gather()))
            if level == 1:
                unk_combs1 = combs2check[lb_l_min_values <= 0]
            min_lb_l = lb_l_min_values.min().item()
            if min_lb_l > 0 or level == self.__e: break
            num_lt_0 = int(lb_l_min_values.le(0).sum().item())
            unk_combs = combs2check[lb_l_min_values.topk(min(self.__probe_thickness, num_lt_0), largest=False, sorted=False).indices]
            combs2check = canonical_split(unk_combs, self.__v)
        if min_lb_l <= 0: level += 1
        return level, min_lb_l, unk_combs1

    def __sample(self) -> tuple[list[int], list[int], dict, dict, dict, torch.Tensor]:
        """Sample concrete nodes of every level and super nodes of the promising levels (Algorithm 2, line 3)."""
        sample_counts, sample_successes = [0] * (self.__e + 1), [0] * (self.__e + 1)
        super_safe_rate, mean_supercheck_time = {}, {}
        mean_size_limits = {l: self.__refinement_db_max_v for l in range(CVD_MIN_T, min(self.__e + 1, CVD_MAX_T + 1))}
        unk_combs1 = torch.empty((0, 1), dtype=torch.int64)
        if not self.__strategy == Strategy.toptCoVerD:
            first_all_safe_level, min_lb_l, unk_combs1 = self.__probe()
            print(f'Finished probing, first_all_safe_level: {first_all_safe_level}, min_lb_l: {min_lb_l:.4f}, took {round(time.time() - self.verifier_start_time, 3)} s')
            self.__gpus.broadcast((P.PROBE_DONE, first_all_safe_level))
            if first_all_safe_level <= 1:
                sample_counts[first_all_safe_level:] = [math.comb(self.__v, l) for l in range(first_all_safe_level, self.__e + 1)]
                sample_successes = sample_counts.copy()
            else:
                for l in mean_size_limits:
                    if l < first_all_safe_level:
                        mean_size_limits[l] = MAX_MEAN_SIZE_NOT_ALL_SAFE
                sample_counts[:2] = [1, self.__v]
                sample_successes[:2] = [0, 0 if self.__strategy.require_supercheck() else self.__v - unk_combs1.shape[0]]
                for sc, ss in self.__gpus.gather():
                    sample_counts = [a + b for a, b in zip(sample_counts, sc)]
                    sample_successes = [a + b for a, b in zip(sample_successes, ss)]
        concrete_safe_rate = [s / c if c > 0 else 0. for s, c in zip(sample_successes, sample_counts)]
        print(f'Sampled concrete safe rates: {concrete_safe_rate}')
        if self.__strategy.require_supercheck() \
            or self.__strategy == Strategy.FocusBranch and sum(concrete_safe_rate[:2]) == 0:  # if level 0 safe or any level 1 safe, go CCSaP directly
            print(f'Sampling super-nodes on level', end='', flush=True)
            # Sample super-nodes at levels with high concrete safe rate
            sample_all_k_end_level = min(self.__e, CVD_MAX_T)
            for l in range(CVD_MIN_T, sample_all_k_end_level + 1):
                if self.__strategy != Strategy.toptCoVerD and concrete_safe_rate[l] >= SR_THRES_TO_SAMPLE_SUPER[l] or l == sample_all_k_end_level:
                    print(f' {l}', end='\n' if l == sample_all_k_end_level else '', flush=True)
                    self.__gpus.broadcast((P.SAMPLE_SUPER, l))
                    super_safe_rate[l], mean_supercheck_time[l] = planner.aggregate_super_samples(self.__gpus.gather(), l)
        return sample_counts, sample_successes, super_safe_rate, mean_supercheck_time, mean_size_limits, unk_combs1

    def __plan(self) -> dict:
        """Choose the starting level and the branching strategy with the smallest estimated time (Algorithm 3)."""
        sample_counts, sample_successes, super_safe_rate, mean_supercheck_time, mean_size_limits, unk_combs1 = self.__sample()
        concrete_safe_rate = [s / c if c > 0 else 0. for s, c in zip(sample_successes, sample_counts)]
        estimated_CC_time_per_l = [0.0] * (self.__e + 1)
        estimated_SaP_time_per_l = [0.0] * (self.__e + 1)  # only estimate SaP time for levels >= 2, since level 0 and 1 are usually very fast and don't need pruning
        first_any_safe_level = 0
        for l in range(self.__e + 1):
            if concrete_safe_rate[l] == 0.0: # not to start on this level
                first_any_safe_level = min(first_any_safe_level+1, self.__e)
                continue
            if l < 2:  # avoid estimating SaP time for level 0 and 1
                estimated_nodes_this_level = math.comb(self.__v, l)
            else:
                estimated_nodes_this_level = math.ceil(math.comb(self.__v, l) * (1 - concrete_safe_rate[l - 1]) ** 2)  # Eq. (24)
                if l < self.__e:
                    estimated_unk_nodes_this_level = math.comb(self.__v, l) * (1 - concrete_safe_rate[l])
                    estimated_SaP_time_per_l[l] = self.estimate_SaP_time(estimated_unk_nodes_this_level, l)
            estimated_CC_time_per_l[l] = self.mean_boundconcrete_time[l] * estimated_nodes_this_level
            if concrete_safe_rate[l] == 1.0:
                break  # this level is estimated to be 100% robust and verified, no need to go deeper

        candidate_strategies: list[tuple[float, str, int]] = []
        candidate_geometries: list[dict[str, object]] = []
        fraction_robusts: dict[int, dict[int, dict[int, float]]] = {}
        VTs: dict[int, dict[int, float]] = {}
        refinement_strategies: dict[int, dict[int, int|None]] = {}
        chosen_geometries: dict[int, Geometry] = {}
        chosen_geometries_info: dict[int, dict[str, object]] = {}

        choosing_strategy_start = time.time()
        for l in range(first_any_safe_level, self.__e + 1):
            if sample_successes[l] == math.comb(self.__v, l) and not self.__strategy.require_supercheck():  # all nodes are safe, skip this level
                candidate_strategies.append((estimated_CC_time_per_l[l], 'skip', l))
                break
            tail_CC_time = sum(estimated_CC_time_per_l[l + 1:])
            tail_SaP_time = sum(estimated_SaP_time_per_l[l:])
            if tail_SaP_time > 1.0: tail_SaP_time /= self.__num_gpu_workers
            if not self.__strategy.require_supercheck():
                CCSaP_time = (self.mean_boundconcrete_time[l] * math.comb(self.__v, l) + tail_CC_time) / self.__num_gpu_workers + tail_SaP_time
                candidate_strategies.append((CCSaP_time, 'CCSaP', l))
            if l in super_safe_rate:
                fraction_robusts[l] = planner.fraction_robust(super_safe_rate[l], l, self.__refinement_db_max_v)
                VTs[l], refinement_strategies[l] = planner.compute_super_branching_strategy(fraction_robusts[l], mean_supercheck_time[l], super_safe_rate[l], self.covering_sizes[l], l, self.__refinement_db_max_v)
                geometry, geometries_data = planner.choose_geometry(VTs[l], self.__v, l, self.__refinement_db_max_v, mean_size_limits[l])
                chosen_geometries[l], chosen_geometries_info[l] = geometry, geometries_data[0]
                candidate_geometries.extend(geometries_data)
                SCSaP_time: float = (chosen_geometries_info[l]["estimated_summed_duration"] + tail_CC_time) / self.__num_gpu_workers + self.cvd_extra_time(geometry) + tail_SaP_time
                candidate_strategies.append((round(SCSaP_time, 3), 'SCSaP', l))
        choosing_strategy_duration = round(time.time() - choosing_strategy_start, 3)
        planning_duration = round(time.time() - self.verifier_start_time, 3)
        candidate_strategies = sorted(map(lambda s: (round(s[0] + planning_duration, 3), s[1], s[2]), candidate_strategies), key=lambda s: s[0])
        best_strategy = candidate_strategies[0]
        candidate_strategies = list(filter(lambda s: s[0] <= self.__timeout * 10, candidate_strategies))  # filter out noise strategies
        if len(candidate_strategies) == 0: candidate_strategies.append(best_strategy)  # at least have one strategy to show, even if it's an outlier
        print(f"Planning stage took {planning_duration} s, choosing strategy duration: {choosing_strategy_duration} s, best strategy: {best_strategy}")
        print(f"Candidate strategies: {candidate_strategies}")

        results = {
            'verified': best_strategy[1] == 'skip',
            'timed_out': self.timed_out(),
            'choosing_strategy_duration': choosing_strategy_duration,
            'best_strategy': best_strategy,
            'estimated_verification_time': best_strategy[0],
            'starting_level': best_strategy[2],
            'planning_duration': planning_duration,
            'candidate_strategies': candidate_strategies
        }
        if len(chosen_geometries) > 0:
            results['chosen_geometries'] = {l:str(g) for l,g in chosen_geometries.items()}
            results['candidate_geometries'] = candidate_geometries
        level = best_strategy[2]
        plan = {'results': results, 'strategy': best_strategy[1], 'level': level, 'unk_combs1': unk_combs1}
        if best_strategy[1] == 'SCSaP':
            plan.update(geometry=chosen_geometries[level], geometry_info=chosen_geometries_info[level],
                        refinement_strategy=refinement_strategies[level], super_safe_rate=super_safe_rate[level],
                        mean_supercheck_time=mean_supercheck_time[level], VT=VTs[level], fraction_robust=fraction_robusts[level])
        return plan

    # ------------------------------------------------------------------ verification
    def verify(self):
        assert self.mean_boundconcrete_time is not None, "mean_boundconcrete_time should have been set by now."
        print('Planning stage started')
        self.verifier_start_time = time.time()
        plan = self.__plan()
        results, best_strategy = plan['results'], plan['results']['best_strategy']
        self.__level = plan['level']

        # to be deleted later if needed
        if best_strategy[1] == 'CCSaP' and best_strategy[2] >= 4 or \
            best_strategy[0] > self.__timeout * 1.3 and best_strategy[2] == self.__e and not self.__until_timedout or \
            best_strategy[0] > self.__timeout * 2.0 and best_strategy[2]  < self.__e and not self.__until_timedout:  # SaP estimation might be inaccurate
            results['timed_out'] = True
            print(f'{best_strategy[1]} at level {best_strategy[2]} estimated to take too long, skipping this image.')

        if results['verified'] or results['timed_out'] or self.__plan_only:
            results['running_time'] = round(time.time() - self.verifier_start_time, 3)
            self.__gpus.broadcast((P.SKIP, 0))
            self.__milp.stop()
            print('Early termination triggered')
            if results['verified']: results['end_level'] = results['starting_level']
            return results

        # the level-eps unknown nodes of super branching may repeat, so deduplicate them once they are all known
        self.__milp.start(self.verifier_start_time, deduplicate=plan['strategy'] == 'SCSaP' and self.__level == self.__e)
        try:
            if plan['strategy'] == 'CCSaP':
                unk_combs = self.__run_concrete_start(plan)
            elif plan['strategy'] == 'SCSaP':
                unk_combs = self.__run_super_branching(plan, results)
            else: assert False, f'Unknown best strategy: {best_strategy}'

            if self.timed_out() or self.__milp.done():
                results['timed_out'] = self.timed_out()
            else:
                handle_unk_combs_start = time.time()
                results['timed_out'] = self.__handle_unk_combs(unk_combs, deduplicate=plan['strategy'] == 'SCSaP')
                handle_unk_combs_duration = round(time.time() - handle_unk_combs_start, 3)
                if handle_unk_combs_duration > 0.01:
                    print(f'Handling unk_combs duration: {handle_unk_combs_duration} s.')
                    results['handle_unk_combs_duration'] = handle_unk_combs_duration
        except WorkerError as e:  # the image cannot be verified; stop everything and report it
            print(f'A GPU worker failed, abandoning this image: {e}')
            results['worker_error'] = str(e)
        self.__milp.settle()

        waiting_milp_start = time.time()
        self.__milp.join()
        waiting_milp_duration = round(time.time() - waiting_milp_start, 3)
        if not results['timed_out']: results['timed_out'] = self.timed_out()
        results['running_time'] = round(time.time() - self.verifier_start_time, 3)
        if self.__milp.unfinished > 0:
            print(f'MILP jobs unfinished: {self.__milp.unfinished}, finished: {len(self.__milp.innocent)}')
        if waiting_milp_duration > 0.1:
            print(f'Waiting for MILP duration: {waiting_milp_duration} s.')
            results['waiting_milp_duration'] = waiting_milp_duration
        if self.__milp.found_adversarial.is_set():
            results['adversarial_results'] = self.__milp.adversarial_results
        elif not results['timed_out'] and 'worker_error' not in results:  # a failed GPU worker leaves the image unverified
            results['verified'] = True
        if len(self.__milp.innocent) + len(self.__milp.waiting) > 0:
            results['proved_with_milp'] = len(self.__milp.innocent)
            results['waiting_for_milp'] = len(self.__milp.waiting)
        results['end_level'] = self.__level
        self.__stop_workers()
        return results

    def __run_concrete_start(self, plan: dict) -> torch.Tensor:
        """Bound every concrete node of the starting level (Θ_l with R[V] = l)."""
        self.__gpus.broadcast((P.START_CONCRETE, self.__level))  # no geometry, no refinement strategy
        current_level_start_time = time.time()
        if self.__level == 1:
            unk_combs = plan['unk_combs1']  # all level-1 nodes were bounded while probing
        else:
            unk_combs = torch.from_numpy(self.__gpus.gather_rows())
        print(f'Level {self.__level} finished, unk_combs/all: {unk_combs.shape[0]}/{math.comb(self.__v, self.__level)}'
            f' using {round(time.time() - current_level_start_time, 3)} s')
        return unk_combs

    def __run_super_branching(self, plan: dict, results: dict) -> torch.Tensor:
        """Branch on super nodes starting from the covering verification design (R[V] = CVD), see `gpu_worker.SuperBranching`."""
        geometry, refinement_strategy = plan['geometry'], plan['refinement_strategy']
        mean_size = int(geometry.block_size() * self.__v / geometry.number_of_points()) + 1
        step_to_level, current_size = 0, mean_size
        while current_size > self.__level:
            step_to_level += 1
            current_size = refinement_strategy[current_size]
        print(f"{step_to_level} steps to go from mean size {mean_size} to size {self.__level}")
        assert self.__level == min(refinement_strategy.keys()), f"Level mismatch: {self.__level} vs {min(refinement_strategy.keys())}"
        chosen_indices = random.sample(range(geometry.number_of_points()), self.__v)
        random.shuffle(chosen_indices)
        self.__gpus.broadcast((P.START_SUPER, geometry, chosen_indices, refinement_strategy))
        print(f'Chosen {plan["geometry_info"]}; Operating on level: {self.__level}')

        refinement_sizes = {v: self.covering_sizes[self.__level][(v, k)] for v, k in refinement_strategy.items() if k is not None}
        stats_by_size = defaultdict(lambda: {'num_checked': 0, 'successes': 0, 'total_duration': 0.})
        start = time.time()
        unk_combs = self.__collect_super_branching(stats_by_size, geometry.number_of_blocks(), refinement_strategy, refinement_sizes)
        duration = round(time.time() - start, 3)
        summed_duration = round(sum(stats['total_duration'] for stats in stats_by_size.values()), 3)
        print(f'GPU iterating coverings took {duration} s, Summed duration (all gpu workers): {summed_duration} s', flush=True)
        if self.__milp.unfinished > 0:
            print(f'Waiting for MILP jobs to finish, remaining: {self.__milp.unfinished}, finished: {len(self.__milp.innocent)}', flush=True)
        results['SCSaP'] = {
            'chosen_geometry': plan['geometry_info'],
            'iterating_covering_duration': duration,
            'summed_duration': summed_duration
        }
        results['exec_details'] = {
            "Safe Rate": {k:round(v, 5) for k, v in plan['super_safe_rate'].items()},
            "Mean Time": {k:round(v, 6) for k, v in plan['mean_supercheck_time'].items()},
            "VT": {k:round(v, 6) for k, v in plan['VT'].items()},
            "R": refinement_strategy,
            "refinement_sizes": refinement_sizes,
            "fraction_robust": {v: round(plan['fraction_robust'][v][k], 5) for v, k in refinement_strategy.items() if k is not None},
            "num_checked": {size: stats['num_checked'] for size, stats in stats_by_size.items()},
            "successes": {size: stats['successes'] for size, stats in stats_by_size.items()},
            "safe_rate": {size: round(stats['successes'] / stats['num_checked'], 5) if stats['num_checked'] > 0 else 0 for size, stats in stats_by_size.items()},
            "total_duration": {size: round(stats['total_duration'], 6) for size, stats in stats_by_size.items()},
            "mean_duration": {size: round(stats['total_duration'] / stats['num_checked'], 6) if stats['num_checked'] > 0 else 0 for size, stats in stats_by_size.items()},
        }
        return unk_combs

    def __collect_super_branching(
        self,
        stats_by_size: dict,
        initial_covering_size: int,
        refinement_strategy: dict[int, int | None],
        refinement_sizes: dict[int, int]
    ) -> torch.Tensor:
        """
        Collect the workers' super branching results until every worker is done, MILP found an adversarial example or
        the time is up: statistics per focus-set size, unknown concrete nodes below level eps, and level-eps suspects for
        MILP. Returns the unknown concrete nodes.
        """
        workers = self.__gpus.workers
        num_finished, done_count = 0, 0
        total_covering_size = initial_covering_size
        unk_combs_lst: list[np.ndarray] = []
        pbar = tqdm(total=initial_covering_size, desc="SuperChecking", bar_format=TQDM_BAR_FORMAT, file=sys.stdout)
        last_print_time = time.time()
        while done_count < self.__num_gpu_workers and not self.__milp.done() and not self.timed_out():
            last_print_time = self.__print_progress(pbar, total_covering_size, num_finished, last_print_time)
            for conn in wait(workers, timeout=0.5):
                message = check_worker_error(conn.recv())
                tag = message[0]
                if tag == P.SUSPECTS:
                    self.__milp.submit(message[1])
                elif tag == P.UNKNOWN:
                    unk_combs_lst.append(message[1])
                elif tag == P.DONE:
                    done_count += 1
                elif tag == P.STATS:
                    _, size, num_checked, num_verified, duration = message
                    size_statistics = stats_by_size[size]
                    if size < self.__level:
                        num_finished += num_verified
                    elif refinement_strategy.get(size) is not None:  # could potentially be refined
                        num_finished += num_checked
                        size_statistics['num_checked'] += num_checked
                        size_statistics['successes'] += num_verified
                        size_statistics['total_duration'] += duration
                        num_failed = num_checked - num_verified
                        if num_failed > 0:  # to be refined later, add to total covering size
                            total_covering_size += refinement_sizes[size] * num_failed
                    else:  # concrete nodes
                        num_finished += num_checked
                        size_statistics['num_checked'] += num_checked
                        size_statistics['total_duration'] += duration
                        size_statistics['successes'] += num_verified
                else: assert False, f"Unexpected message {message}"
        pbar.close()
        print(f"num_finished: {num_finished}, total_covering_size: {total_covering_size}, remaining: {total_covering_size - num_finished}")
        return torch.from_numpy(np.vstack(unk_combs_lst)) if unk_combs_lst else torch.empty((0, self.__level), dtype=torch.int64)

    def __handle_unk_combs(self, unk_combs: torch.Tensor, deduplicate: bool = False) -> bool:  # return True if we should terminate early due to timeout
        """Concrete branching and pruning, level by level (Algorithm 6)."""
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
                self.__gpus.broadcast((P.SPLIT_PRUNE_CHECK, unk_combs.numpy()))
                num_new_combs_per_worker = self.__gpus.gather()
                concretecheck_start_time = time.time()
                SaP_duration = round(concretecheck_start_time - new_level_start_time, 3)
                self.__level += 1
                num_after_prune = sum(num_new_combs_per_worker)
                print(f'Current level: {self.__level}, SaP duration: {SaP_duration}s, Total After Pruning: {num_after_prune}, each worker got: {num_new_combs_per_worker}')
            else:    # local split and prune
                new_combs_tensor, reference_bitmasks, target_s_cnt = prepare4pruning(unk_combs, self.__v)
                if new_combs_tensor.shape[0] == 0: return self.timed_out()
                pruned_combs = prune(new_combs_tensor, reference_bitmasks, target_s_cnt, self.__level)
                if pruned_combs.shape[0] == 0: return self.timed_out()
                concretecheck_start_time = time.time()
                SaP_duration = round(concretecheck_start_time - new_level_start_time, 3)
                self.__level += 1
                num_after_prune = pruned_combs.shape[0]
                print(f'Current level: {self.__level}, SaP duration: {SaP_duration}s, Total After Pruning: {num_after_prune}')
                self.__gpus.scatter(P.BOUND_CONCRETE, pruned_combs)
            if self.timed_out(estimated_more=(num_after_prune / self.__num_gpu_workers) * self.mean_boundconcrete_time[self.__level]):
                print(f'Level {self.__level} estimated concrete check time: {(num_after_prune / self.__num_gpu_workers) * self.mean_boundconcrete_time[self.__level]:.2f} s, which is too long, terminating early.')
                return True
            unk_combs = torch.from_numpy(self.__gpus.gather_rows())
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
        self.__gpus.broadcast(P.STOP)
        self.__milp.send_stop()
        self.__gpus.wait_stopped()
        self.__milp.wait_stopped()
