from AbstractInterpreter import L0BoundedInput, SequentialAT
from split_and_prune import split_and_prune
from bimm import combs2bitmasks
from covering_generator import launch_covering_generator
from data import load_covering
from utils import DEVICE, TQDM_BAR_FORMAT, TimeoutException
from coverd_finite_geometries import Geometry
from multiprocessing.connection import Client, Listener
from time import perf_counter
from functools import lru_cache
from collections import defaultdict
import threading
import math
import signal
import torch
from tqdm import tqdm


def batched_randsample(upper_bound: int, sample_size: int, batch_size: int) -> torch.Tensor:
    """
    Generate a tensor of shape (batch_size, sample_size) with random indices from 0 to upper_bound-1 on cpu.
    The indices are no need to be unique within each row.
    The indices are sorted in ascending order.
    """
    rand_combs = torch.rand(batch_size, upper_bound, device=DEVICE).topk(sample_size, dim=1, sorted=False).indices
    return rand_combs.sort().values.cpu()

@lru_cache(maxsize=None)
def selected_indices_sample(k: int, num_select: int, repetitions: int) -> torch.Tensor:
    """
    Generate a tensor of shape (repetitions, num_select) with random indices from 0 to k-1.
    The indices are sorted in ascending order.

    Sample plan A: for comb(k, num_select) >= (repetitions*100). use torch.randperm(k)[:num_select] for each row
    Sample plan B: for (repetitions) < comb(k, num_select) <  (repetitions*100). get all combinations and then use randperm to select a batch of combinations
    Sample plan C: for comb(k, num_select) < (repetitions), use torch.combinations(torch.arange(k), num_select) to generate all combinations
    """
    if math.comb(k, num_select) > repetitions * 100:  # If k is too large, we use the plan A
        return batched_randsample(k, num_select, repetitions)
    else:  # If k is small enough, we use the plan B
        all_combs = torch.combinations(torch.arange(k, device=DEVICE), num_select)
        if len(all_combs) > repetitions:
            choices = torch.randperm(len(all_combs), device=DEVICE)[:repetitions]
            return all_combs[choices].cpu()
        else:
            return all_combs.cpu()

def get_refinement_subsets(partial_indices: list[int], covering: list[list[int]]) -> list[list[int]]:
    return [[partial_indices[i] for i in cover] for cover in covering]

def load_refinement_coverings(refinement_strategy: dict[int, int|None]) -> dict[int, list[list[int]]]:
    level = min(refinement_strategy.keys())
    refinement_coverings: dict[int, list[list[int]]] = {}
    for v, k in refinement_strategy.items():
        if k is None: continue  # different strategy
        refinement_coverings[v] = load_covering(v, k, level)
    return refinement_coverings

class Verifier:
    def __init__(self,
        torch_net: torch.nn.Module,
        seqAT: SequentialAT,
        bounded_x0: L0BoundedInput,
        batch_size: int,
        targeted: bool=False,
        max_k: int=200,
        rep_each_k: int=50,
        rep_each_level: int=1000,
        timeout: int=0,
        demo: bool=False
    ) -> None:
        self.torch_net = torch_net.cpu()
        self.seqAT = seqAT
        self.bounded_x0 = bounded_x0
        self.eps = self.bounded_x0.eps
        self.num_pixels = self.bounded_x0.num_pixels
        self.device = self.bounded_x0.device
        self.batch_size = batch_size
        self.targeted = targeted
        self.max_k = max_k
        self.rep_each_k = rep_each_k
        self.rep_each_level = rep_each_level
        self.timeout = timeout
        self.demo = demo
        self.__greater_batch = self.batch_size * 10
        self.start_levels: defaultdict[int, int] = defaultdict(lambda: 0)  #key: k, value: start level to probe and verify

    def __check_current_batch(self) -> torch.Tensor:
        """
        Check if each of the current batch is safe.

        :return: A 1D Tensor on self.device of shape (batch_size,) with True if the corresponding input is safe, False otherwise.
        """
        self.seqAT.update_activation_rp(bounded_in=self.bounded_x0)
        lrp = self.seqAT.tail.backsubstitution_lrp()
        lb_l = self.bounded_x0.compute_lbound(lrp).squeeze(2)
        return lb_l.ge(0).all(dim=1)

    def first_safe_level(self) -> int:
        for level in range(self.eps + 1):
            all_safe = True
            if level == 0:
                combs_to_check = torch.empty((1, 0), dtype=torch.int64)
            elif level == 1:
                combs_to_check = torch.arange(self.num_pixels).reshape(-1, 1)
            else:
                combs_to_check = selected_indices_sample(self.num_pixels, level, self.rep_each_level * 10)  # tunable parameter
            for i in range(0, combs_to_check.shape[0], self.batch_size):
                combs_to_check_batch = combs_to_check[i:i + self.batch_size].to(self.device)
                with self.bounded_x0.selecting(combs_to_check_batch):
                    safes = self.__check_current_batch()
                if not safes.all():
                    all_safe = False
                    break
            if all_safe:
                break
        return level

    def probe(self, partial_indices: torch.Tensor | None=None, start_level: int=0) -> tuple[float, int]:
        """
        Probe from start_level downward with the given partial indices and estimate the verification time.

        :param partial_indices: Indices of the pixels to be considered in the computation of diff.
        :type partial_indices: torch.Tensor with shape (num_indices,) | None
        :param start_level: The level to start probing from.
        :return: Estimated verification time and the actual start level.
        """
        k = partial_indices.shape[0] if partial_indices is not None else self.num_pixels
        nodes_per_second = 0.0
        estimated_total_nodes = 0
        last_success_rate = 0.
        first_any_safe_level = start_level
        if self.demo:
            sampled_nodes = 0
            start = perf_counter()
        with self.bounded_x0.partial(partial_indices):
            for level in range(start_level, self.eps + 1):
                if level == 0:
                    combs_to_check = torch.empty((1, 0), dtype=torch.int64)
                    estimated_nodes_this_level = 1
                elif level == 1:
                    combs_to_check = torch.arange(k).reshape(-1, 1)
                    estimated_nodes_this_level = k
                else:
                    combs_to_check = selected_indices_sample(k, level, self.rep_each_level)
                    estimated_nodes_this_level = math.ceil(math.comb(k, level) * (1 - last_success_rate) ** 2)
                sample_nodes_this_level = combs_to_check.shape[0]
                num_safe_nodes = 0
                level_start = perf_counter()
                for i in range(0, sample_nodes_this_level, self.batch_size):
                    combs_to_check_batch = combs_to_check[i:i + self.batch_size].to(self.device)
                    with self.bounded_x0.selecting(combs_to_check_batch):
                        num_safe_nodes += int(self.__check_current_batch().sum().item())

                if num_safe_nodes == 0: first_any_safe_level += 1
                current_speed = sample_nodes_this_level / (perf_counter() - level_start)
                if level != self.eps: nodes_per_second = max(current_speed, nodes_per_second)  # not the last level
                last_success_rate = num_safe_nodes / sample_nodes_this_level
                estimated_total_nodes += estimated_nodes_this_level
                if self.demo:
                    sampled_nodes += sample_nodes_this_level
                    print(f"Level {level}: {100*last_success_rate:>6.2f}% safe, speed: {current_speed:>8.2f} nodes/s, estimated nodes: {estimated_nodes_this_level}")
                if num_safe_nodes == sample_nodes_this_level:
                    break  # All nodes are safe, no need to go deeper
        estimated_verification_time = estimated_total_nodes / nodes_per_second
        if self.demo: print(f"Sample Time: {perf_counter() - start:2.2}s, Sampled Nodes: {sampled_nodes}, Speed: {nodes_per_second:.2f} nodes/s, Estimated nodes: {estimated_total_nodes}\nEstimated verification time: {estimated_verification_time:.2f}s, First any safe level: {first_any_safe_level}")
        return estimated_verification_time, first_any_safe_level

    def adapt_target_images(self, images: torch.Tensor, true_labels: torch.Tensor, target_label: int | None=None, check_first=True) -> None:
        """
        Adapt the input image and true label for the verification process. Will first check if the image matches the true label if check_first is True.

        :param images: Input images to be verified. If there is only one image, the batch size is 1.
        :type images: torch.Tensor with shape (batch_size, channel, height, width) or (batch_size, -1, 1).
        :param true_labels: True labels for the input images. Must be a tensor of shape (batch_size,).
        :type true_labels: torch.Tensor with shape (batch_size,).
        :param target_label: Target label for targeted verification, if any.
        :param check_first: If True, will assert that the input image matches the true label.
        """
        if check_first:
            assert images.ndim == 4, "Input images must be of shape (batch_size, channel, height, width)."
            assert self.torch_net(images.cpu()).argmax(dim=1).eq(true_labels).all(), "The input image does not match the true label."
            images = images.flatten(start_dim=1).unsqueeze(2)
        assert images.ndim == 3, "Input images must be of shape (batch_size, -1, 1)."
        self.bounded_x0.update_x(images)  # (batch, d1, 1)
        if target_label is not None:
            self.seqAT.tail.make_marginal(true_labels=true_labels, target_label=target_label, device=self.device)
        else:
            self.seqAT.tail.make_marginals(true_labels=true_labels, device=self.device)
        self.start_levels = defaultdict(lambda: 0)

    def verify1(self, partial_indices: torch.Tensor | None=None, start_level: int=0) -> bool:
        """
        Deprecated.
        """
        num_indices = self.num_pixels
        if partial_indices is not None:
            num_indices = partial_indices.shape[0]
            if start_level == 0:
                start_level = self.start_levels[num_indices]  # use the start level for this k
        bitmask_length = (num_indices + 63) // 64
        if math.comb(num_indices, start_level) > 10 * 1024 ** 3 * 8 * num_indices:
            return False
        combs_to_check = torch.combinations(torch.arange(num_indices), start_level)
        if start_level == 0: combs_to_check = torch.empty((1, 0), dtype=torch.int64)
        safe_bitmasks: torch.Tensor = torch.empty((0, bitmask_length), dtype=torch.long)
        success = False
        if self.demo:
            start = perf_counter()
            branching_time, total_nodes = 0.0, 0
        for level in range(start_level, self.eps + 1):
            num_to_check = combs_to_check.shape[0]
            if num_to_check == 0:
                success = True
                break
            unk_combs_lst: list[torch.Tensor] = []
            new_safe_bitmasks_lst: list[torch.Tensor] = [safe_bitmasks]
            if self.demo:
                safe_cnt, unk_cnt = 0, 0
                if self.targeted:
                    min_value = float("inf")
            level_start = perf_counter()
            pbar = tqdm(total=num_to_check, desc=f"Level {level}", bar_format=TQDM_BAR_FORMAT, disable=not self.demo)
            with self.seqAT.working_on(self.device), self.bounded_x0.partial(partial_indices):
                for i in range(0, num_to_check, self.batch_size):
                    combs_to_check_batch = combs_to_check[i:i + self.batch_size].to(self.device)
                    with self.bounded_x0.selecting(combs_to_check_batch):
                        self.seqAT.update_activation_rp(bounded_in=self.bounded_x0)
                        lrp = self.seqAT.tail.backsubstitution_lrp()
                        lb_l = self.bounded_x0.compute_lbound(lrp).squeeze(2)

                    safes = lb_l.ge(0).all(dim=1)
                    new_safe_bitmasks_lst.append(combs2bitmasks(combs_to_check_batch[safes], bitmask_length).cpu())
                    unk_combs_lst.append(combs_to_check_batch[~safes].cpu())

                    if self.demo:
                        safe_cnt += new_safe_bitmasks_lst[-1].shape[0]
                        unk_cnt += unk_combs_lst[-1].shape[0]
                        posfix: dict = {"Safe:UNK": f"{safe_cnt}:{unk_cnt}"}
                        if self.targeted:
                            if lb_l.min() < min_value:
                                min_value = lb_l.min().item()
                                min_comb = combs_to_check_batch[lb_l.argmin()]
                            posfix["Min"] = f"{min_value:.2f}"
                            posfix["Comb"] = min_comb.tolist()
                        pbar.set_postfix(posfix)
                        pbar.update(combs_to_check_batch.shape[0])
                    if self.timeout != 0 and i == self.batch_size * 100:  # calculate remaining time to decide if to early stop
                        if not self.demo: # calculate remaining time without pbar
                            rate = (perf_counter() - level_start) / 100
                            if ((num_to_check - i) / self.batch_size) * rate > self.timeout:
                                return False
                        elif (pbar.total - pbar.n) / pbar.format_dict['rate'] > self.timeout:
                            signal.alarm(0)
                            raise TimeoutException()
            pbar.close()

            if self.demo: total_nodes += num_to_check
            unk_combs = torch.cat(unk_combs_lst, dim=0)
            if unk_combs.shape[0] == 0:
                success = True
                break
            if level == self.eps: break
            safe_bitmasks = torch.cat(new_safe_bitmasks_lst, dim=0)
            del combs_to_check, combs_to_check_batch, lb_l, lrp, new_safe_bitmasks_lst, unk_combs_lst, safes
            torch.cuda.empty_cache()
            level_start = perf_counter()
            combs_to_check = split_and_prune(unk_combs, safe_bitmasks, level, num_indices, self.demo)
            if self.demo: branching_time += perf_counter() - level_start
            del unk_combs
            torch.cuda.empty_cache()
        if self.demo:
            print(f"Total: Nodes {total_nodes}; Time {perf_counter() - start:.2f}s; Branching Time {branching_time:.2f}s")
            print(f"Safe: {success}")
        return success

    def sample_all_level(self) -> tuple[list[int], list[int], list[float]]:
        sample_counts, sample_successes = [0] * (self.eps + 1), [0] * (self.eps + 1)
        sample_durations = [0.0] * (self.eps + 1)
        for level in range(self.eps + 1):
            sample_start = perf_counter()
            if level == 0:
                combs_to_check = torch.empty((1, 0), dtype=torch.int64)
            elif level == 1:
                combs_to_check = torch.arange(self.num_pixels).reshape(-1, 1)
            else:
                combs_to_check = selected_indices_sample(self.num_pixels, level, self.rep_each_level * 10)  # tunable parameter
            sample_counts[level] = combs_to_check.shape[0]
            for i in range(0, combs_to_check.shape[0], self.batch_size):
                combs_to_check_batch = combs_to_check[i:i + self.batch_size].to(self.device)
                with self.bounded_x0.selecting(combs_to_check_batch):
                    sample_successes[level] += int(self.__check_current_batch().sum().item())
            sample_durations[level] = perf_counter() - sample_start
        return sample_counts, sample_successes, sample_durations

    def sample_all_k(self, baseline: dict[int, float] | None=None) -> dict[int, float]:
        """
        Deprecated.
        Sample all k from eps to max_k self.rep_each_k times and return the mean estimated verification time for each k.
        If baseline is provided, it will stop sampling when the estimated time exceeds the baseline for that k.

        :param baseline: A dictionary with k as keys and baseline verification time as values. If None, no baseline is used.
        :type baseline: dict[int, float] | None
        :return: A dictionary with k as keys and estimated verification time as values.
        """
        estimated_times: defaultdict[int, float] = defaultdict(float)
        pbar = tqdm(total=self.max_k - self.eps + 1, desc="Sampling k", disable=not self.demo, bar_format=TQDM_BAR_FORMAT)
        self.demo = False
        for k in range(self.eps, self.max_k + 1):
            all_partial_indices = batched_randsample(self.num_pixels, k, self.rep_each_k)
            self.start_levels[k] = start_level = self.start_levels[k-1]  # init from the previous k
            ###
            if self.start_levels[k] == 0:  # only batch on root level if start level is 0, otherwise skip this part
                batch_on_root_start = perf_counter()
                all_partial_indices = all_partial_indices[~self.check_partials_on_root(all_partial_indices)]  # root level unk partial indices
                estimated_times[k] += perf_counter() - batch_on_root_start
                if all_partial_indices.shape[0] == self.rep_each_k:
                    self.start_levels[k] = 1  # if all sampled partial are not safe at root level, start from level 1
                start_level = 1  # for the probe, since level 0 is already checked
            ###
            num_actual_start_level_increased = 0
            for partial_indices in all_partial_indices:
                estimated_time, actual_start_level = self.probe(partial_indices, start_level)
                estimated_times[k] += estimated_time
                if actual_start_level > start_level:
                    num_actual_start_level_increased += 1
            if num_actual_start_level_increased == self.rep_each_k:
                self.start_levels[k] += 1
            estimated_times[k] /= self.rep_each_k
            if k != self.eps and baseline is not None and estimated_times[k] > baseline[k] * 1.5:  # tunable coefficient
                break
            pbar.set_postfix({"k": k, "Estimated Time": f"{estimated_times[k]:.4f}s",
                "Start Level": self.start_levels[k], "Start Level Increased": num_actual_start_level_increased})
            pbar.update(1)
        pbar.close()
        if k != self.max_k:
            self.start_levels.default_factory = lambda: self.start_levels[k]  # set the default factory to the last start level

        return estimated_times

    def sample_all_k_on_root(self):
        sampling_counts, sampling_successes , sampling_time = [], [], []
        rep_each_k = self.rep_each_k
        for k in range(self.__level, self.max_k + 1):
            all_partial_indices = batched_randsample(self.num_pixels, k, rep_each_k)
            batch_on_root_start = perf_counter()
            sampling_successes.append(self.check_partials_on_root(all_partial_indices).sum().item())
            sampling_time.append(perf_counter() - batch_on_root_start)
            sampling_counts.append(all_partial_indices.shape[0])
            if sampling_successes[-1] == 0:
                rep_each_k = self.batch_size  # if no safe partials, use smaller batch size to sample more
                self.start_levels[k] = 1  # if all sampled partials are not safe at root level, start from level 1
            else:
                self.start_levels[k] = 0  # if there are safe partials, start from level 0
        return sampling_counts, sampling_successes, sampling_time

    def sample_all_k_on_root_non_batched(self):
        """
        Deprecated.
        """
        sampling_counts, sampling_successes , sampling_time = [], [], []
        rep_each_k = self.rep_each_k
        for k in range(self.eps, self.max_k + 1):
            all_partial_indices = batched_randsample(self.num_pixels, k, rep_each_k)
            t0 = perf_counter()
            success_cnt = 0
            for partial_indices in all_partial_indices:
                if self.check_partial(partial_indices, end_level=0)[0]:
                    success_cnt += 1
            sampling_successes.append(success_cnt)
            sampling_time.append(perf_counter() - t0)
            sampling_counts.append(all_partial_indices.shape[0])
            if sampling_successes[-1] == 0:
                rep_each_k = self.batch_size  # if no safe partials, use smaller batch size to sample more
                self.start_levels[k] = 1  # if all sampled partials are not safe at root level, start from level 1
            else:
                self.start_levels[k] = 0  # if there are safe partials, start from level 0
        return sampling_counts, sampling_successes, sampling_time

    def check_partial(self, partial_indices: torch.Tensor, start_level: int=0, end_level: int=0) -> tuple[bool, torch.Tensor|None]:
        """
        Deprecated.
        Speed up version of verify1 for a given partial indices.
        No demo, no seqAT.working_on context manager, no raising of TimeoutException and no cuda cache clearing.
        This method checks if the partial indices are safe from start_level to end_level.

        :param partial_indices: Indices of the pixels to be considered in the computation of diff.
        :type partial_indices: torch.Tensor with shape (num_indices,)
        :param start_level: The level to start verification. Default to self.start_levels[num_indices].
        :param end_level: The level to end verification. Default to 0 (root level).
        :return: A tuple (is_safe, unk_combs) where is_safe is True if the partial indices are safe til end_level, False otherwise. unk_combs is a tensor of shape (num_unk, self.eps) on cpu with the unknown combinations only if the partial is not safe til level=self.eps, None otherwise.
        """
        num_indices = partial_indices.shape[0]
        start_level = max(start_level, self.start_levels[num_indices])  # use the start level for this k
        assert end_level >= start_level, "end_level must be greater than or equal to start_level"
        bitmask_length = (num_indices + 63) // 64
        if start_level == 0: combs_to_check = torch.empty((1, 0), dtype=torch.long)
        else: combs_to_check = torch.combinations(torch.arange(num_indices), start_level)
        safe_bitmasks: torch.Tensor = torch.empty((0, bitmask_length), dtype=torch.long)
        for level in range(start_level, end_level + 1):
            num_to_check = combs_to_check.shape[0]
            if num_to_check == 0: return True, None
            unk_combs_lst: list[torch.Tensor] = []
            new_safe_bitmasks_lst: list[torch.Tensor] = [safe_bitmasks]
            with self.bounded_x0.partial(partial_indices):
                for i in range(0, num_to_check, self.batch_size):
                    combs_to_check_batch = combs_to_check[i:i + self.batch_size].to(self.device)
                    with self.bounded_x0.selecting(combs_to_check_batch):
                        safes = self.__check_current_batch()
                    new_safe_bitmasks_lst.append(combs2bitmasks(combs_to_check_batch[safes], bitmask_length).cpu())
                    unk_combs_lst.append(combs_to_check_batch[~safes].cpu())

            unk_combs = torch.cat(unk_combs_lst, dim=0)
            if unk_combs.shape[0] == 0: return True, None
            if level == self.eps: return False, unk_combs
            if level == end_level: return False, None
            safe_bitmasks = torch.cat(new_safe_bitmasks_lst, dim=0)
            combs_to_check = split_and_prune(unk_combs, safe_bitmasks, level, num_indices, False)
        assert False, "Should not reach here"

    def check_partials_on_root(self, partial_indices2d: torch.Tensor) -> torch.Tensor:
        """
        Check safe for all the partials on root level with batch processing. (batch on root technique)
        Returns a boolean tensor.

        :param partial_indices2d: Each row is a set of partial indices.
        :type partial_indices2d: torch.Tensor of shape (num_partials, num_indices)
        :return: A 1D tensor on cpu of shape (num_partials,) with True if the corresponding partial indices is safe, False otherwise.
        """
        assert partial_indices2d.ndim == 2, "partial_indices2d must be of shape (num_partials, num_indices)"
        num_partials = partial_indices2d.shape[0]
        # start_level = max(start_level, self.start_levels[num_indices])
        safes_book = torch.zeros(num_partials, dtype=torch.bool, device=self.device)
        if partial_indices2d.device != self.device:
            partial_indices2d = partial_indices2d.to(self.device)
        for i in range(0, num_partials, self.batch_size):
            with self.bounded_x0.partial(partial_indices2d[i:i + self.batch_size]):
                safes_book[i:i + self.batch_size] = self.__check_current_batch()
        return safes_book.cpu()

    def __handle_partials(self, num_indices: int, end_level: int=0):
        if self.stop_event.is_set(): return
        start = perf_counter()
        partial_indices2d = torch.tensor(self.batch_pool[num_indices])
        self.batch_pool[num_indices].clear()
        safes = self.check_partials_on_root(partial_indices2d=partial_indices2d)
        if num_indices == self.__level:  # concrete combs, cannot refine
            self.conn.send((num_indices, partial_indices2d.shape[0], safes.sum().item(), perf_counter() - start))
            if self.__level == self.eps:
                for suspect in partial_indices2d[~safes].tolist():
                    self.conn.send('adversarial-example-suspect')
                    self.conn.send(suspect)
            else:
                self.conn.send('concrete-combs')
                self.conn.send(partial_indices2d[~safes].numpy())
            return
        if end_level != 0:  # need to check in deeper level
            for unk_i in torch.where(~safes)[0]:
                safes[unk_i], unk_combs = self.check_partial(partial_indices=partial_indices2d[unk_i], start_level=1, end_level=end_level)
                if unk_combs is not None:  # not safe and end_level == self.eps (i.e., go sbap and cannot verify)
                    for suspect in unk_combs.tolist():
                        self.conn.send('adversarial-example-suspect')
                        self.conn.send(suspect)
        self.conn.send((num_indices, partial_indices2d.shape[0], safes.sum().item(), perf_counter() - start))
        if end_level != self.__level:  # need to refine
            refined_partials = []
            for unk_partial in partial_indices2d[~safes].tolist():
                refined_partials.extend(get_refinement_subsets(unk_partial, self.refinement_coverings[num_indices]))
            for refined_partial in refined_partials:
                self.__handle_partial(refined_partial)

    def __handle_partial_non_batched(self, partial_indices: list[int]):
        """
        Deprecated.
        """
        if self.stop_event.is_set(): return
        num_indices = len(partial_indices)
        # message: (num_indices, runs, successes, duration)
        if num_indices < self.eps:
            self.conn.send((num_indices, 1, 1, 0.))
        elif self.refinement_strategy[num_indices] is not None:  # check on root first, refine if not safe
            start = perf_counter()
            if self.check_partial(partial_indices=torch.tensor(partial_indices), end_level=0)[0]:
                self.conn.send((num_indices, 1, 1, perf_counter() - start))
            else:
                self.conn.send((num_indices, 1, 0, perf_counter() - start))
                if num_indices == self.eps:
                    self.conn.send('adversarial-example-suspect')
                    self.conn.send(partial_indices)
                else:
                    for partial in get_refinement_subsets(partial_indices, self.refinement_coverings[num_indices]):
                        self.__handle_partial_non_batched(partial)
        else:  # go sbap without batch on root, no refinement needed
            self.__handle_sbap_partial(partial_indices)

    def __handle_partial(self, partial_indices: list[int]):
        if self.stop_event.is_set(): return
        num_indices = len(partial_indices)
        # message: (num_indices, runs, successes, duration)
        if num_indices < self.__level:
            self.conn.send((num_indices, 1, 1, 0.))
        elif num_indices <= self.max_size_to_batch:  # batch on root
            self.batch_pool[num_indices].append(partial_indices)
            if len(self.batch_pool[num_indices]) == self.__greater_batch:
                self.__handle_partials(num_indices, end_level=0 if self.refinement_strategy[num_indices] else self.__level)
        elif self.refinement_strategy[num_indices]:  # no need to check on the root level (default to not safe), just refine
            self.conn.send((num_indices, 1, 0, 0.))
            for partial in get_refinement_subsets(partial_indices, self.refinement_coverings[num_indices]):
                self.__handle_partial(partial)
        else:  # go sbap directly without batch on root, no refinement needed
            self.__handle_sbap_partial(partial_indices)

    def __handle_sbap_partial(self, partial_indices: list[int]):
        num_indices = len(partial_indices)
        start = perf_counter()
        safe, unk_combs = self.check_partial(partial_indices=torch.tensor(partial_indices), end_level=self.eps)
        if safe:
            self.conn.send((num_indices, 1, 1, perf_counter() - start))
        else:  # go sbap and cannot verify
            self.conn.send((num_indices, 1, 0, perf_counter() - start))
            for partial in unk_combs.tolist():
                self.conn.send('adversarial-example-suspect')
                self.conn.send(partial)

    def __prove_geometry(self, geometry: Geometry, chosen_indices: list[int], num_workers: int, worker_id: int):
        num_of_blocks = math.floor(geometry.number_of_blocks() / num_workers)
        if worker_id < geometry.number_of_blocks() % num_workers:
            num_of_blocks += 1
        with Listener(('localhost', 0)) as blocks_listener:
            self.blocks_generator_process = launch_covering_generator(
                q=geometry.q, m=geometry.m, t=geometry.t,
                port=blocks_listener.address[1],
                chosen_indices=chosen_indices,
                num_workers=num_workers,
                worker_id=worker_id
            )
            with blocks_listener.accept() as blocks_conn:
                for _ in range(num_of_blocks):
                    if self.stop_event.is_set(): return
                    partial_indices = blocks_conn.recv()  # assert partial_indices is sorted and of type list[int]
                    self.__handle_partial(partial_indices)
                    # self.__handle_partial_non_batched(partial_indices)
                assert blocks_conn.recv() == 'done'
                blocks_conn.send('done')

        # handle remaining partials in the pool in descending order according to the key (num_indices). The batch_pool is created in this order
        for num_indices in self.batch_pool:
            if len(self.batch_pool[num_indices]) > 0:
                self.__handle_partials(num_indices,
                    end_level=0 if self.refinement_strategy[num_indices] else self.__level)

        self.conn.send('done')
        self.bounded_x0.level = None

    def __prove_concrete_combs(self, concrete_combs: torch.Tensor):
        unk_combs_lst: list[torch.Tensor] = [torch.empty((0, concrete_combs.shape[1]), dtype=torch.int64)]
        for i in range(0, concrete_combs.shape[0], self.batch_size):
            if self.stop_event.is_set(): break
            combs_batch = concrete_combs[i:i + self.batch_size].to(self.device)
            with self.bounded_x0.selecting(combs_batch):
                safes = self.__check_current_batch()
            unk_combs_lst.append(combs_batch[~safes].cpu())
        unk_combs = torch.cat(unk_combs_lst, dim=0).numpy()
        # send the unknown combinations to the main process
        self.conn.send(unk_combs)

    def work_as_worker(self, port: int, num_workers: int=1, worker_id: int=0):
        with Client(('localhost', port)) as self.conn:
            # Every iteration of this loop is one image
            message = self.conn.recv()
            while message != 'terminate':
                seed, image, label = message
                torch.manual_seed(seed)
                torch_img = torch.tensor(image, dtype=torch.float32).reshape(1, -1, 1)
                self.adapt_target_images(images=torch_img, true_labels=torch.tensor([label], dtype=torch.int64), check_first=False)

                self.conn.send(self.sample_all_level())
                message = self.conn.recv()
                max_sizes_to_batch: dict[int, int] = {}
                while message[0] == 'sample-all-k':
                    l = int(message[1])
                    self.bounded_x0.level = l
                    self.__level = l
                    self.conn.send(self.sample_all_k_on_root())
                    max_sizes_to_batch[l] = max(k for k, v in self.start_levels.items() if v == 0)
                    self.start_levels = defaultdict(lambda: 0)  # reset start levels
                    message = self.conn.recv()
                self.bounded_x0.level = None
                if message[0] == 'skip':
                    message = self.conn.recv()
                    continue
                if message[0] == 'sbap':
                    self.__level = int(message[1])
                    all_concrete_combs = torch.combinations(torch.arange(self.num_pixels), self.__level)
                    num2check_each = all_concrete_combs.shape[0] // num_workers
                    if worker_id == num_workers - 1:
                        concrete_combs = all_concrete_combs[worker_id * num2check_each:]
                    else:
                        concrete_combs = all_concrete_combs[worker_id * num2check_each: (worker_id + 1) * num2check_each]
                    del all_concrete_combs
                    prove_thread = threading.Thread(target=self.__prove_concrete_combs, args=(concrete_combs,), daemon=True)
                elif message[0] == 'coverd':
                    _, geometry, chosen_indices, self.refinement_strategy = message
                    self.__level = geometry.t
                    self.max_size_to_batch = max_sizes_to_batch[self.__level]
                    self.bounded_x0.level = self.__level
                    self.batch_pool: dict[int, list[list[int]]] = {k: [] for k in list(range(self.__level, self.max_size_to_batch + 1))[::-1]}
                    self.refinement_coverings = load_refinement_coverings(self.refinement_strategy)
                    prove_thread = threading.Thread(target=self.__prove_geometry, args=(geometry, chosen_indices, num_workers, worker_id), daemon=True)
                else: assert False, f"Unknown message {message}"
                self.stop_event = threading.Event()
                prove_thread.start()
                while True:
                    if self.conn.poll():
                        message = self.conn.recv()
                        if isinstance(message, str) and message == 'stop':
                            self.stop_event.set()
                            if hasattr(self, 'blocks_generator_process') and self.blocks_generator_process.poll() is None:  # means still running
                                self.blocks_generator_process.terminate()
                            break
                        self.__level += 1
                        prove_thread = threading.Thread(target=self.__prove_concrete_combs, args=(torch.from_numpy(message),), daemon=True)
                        prove_thread.start()
                prove_thread.join()
                self.conn.send('stopped')
                message = self.conn.recv()

if __name__ == '__main__':
    import argparse
    from utils.shared import add_shared_arguments, get_verifier
    parser = argparse.ArgumentParser(description='L0 Robustness Verifier: (GPU) Worker',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_shared_arguments(parser)
    parser.add_argument('--worker_id', type=int, default=0, help='The index of the worker')
    parser.add_argument('--num-workers', type=int, default=1, help='Total number of workers')
    parser.add_argument('--port', type=int, default=0, help='The port main verification process is listening on')
    args = parser.parse_args()

    verifier = get_verifier(args)
    verifier.work_as_worker(port=args.port, num_workers=args.num_workers, worker_id=args.worker_id)