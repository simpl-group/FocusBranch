from AbstractInterpreter import L0BoundedInput, SequentialAT
from split_and_prune import split_and_prune, prepare4pruning
from bimm import combs2bitmasks
from data import Strategy
from data.db import insert_mean_concretecheck_time, get_mean_concretecheck_time
from utils import DEVICE, TQDM_BAR_FORMAT, TimeoutException
from time import perf_counter
import threading
import math
import signal
import numpy as np
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

def selected_indices_sample(k: int, num_select: int, repetitions: int, num_workers: int=1, worker_id: int=0) -> torch.Tensor:
    """
    Generate a tensor of shape (<= repetitions, num_select) with distinct random combinations of indices from 0 to k-1.
    Each row is sorted in ascending order. The samples depend only on the torch seed, so every image gets its own
    reproducible sample regardless of the images verified before it.

    Small spaces are enumerated and split among the workers (striding), so that the workers sample disjoint rows;
    larger ones are sampled uniformly and deduplicated.
    """
    if math.comb(k, num_select) <= 4 * repetitions * num_workers:
        all_combs = torch.combinations(torch.arange(k, device=DEVICE), num_select)[worker_id::num_workers]
        if len(all_combs) > repetitions:
            all_combs = all_combs[torch.randperm(len(all_combs), device=DEVICE)[:repetitions]]
        return all_combs.cpu()
    samples = batched_randsample(k, num_select, repetitions).unique(dim=0)
    while samples.shape[0] < repetitions:
        samples = torch.cat((samples, batched_randsample(k, num_select, repetitions - samples.shape[0]))).unique(dim=0)
    return samples

class BoundingEngine:
    def __init__(self,
        torch_net: torch.nn.Module,
        seqAT: SequentialAT,
        bounded_x0: L0BoundedInput,
        concretecheck_batch: int=200,
        supercheck_batch: int=200,
        max_k: int=200,
        rep_each_k: int=50,
        rep_each_level: int=5000,
        pool_size: int=2000,
        timeout: int=0,
        strategy: Strategy=Strategy.FocusBranch,
        demo: bool=False
    ) -> None:
        self.torch_net = torch_net.cpu()
        self.seqAT = seqAT
        self.bounded_x0 = bounded_x0
        self.eps = self.bounded_x0.eps
        self.num_pixels = self.bounded_x0.num_pixels
        self.device = self.bounded_x0.device
        self.concretecheck_batch = concretecheck_batch
        self.supercheck_batch = supercheck_batch
        self.max_k = max_k
        self.rep_each_k = rep_each_k
        self.rep_each_level = rep_each_level
        self.timeout = timeout
        self.demo = demo
        self.pool_size = pool_size
        self.strategy = strategy
        self.max_batch_size = max(concretecheck_batch, supercheck_batch)  # lowered when the GPU runs out of memory

    def bound_current_batch(self) -> torch.Tensor:
        """
        Check if each of the current batch is safe.

        :return: A 1D Tensor on self.device of shape (batch_size,) with True if the corresponding input is safe, False otherwise.
        """
        self.seqAT.update_activation_rp(bounded_in=self.bounded_x0)
        lrp = self.seqAT.tail.backsubstitution_lrp()
        lb_l = self.bounded_x0.compute_lbound(lrp).squeeze(2)
        return lb_l.gt(0).all(dim=1)  # safe iff the true label beats every other label

    def bound_nodes(self, focus: torch.Tensor, level: int | None = None, batch_size: int | None = None) -> torch.Tensor:
        """
        Bound the nodes (focus[i], level) in batches.

        :param focus: (num_nodes, |S|) focus sets.
        :param level: The level shared by the nodes. None for concrete nodes (level |S|), which are bounded by selecting
            their pixels; otherwise the nodes are bounded as super nodes, restricted to their focus sets.
        :param batch_size: Defaults to the concrete or super batch size depending on the node type.
        :return: A boolean tensor on self.device, True for the nodes proven safe.
        """
        if batch_size is None:
            batch_size = self.concretecheck_batch if level is None else self.supercheck_batch
        batch_size = min(batch_size, self.max_batch_size)
        safes = torch.empty(focus.shape[0], dtype=torch.bool, device=self.device)
        if focus.device != self.device:
            focus = focus.to(self.device)
        self.bounded_x0.level = level
        try:
            i = 0
            while i < focus.shape[0]:
                try:
                    with self.bounded_x0.selecting(focus[i:i + batch_size]) if level is None else self.bounded_x0.partial(focus[i:i + batch_size]):
                        safes[i:i + batch_size] = self.bound_current_batch()
                except torch.OutOfMemoryError:
                    if batch_size == 1: raise
                    self.seqAT.remove_activation_rp()
                    torch.cuda.empty_cache()
                    batch_size = self.max_batch_size = max(1, batch_size // 2)  # sticky: the GPU is shared or too small
                    print(f"CUDA out of memory, bounding at most {batch_size} nodes per batch from now on", flush=True)
                    continue
                i += batch_size
        finally:
            self.bounded_x0.level = None
        return safes

    def unknown_concrete(self, combs: torch.Tensor, stop_event: threading.Event | None = None) -> np.ndarray:
        """The concrete nodes among `combs` that cannot be proven safe (the ones bounded so far if stopped early)."""
        unknown: list[torch.Tensor] = [torch.empty((0, combs.shape[1]), dtype=torch.int64)]
        for i in range(0, combs.shape[0], self.concretecheck_batch):
            if stop_event is not None and stop_event.is_set(): break
            batch = combs[i:i + self.concretecheck_batch].to(self.device)
            unknown.append(batch[~self.bound_nodes(batch)].cpu())
        return torch.cat(unknown, dim=0).numpy()

    def first_safe_level(self) -> int:
        for level in range(self.eps + 1):
            all_safe = True
            if level == 0:
                focus_sets = torch.empty((1, 0), dtype=torch.int64)
            elif level == 1:
                focus_sets = torch.arange(self.num_pixels).reshape(-1, 1)
            else:
                focus_sets = selected_indices_sample(self.num_pixels, level, self.rep_each_level)
            for i in range(0, focus_sets.shape[0], self.concretecheck_batch):
                focus_sets_batch = focus_sets[i:i + self.concretecheck_batch].to(self.device)
                with self.bounded_x0.selecting(focus_sets_batch):
                    safes = self.bound_current_batch()
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
        last_safe_rate = 0.
        first_any_safe_level = start_level
        if self.demo:
            sampled_nodes = 0
            start = perf_counter()
        with self.bounded_x0.partial(partial_indices):
            for level in range(start_level, self.eps + 1):
                if level == 0:
                    focus_sets = torch.empty((1, 0), dtype=torch.int64)
                    estimated_nodes_this_level = 1
                elif level == 1:
                    focus_sets = torch.arange(k).reshape(-1, 1)
                    estimated_nodes_this_level = k
                else:
                    focus_sets = selected_indices_sample(k, level, self.rep_each_level)
                    estimated_nodes_this_level = math.ceil(math.comb(k, level) * (1 - last_safe_rate) ** 2)
                sample_nodes_this_level = focus_sets.shape[0]
                num_safe_nodes = 0
                level_start = perf_counter()
                for i in range(0, sample_nodes_this_level, self.concretecheck_batch):
                    focus_sets_batch = focus_sets[i:i + self.concretecheck_batch].to(self.device)
                    with self.bounded_x0.selecting(focus_sets_batch):
                        num_safe_nodes += int(self.bound_current_batch().sum().item())

                if num_safe_nodes == 0: first_any_safe_level += 1
                current_speed = sample_nodes_this_level / (perf_counter() - level_start)
                if level != self.eps: nodes_per_second = max(current_speed, nodes_per_second)  # not the last level
                last_safe_rate = num_safe_nodes / sample_nodes_this_level
                estimated_total_nodes += estimated_nodes_this_level
                if self.demo:
                    sampled_nodes += sample_nodes_this_level
                    print(f"Level {level}: {100*last_safe_rate:>6.2f}% safe, speed: {current_speed:>8.2f} nodes/s, estimated nodes: {estimated_nodes_this_level}")
                if num_safe_nodes == sample_nodes_this_level:
                    break  # All nodes are safe, no need to go deeper
        estimated_verification_time = estimated_total_nodes / nodes_per_second
        if self.demo: print(f"Sample Time: {perf_counter() - start:2.2}s, Sampled Nodes: {sampled_nodes}, Speed: {nodes_per_second:.2f} nodes/s, Estimated nodes: {estimated_total_nodes}\nEstimated verification time: {estimated_verification_time:.2f}s, First any safe level: {first_any_safe_level}")
        return estimated_verification_time, first_any_safe_level

    def probeps(self, start_level: int=1) -> int:
        focus_sets = torch.arange(self.num_pixels).reshape(-1, 1)
        start = perf_counter()
        for level in range(start_level, self.eps + 1):
            sample_nodes_this_level = focus_sets.shape[0]
            num_safe_nodes = 0
            all_lb_l_list = []
            for i in range(0, sample_nodes_this_level, self.concretecheck_batch):
                focus_sets_batch = focus_sets[i:i + self.concretecheck_batch].to(self.device)
                with self.bounded_x0.selecting(focus_sets_batch):
                    self.seqAT.update_activation_rp(bounded_in=self.bounded_x0)
                    lb_l = self.bounded_x0.compute_lbound(self.seqAT.tail.backsubstitution_lrp()).squeeze(2).cpu()
                    all_lb_l_list.append(lb_l)
                    num_safe_nodes += int(lb_l.gt(0).all(dim=1).sum().item())
            all_lb_l = torch.cat(all_lb_l_list, dim=0)
            lb_l_min_values = all_lb_l.min(dim=1).values
            num_lt_0 = int(lb_l_min_values.le(0).sum().item())
            print(f"Level {level} min: {lb_l_min_values.min().item():.2f}\t(safe:unk)/all: ({num_safe_nodes}:{num_lt_0})/{sample_nodes_this_level}, using {perf_counter() - start:.2f}s")
            start = perf_counter()
            least_indices = focus_sets[lb_l_min_values.topk(min(5, num_lt_0), largest=False, sorted=False).indices]
            if num_safe_nodes == sample_nodes_this_level:
                return 0
            if level == self.eps:
                return num_lt_0
            focus_sets, _, _ = prepare4pruning(least_indices, self.num_pixels)
        assert False, "Should never reach here"

    def min_margin_lbs(self, concrete_combs: torch.Tensor) -> torch.Tensor:
        """The smallest margin lower bound of each concrete node (on cpu); a node is unknown iff it is <= 0."""
        lb_l_min_values_lst: list[torch.Tensor] = [torch.empty((0), dtype=torch.float32)]
        for i in range(0, concrete_combs.shape[0], self.concretecheck_batch):
            combs_batch = concrete_combs[i:i + self.concretecheck_batch].to(self.device)
            with self.bounded_x0.selecting(combs_batch):
                self.seqAT.update_activation_rp(bounded_in=self.bounded_x0)
                lb_l = self.bounded_x0.compute_lbound(self.seqAT.tail.backsubstitution_lrp()).squeeze(2)
                lb_l_min_values_lst.append(lb_l.min(dim=1).values.cpu())
        return torch.cat(lb_l_min_values_lst, dim=0)

    def adapt_target_images(self, images: torch.Tensor, true_label: int, check_first=True) -> None:
        """
        Set the image to verify and its label: the output rows become the margins of the true label over the others.
        Will first check if the image matches the true label if check_first is True.

        :param images: The input image, of shape (1, channel, height, width), or (1, -1, 1) if check_first is False.
        :param true_label: The true label of the image.
        :param check_first: If True, will assert that the input image matches the true label.
        """
        if check_first:
            assert images.ndim == 4, "Input images must be of shape (batch_size, channel, height, width)."
            assert self.torch_net(images.cpu()).argmax(dim=1).item() == true_label, "The input image does not match the true label."
            images = images.flatten(start_dim=1).unsqueeze(2)
        assert images.ndim == 3, "Input images must be of shape (batch_size, -1, 1)."
        self.bounded_x0.update_x(images)  # (batch, d1, 1)
        self.seqAT.tail.make_marginals(true_label=true_label, device=self.device)
        self.max_sizes_to_batch: dict[int, int] = {}  # key: level, value: max size to batch at this level

    def verify1(self, partial_indices: torch.Tensor | None=None, start_level: int=0) -> bool:
        num_indices = self.num_pixels
        if partial_indices is not None:
            num_indices = partial_indices.shape[0]
        bitmask_length = (num_indices + 63) // 64
        if math.comb(num_indices, start_level) > 10 * 1024 ** 3 * 8 * num_indices:
            return False
        focus_sets = torch.combinations(torch.arange(num_indices), start_level)
        if start_level == 0: focus_sets = torch.empty((1, 0), dtype=torch.int64)
        safe_bitmasks: torch.Tensor = torch.empty((0, bitmask_length), dtype=torch.long)
        success = False
        if self.demo:
            start = perf_counter()
            branching_time, total_nodes = 0.0, 0
        for level in range(start_level, self.eps + 1):
            num_to_check = focus_sets.shape[0]
            if num_to_check == 0:
                success = True
                break
            unk_combs_lst: list[torch.Tensor] = []
            new_safe_bitmasks_lst: list[torch.Tensor] = [safe_bitmasks]
            if self.demo:
                safe_cnt, unk_cnt = 0, 0
            level_start = perf_counter()
            pbar = tqdm(total=num_to_check, desc=f"Level {level}", bar_format=TQDM_BAR_FORMAT, disable=not self.demo)
            with self.seqAT.working_on(self.device), self.bounded_x0.partial(partial_indices):
                for i in range(0, num_to_check, self.concretecheck_batch):
                    focus_sets_batch = focus_sets[i:i + self.concretecheck_batch].to(self.device)
                    with self.bounded_x0.selecting(focus_sets_batch):
                        self.seqAT.update_activation_rp(bounded_in=self.bounded_x0)
                        lrp = self.seqAT.tail.backsubstitution_lrp()
                        lb_l = self.bounded_x0.compute_lbound(lrp).squeeze(2)

                    safes = lb_l.gt(0).all(dim=1)
                    new_safe_bitmasks_lst.append(combs2bitmasks(focus_sets_batch[safes], bitmask_length).cpu())
                    unk_combs_lst.append(focus_sets_batch[~safes].cpu())

                    if self.demo:
                        safe_cnt += new_safe_bitmasks_lst[-1].shape[0]
                        unk_cnt += unk_combs_lst[-1].shape[0]
                        posfix: dict = {"Safe:UNK": f"{safe_cnt}:{unk_cnt}"}
                        pbar.set_postfix(posfix)
                        pbar.update(focus_sets_batch.shape[0])
                    if self.timeout != 0 and i == self.concretecheck_batch * 100:  # calculate remaining time to decide if to early stop
                        if not self.demo: # calculate remaining time without pbar
                            rate = (perf_counter() - level_start) / 100
                            if ((num_to_check - i) / self.concretecheck_batch) * rate > self.timeout:
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
            del focus_sets, focus_sets_batch, lb_l, lrp, new_safe_bitmasks_lst, unk_combs_lst, safes
            torch.cuda.empty_cache()
            level_start = perf_counter()
            focus_sets = split_and_prune(unk_combs, safe_bitmasks, level, num_indices, self.demo)
            if self.demo: branching_time += perf_counter() - level_start
            del unk_combs
            torch.cuda.empty_cache()
        if self.demo:
            print(f"Total: Nodes {total_nodes}; Time {perf_counter() - start:.2f}s; Branching Time {branching_time:.2f}s")
            print(f"Safe: {success}")
        return success

    def estimate_mean_boundconcrete_time(self, netname: str, sample_size_coeff: int=10, force_reestimate: bool=False) -> list[float]:
        mean_boundconcrete_time = [0.0] * (self.eps + 1)
        start_level = 0
        if not force_reestimate:
            for level in range(self.eps + 1):
                mean_time = get_mean_concretecheck_time(net=netname, batch_size=self.concretecheck_batch, eps=self.eps, level=level)
                if mean_time is not None:
                    mean_boundconcrete_time[level] = mean_time
                    start_level = level + 1
                else: break
            if start_level > self.eps:
                return mean_boundconcrete_time
        print(f"Estimating mean ConcreteCheck time for net={netname}, batch_size={self.concretecheck_batch}, eps={self.eps} on level", end="")
        sample_size = sample_size_coeff * self.concretecheck_batch
        for level in range(start_level, self.eps + 1):
            print(f" {level}: ", end="", flush=True)
            focus_sets = selected_indices_sample(self.num_pixels, level, sample_size) if level > 0 else torch.empty((1, 0), dtype=torch.int64)
            start = perf_counter()
            self.bound_nodes(focus_sets)
            mean_boundconcrete_time[level] = (perf_counter() - start) / focus_sets.shape[0]
            print(f"{mean_boundconcrete_time[level]:.7f}s,", end="" if level != self.eps else "\n", flush=True)
            insert_mean_concretecheck_time(net=netname, batch_size=self.concretecheck_batch, eps=self.eps, level=level, mean_time=mean_boundconcrete_time[level])
        self.bounded_x0.clear_cache()
        self.seqAT.remove_activation_rp()
        torch.cuda.empty_cache()
        return mean_boundconcrete_time

    def check_partial(self, partial_indices: torch.Tensor, start_level: int=0, end_level: int=0) -> tuple[bool, torch.Tensor|None]:
        """
        Deprecated but worth keeping for reference.
        Speed up version of verify1 for a given partial indices.
        No demo, no seqAT.working_on context manager, no raising of TimeoutException and no cuda cache clearing.
        This method checks if the partial indices are safe from start_level to end_level.

        :param partial_indices: Indices of the pixels to be considered in the computation of diff.
        :type partial_indices: torch.Tensor with shape (num_indices,)
        :param start_level: The level to start verification.
        :param end_level: The level to end verification. Default to 0 (root level).
        :return: A tuple (is_safe, unk_combs) where is_safe is True if the partial indices are safe til end_level, False otherwise. unk_combs is a tensor of shape (num_unk, self.eps) on cpu with the unknown combinations only if the partial is not safe til level=self.eps, None otherwise.
        """
        num_indices = partial_indices.shape[0]
        assert end_level >= start_level, "end_level must be greater than or equal to start_level"
        bitmask_length = (num_indices + 63) // 64
        if start_level == 0: focus_sets = torch.empty((1, 0), dtype=torch.long)
        else: focus_sets = torch.combinations(torch.arange(num_indices), start_level)
        safe_bitmasks: torch.Tensor = torch.empty((0, bitmask_length), dtype=torch.long)
        for level in range(start_level, end_level + 1):
            num_to_check = focus_sets.shape[0]
            if num_to_check == 0: return True, None
            unk_combs_lst: list[torch.Tensor] = []
            new_safe_bitmasks_lst: list[torch.Tensor] = [safe_bitmasks]
            with self.bounded_x0.partial(partial_indices):
                for i in range(0, num_to_check, self.concretecheck_batch):
                    focus_sets_batch = focus_sets[i:i + self.concretecheck_batch].to(self.device)
                    with self.bounded_x0.selecting(focus_sets_batch):
                        safes = self.bound_current_batch()
                    new_safe_bitmasks_lst.append(combs2bitmasks(focus_sets_batch[safes], bitmask_length).cpu())
                    unk_combs_lst.append(focus_sets_batch[~safes].cpu())

            unk_combs = torch.cat(unk_combs_lst, dim=0)
            if unk_combs.shape[0] == 0: return True, None
            if level == self.eps: return False, unk_combs
            if level == end_level: return False, None
            safe_bitmasks = torch.cat(new_safe_bitmasks_lst, dim=0)
            focus_sets = split_and_prune(unk_combs, safe_bitmasks, level, num_indices, False)
        assert False, "Should not reach here"

    def bound_supernodes(self, supersets: torch.Tensor, level: int) -> torch.Tensor:
        """
        Check safe for given super nodes (supersets[i], level) with batch processing. (batched SuperCheck)
        No splitting throughout sub-levels for any super node.
        Returns a boolean tensor on cpu.

        :param supersets: Each row is the focus set of a node.
        :type supersets: torch.Tensor of shape (num_supersets, num_indices)
        :param level: The level shared by all the nodes.
        :return: A 1D tensor on cpu of shape (num_supersets,) with True if the corresponding node is safe, False otherwise.
        """
        assert supersets.ndim == 2, "supersets must be of shape (num_supersets, num_indices)"
        return self.bound_nodes(supersets, level, self.supercheck_batch).cpu()
