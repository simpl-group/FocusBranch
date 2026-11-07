from AbstractInterpreter import L0BoundedInput, SequentialAT
from data.db import insert_mean_concretecheck_time, get_mean_concretecheck_time
from utils import DEVICE
from time import perf_counter
import threading
import math
import numpy as np
import torch


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
        pool_factor: int=2,  # pool size = supercheck_batch x pool_factor
        timeout: int=0,
        level_eps_only: bool=False,
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
        assert pool_factor >= 1, "--pool-factor must be at least 1"
        self.pool_size = supercheck_batch * pool_factor
        self.level_eps_only = level_eps_only
        """--level-eps-only: no probing or concrete sampling in planning, super branching from level eps only"""
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

    def adapt_target_images(self, images: torch.Tensor, true_label: int) -> None:
        """
        Set the image to verify and its label: the output rows become the margins of the true label over the others.

        :param images: (1, num_features, 1) the flattened image; the verifier verifies one image at a time.
        :param true_label: The label of the image.
        """
        assert images.ndim == 3, "Input images must be of shape (batch_size, -1, 1)."
        self.bounded_x0.update_x(images)  # (batch, d1, 1)
        self.seqAT.tail.make_marginals(true_label=true_label, device=self.device)
        self.max_sizes_to_batch: dict[int, int] = {}  # key: level, value: max size to batch at this level

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
