from AbstractInterpreter import L0BoundedInput, SequentialAT
from split_and_prune import split_and_prune, prepare4pruning, prune
from bimm import combs2bitmasks
from covering_generator import launch_covering_generator
from data import load_covering
from utils import DEVICE, TQDM_BAR_FORMAT, TimeoutException
from coverd_finite_geometries import Geometry
from multiprocessing.connection import Client, Listener
from time import perf_counter
from functools import lru_cache
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

def get_refinement_subsets(indices_set: list[int], covering: list[list[int]]) -> list[list[int]]:
    return [[indices_set[i] for i in cover] for cover in covering]

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
        rep_each_level: int=5000,
        pool_size: int=2000,
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
        self.__pool_size = pool_size

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
                combs_to_check = selected_indices_sample(self.num_pixels, level, self.rep_each_level)
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
        self.max_sizes_to_batch: dict[int, int] = {}  # key: level, value: max size to batch at this level

    def verify1(self, partial_indices: torch.Tensor | None=None, start_level: int=0) -> bool:
        num_indices = self.num_pixels
        if partial_indices is not None:
            num_indices = partial_indices.shape[0]
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

    def concrete_sample_all_levels(self) -> tuple[list[int], list[int], list[float]]:
        assert self.bounded_x0.level is None, "bounded_x0.level must be None since we are using self.bounded_x0.selecting"
        sample_counts, sample_successes = [0] * (self.eps + 1), [0] * (self.eps + 1)
        sample_durations = [0.0] * (self.eps + 1)
        for level in range(self.eps + 1):
            sample_start = perf_counter()
            if level == 0:
                combs_to_check = torch.empty((1, 0), dtype=torch.int64)
            else:
                combs_to_check = selected_indices_sample(self.num_pixels, level, self.rep_each_level)
            sample_counts[level] = combs_to_check.shape[0]
            for i in range(0, combs_to_check.shape[0], self.batch_size):
                combs_to_check_batch = combs_to_check[i:i + self.batch_size].to(self.device)
                with self.bounded_x0.selecting(combs_to_check_batch):
                    sample_successes[level] += int(self.__check_current_batch().sum().item())
            sample_durations[level] = perf_counter() - sample_start
            if sample_successes[level] == math.comb(self.num_pixels, level): break  # All nodes are safe, no need to go deeper
        return sample_counts, sample_successes, sample_durations

    def superset_sample_all_k(self, level: int) -> tuple[list[int], list[int], list[float]]:
        self.bounded_x0.level = level
        sampling_counts, sampling_successes , sampling_time = [], [], []
        rep_each_k = self.rep_each_k
        for k in range(level, self.max_k + 1):
            k_sized_supersets = batched_randsample(self.num_pixels, k, rep_each_k)
            batched_super_check_start = perf_counter()
            sampling_successes.append(self.check_supersets(k_sized_supersets).sum().item())
            sampling_time.append(perf_counter() - batched_super_check_start)
            sampling_counts.append(k_sized_supersets.shape[0])
            if sampling_successes[-1] == 0:
                rep_each_k = self.rep_each_k // 20  # if no safe supersets, use smaller batch size to sample more
            else:
                self.max_sizes_to_batch[level] = k
        self.bounded_x0.level = None
        return sampling_counts, sampling_successes, sampling_time

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

    def check_supersets(self, supersets: torch.Tensor) -> torch.Tensor:
        """
        Check safe for given supersets with batch processing. (batched super check)
        bounded_x0.level should be set carefully before calling this method.
        No splitting throughout sub-levels for any superset (current check can be seen as sub-level 0, deepest sub-level equals bounded_x0.level).
        Returns a boolean tensor on cpu.

        :param supersets: Each row is a set of partial indices.
        :type supersets: torch.Tensor of shape (num_supersets, num_indices)
        :return: A 1D tensor on cpu of shape (num_supersets,) with True if the corresponding partial indices is safe, False otherwise.
        """
        assert supersets.ndim == 2, "supersets must be of shape (num_supersets, num_indices)"
        num_supersets = supersets.shape[0]
        safes_book = torch.zeros(num_supersets, dtype=torch.bool, device=self.device)
        if supersets.device != self.device:
            supersets = supersets.to(self.device)
        for i in range(0, num_supersets, self.batch_size):
            with self.bounded_x0.partial(supersets[i:i + self.batch_size]):
                safes_book[i:i + self.batch_size] = self.__check_current_batch()
        return safes_book.cpu()

    def __handle_supersets(self, num_indices: int):
        if self.stop_event.is_set(): return
        start = perf_counter()
        supersets = torch.tensor(self.batch_pool[num_indices])
        self.batch_pool[num_indices].clear()
        safes = self.check_supersets(supersets=supersets)
        self.conn.send((num_indices, supersets.shape[0], safes.sum().item(), perf_counter() - start))
        if num_indices == self.__level:  # concrete combs, cannot refine
            if self.__level == self.eps:
                for suspect in supersets[~safes].tolist():
                    self.conn.send('adversarial-example-suspect')
                    self.conn.send(suspect)
            else:
                self.conn.send('unknown-combs')
                self.conn.send(supersets[~safes].numpy())
            return
        if not safes.all():  # need to refine
            refined_supersets = []
            for unk_superset in supersets[~safes].tolist():
                refined_supersets.extend(get_refinement_subsets(unk_superset, self.refinement_coverings[num_indices]))
            for refined_superset in refined_supersets:
                self.__handle_superset(refined_superset)

    def __handle_superset(self, superset: list[int]):
        if self.stop_event.is_set(): return
        num_indices = len(superset)
        # message: (num_indices, runs, successes, duration)
        if num_indices < self.__level:
            self.conn.send((num_indices, 1, 1, 0.))
        elif num_indices <= self.max_size_to_batch:  # batched super check
            self.batch_pool[num_indices].append(superset)
            if len(self.batch_pool[num_indices]) == self.__pool_size:
                self.__handle_supersets(num_indices)
        else:  # no need to do super check (default to not safe), just refine
            self.conn.send((num_indices, 1, 0, 0.))
            for refined_superset in get_refinement_subsets(superset, self.refinement_coverings[num_indices]):
                self.__handle_superset(refined_superset)

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
                    superset = blocks_conn.recv()  # assert superset is sorted and of type list[int]
                    self.__handle_superset(superset)
                assert blocks_conn.recv() == 'done'
                blocks_conn.send('done')

        # handle remaining supersets in the pool in descending order according to the key (num_indices). The batch_pool is created in this order
        for num_indices in self.batch_pool:
            if len(self.batch_pool[num_indices]) > 0:
                self.__handle_supersets(num_indices)
        self.conn.send('done')

    def __safe_prove_geometry(self, *args, **kwargs):
        try:
            self.bounded_x0.level = self.__level
            self.__prove_geometry(*args, **kwargs)
        except EOFError:
            pass  # self.blocks_generator_process terminated early
        finally:
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
            while not isinstance(message, str) or message != 'terminate':
                seed, image, label = message
                torch.manual_seed(seed)
                torch_img = torch.tensor(image, dtype=torch.float32).reshape(1, -1, 1)
                self.adapt_target_images(images=torch_img, true_labels=torch.tensor([label], dtype=torch.int64), check_first=False)

                self.conn.send(self.concrete_sample_all_levels())
                message = self.conn.recv()
                while message[0] == 'sample-all-k':
                    l = int(message[1])
                    self.conn.send(self.superset_sample_all_k(l))
                    message = self.conn.recv()
                if message[0] == 'skip':
                    message = self.conn.recv()
                    continue
                if message[0] == 'CCSaP':
                    self.__level = int(message[1])
                    all_concrete_combs = torch.combinations(torch.arange(self.num_pixels), self.__level)
                    num2check_each = all_concrete_combs.shape[0] // num_workers
                    if worker_id == num_workers - 1:
                        concrete_combs = all_concrete_combs[worker_id * num2check_each:]
                    else:
                        concrete_combs = all_concrete_combs[worker_id * num2check_each: (worker_id + 1) * num2check_each]
                    del all_concrete_combs
                    prove_thread = threading.Thread(target=self.__prove_concrete_combs, args=(concrete_combs,), daemon=True)
                elif message[0] == 'SCSaP':
                    _, geometry, chosen_indices, self.refinement_strategy = message
                    self.__level = geometry.t
                    self.max_size_to_batch = self.max_sizes_to_batch[self.__level]
                    self.batch_pool: dict[int, list[list[int]]] = {k: [] for k in list(range(self.__level, self.max_size_to_batch + 1))[::-1]}
                    self.refinement_coverings = load_refinement_coverings(self.refinement_strategy)
                    prove_thread = threading.Thread(target=self.__safe_prove_geometry, args=(geometry, chosen_indices, num_workers, worker_id), daemon=True)
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
                        if message[0] == 'SaP&CC':
                            new_combs, reference_bitmasks, target_s_cnt = prepare4pruning(torch.from_numpy(message[1]), self.num_pixels)
                            # stepping on new_combs before pruning can make each worker's pruned_combs.shape[0] roughly the same, which can slightly speed up the proving process since the workload is more balanced among workers.
                            new_combs = new_combs.to(self.device)[worker_id::num_workers].contiguous() # faster than on CPU.
                            pruned_combs = prune(new_combs, reference_bitmasks, target_s_cnt, self.__level)
                            self.conn.send(pruned_combs.shape[0])
                        elif message[0] == 'CC':
                            pruned_combs = torch.from_numpy(message[1])
                        else: assert False, f"Unknown message {message}"
                        self.__level += 1
                        prove_thread = threading.Thread(target=self.__prove_concrete_combs, args=(pruned_combs,), daemon=True)
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
    parser.add_argument('--pool-size', type=int, default=2000, help='The size of the batch pool')
    args = parser.parse_args()

    verifier = get_verifier(args)
    verifier.work_as_worker(port=args.port, num_workers=args.num_workers, worker_id=args.worker_id)