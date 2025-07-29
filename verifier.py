from AbstractInterpreter import L0BoundedInput, SequentialAT
from split_and_prune import split_and_prune
from bimm import combs2bitmasks
from utils import DEVICE, TQDM_BAR_FORMAT, TimeoutException
from time import perf_counter
from functools import lru_cache
from collections import defaultdict
import math
import signal
import torch
from tqdm import tqdm


def batched_randsample(upper_bound: int, sample_size: int, batch_size: int) -> torch.Tensor:
    """
    Generate a tensor of shape (batch_size, sample_size) with random indices from 0 to upper_bound-1.
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
        self.start_level: defaultdict[int, int] = defaultdict(lambda: 0)  #key: k, value: start level to probe and verify

    def check_safe_batched(self) -> torch.Tensor:
        """
        Check each of the current batch is safe.

        :return: A tensor of shape (batch_size,) with True if the corresponding input is safe, False otherwise.
        """
        self.seqAT.update_activation_rp(bounded_in=self.bounded_x0)
        lrp = self.seqAT.tail.backsubstitution_lrp()
        lb_l = self.bounded_x0.compute_lbound(lrp).squeeze(2)
        return lb_l.ge(0).all(dim=1)

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
        with self.seqAT.working_on(self.device), self.bounded_x0.partial(partial_indices):
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
                        num_safe_nodes += int(self.check_safe_batched().sum().item())

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

    def verify1(self, partial_indices: torch.Tensor | None=None, start_level: int=0) -> bool:
        num_indices = self.num_pixels
        if partial_indices is not None:
            num_indices = partial_indices.shape[0]
            if start_level == 0:
                start_level = self.start_level[num_indices]  # use the start level for this k
        bitmask_length = (num_indices + 63) // 64
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

    def sample_all_k(self, baseline: dict[int, float] | None=None) -> dict[int, float]:
        """
        Sample all k from eps to max_k self.rep_each_k times and return the mean estimated verification time for each k.
        If baseline is provided, it will stop sampling when the estimated time exceeds the baseline for that k.

        :param baseline: A dictionary with k as keys and baseline verification time as values. If None, no baseline is used.
        :type baseline: dict[int, float] | None
        :return: A dictionary with k as keys and estimated verification time as values.
        """
        estimated_times: dict[int, float] = {}
        pbar = tqdm(total=self.max_k - self.eps + 1, desc="Sampling k", disable=not self.demo, bar_format=TQDM_BAR_FORMAT)
        self.demo = False
        for k in range(self.eps, self.max_k + 1):
            all_partial_indices = batched_randsample(self.num_pixels, k, self.rep_each_k)
            self.start_level[k] = start_level = self.start_level[k-1]  # init from the previous k
            batch_on_root_mean = 0.
            if self.start_level[k] == 0:  # only batch on root level if start level is 0, otherwise skip this part
                root_unk_list: list[torch.Tensor] = []
                batch_on_root_start = perf_counter()
                with self.seqAT.working_on(self.device):
                    for i in range(0, all_partial_indices.shape[0], self.batch_size):
                        batched_partial_indices = all_partial_indices[i:i + self.batch_size].to(self.device)
                        with self.bounded_x0.partial(batched_partial_indices):  # batch on root level
                            safes = self.check_safe_batched()
                        root_unk_list.append(batched_partial_indices[~safes])
                batch_on_root_mean = (perf_counter() - batch_on_root_start) / self.rep_each_k
                all_partial_indices = torch.vstack(root_unk_list)  # root level unk partial indices
                if all_partial_indices.shape[0] == self.rep_each_k:
                    self.start_level[k] = 1  # if all sampled partial are not safe at root level, start from level 1
                start_level = 1  # for the probe, since level 0 is already checked
            num_actual_start_level_increased = 0
            accumulated_time = 0.
            for i, partial_indices in enumerate(all_partial_indices):
                estimated_time, actual_start_level = self.probe(partial_indices, start_level)
                accumulated_time += estimated_time
                if actual_start_level > start_level:
                    num_actual_start_level_increased += 1
            if num_actual_start_level_increased == self.rep_each_k:
                self.start_level[k] += 1
            estimated_times[k] = batch_on_root_mean + accumulated_time / self.rep_each_k
            if k != self.eps and baseline is not None and estimated_times[k] > baseline[k] * 0.4:  # tunable coefficient
                break
            pbar.set_postfix({"k": k, "Estimated Time": f"{estimated_times[k]:.4f}s",
                "Start Level": self.start_level[k], "Start Level Increased": num_actual_start_level_increased})
            pbar.update(1)
        pbar.close()
        if k != self.max_k:
            self.start_level.default_factory = lambda: self.start_level[k]  # set the default factory to the last start level

        return estimated_times
