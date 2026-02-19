import time
import math
import torch
from tqdm import tqdm
from bimm import combs2bitmasks, bitmasks2combs, superset_sum
from utils import DEVICE, TQDM_BAR_FORMAT

chunk_size = 4000  # Adjust based on GPU memory availability
MEM_BUDGET = 128 * 1024**3  # 128 Giga Bytes

def split_and_prune(
    unk_combs: torch.Tensor,
    safe_bitmasks: torch.Tensor,
    level: int,
    num_indices: int,
    demo: bool=False,
) -> torch.Tensor:
    """
    Deprecated. Keep for reference.
    """
    # unk_combs are naturally sorted in each row
    last_item = unk_combs[:, -1] if level > 0 else torch.tensor([-1], dtype=torch.int64)
    num_children = num_indices - last_item - 1
    children_start_offset = num_children.cumsum(dim=0) - num_children
    total_children = int(children_start_offset[-1] + num_children[-1])
    if total_children == 0: return torch.empty((0, level + 1), dtype=torch.int64)  # in case all unk_combs ends with the last index
    if total_children * (level + 3) * 8 > MEM_BUDGET:
        raise torch.OutOfMemoryError(
            f"Memory budget exceeded: {total_children * (level + 3) * 8/1024**3:.1f}GB > {MEM_BUDGET/1024**3:.1f}GB. "
            "Consider reducing the number of prefixes or increasing the memory budget."
        )
    prefix_idx = torch.arange(unk_combs.shape[0]).repeat_interleave(num_children)
    arange_within = torch.arange(total_children) - children_start_offset[prefix_idx]
    new_combs = torch.empty((total_children, level + 1), dtype=torch.int64)
    new_combs[:, :level] = unk_combs[prefix_idx]
    new_combs[:, level] = last_item[prefix_idx] + arange_within + 1
    if safe_bitmasks.shape[0] == 0: return new_combs  # no safe_bitmasks, no pruning

    bitmask_length = (num_indices + 63) // 64
    if len(unk_combs) < len(safe_bitmasks):
        reference_bitmasks_chunks = combs2bitmasks(unk_combs.to(DEVICE), bitmask_length).split(chunk_size*4)
        target_s_cnt = level + 1
    else:
        reference_bitmasks_chunks = safe_bitmasks.to(DEVICE).split(chunk_size*4)
        target_s_cnt = 0
    pruned_new_combs_list = []

    pbar = tqdm(total=new_combs.shape[0], desc="Pruning next level combs", disable=not demo, bar_format=TQDM_BAR_FORMAT)
    for i in range(0, new_combs.shape[0], chunk_size):
        new_bitmasks_chunk = combs2bitmasks(new_combs[i:i + chunk_size].to(DEVICE), bitmask_length)  # only the chunk is on GPU
        superset_cnt = torch.zeros(new_bitmasks_chunk.shape[0], dtype=torch.uint8, device=DEVICE)

        for reference_bitmasks_chunk in reference_bitmasks_chunks:
            superset_cnt += superset_sum(new_bitmasks_chunk, reference_bitmasks_chunk)

        # == 0: Filter out new_combs that are a subset of any safe_combs
        # == level+1: Only keep those new_combs who are the superset of level + 1 sets from the previous level
        valid_new_bm_chunk = new_bitmasks_chunk[superset_cnt == target_s_cnt]
        pruned_new_combs_list.append(bitmasks2combs(valid_new_bm_chunk, level + 1).cpu())
        pbar.update(new_bitmasks_chunk.shape[0])
    pbar.close()

    return torch.cat(pruned_new_combs_list, dim=0)  # on CPU

def prepare4pruning(unk_combs: torch.Tensor, num_indices: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    num_unk, level = unk_combs.shape
    last_item = unk_combs[:, -1] if level > 0 else torch.tensor([-1], dtype=torch.int64)
    num_children = num_indices - last_item - 1
    children_start_offset = num_children.cumsum(dim=0) - num_children
    total_children = int(children_start_offset[-1] + num_children[-1])
    if total_children * (level + 3) * 8 > MEM_BUDGET:
        raise torch.OutOfMemoryError(
            f"Memory budget exceeded: {total_children * (level + 3) * 8/1024**3:.1f}GB > {MEM_BUDGET/1024**3:.1f}GB. "
            "Consider reducing the number of prefixes or increasing the memory budget."
        )
    prefix_idx = torch.arange(num_unk).repeat_interleave(num_children)
    arange_within = torch.arange(total_children) - children_start_offset[prefix_idx]
    new_combs = torch.empty((total_children, level + 1), dtype=torch.int64)
    new_combs[:, :level] = unk_combs[prefix_idx]
    new_combs[:, level] = last_item[prefix_idx] + arange_within + 1

    bitmask_length = (num_indices + 63) // 64
    reference_bitmasks = combs2bitmasks(unk_combs.to(DEVICE), bitmask_length)
    target_s_cnt = level + 1
    if num_unk > math.comb(num_indices, level) * 0.5:  # prune by safe
        unk_bitmasks_chunks = reference_bitmasks.split(chunk_size*4)
        safe_bitmasks_list = []
        all_combs_this_level = torch.combinations(torch.arange(num_indices), level)
        for i in range(0, all_combs_this_level.shape[0], chunk_size):
            all_bitmasks_chunk = combs2bitmasks(all_combs_this_level[i:i + chunk_size].to(DEVICE), bitmask_length)
            superset_cnt = torch.zeros(all_bitmasks_chunk.shape[0], dtype=torch.uint8, device=DEVICE)

            for unk_bm_chunk in unk_bitmasks_chunks:
                superset_cnt += superset_sum(all_bitmasks_chunk, unk_bm_chunk)

            safe_bitmasks_list.append(all_bitmasks_chunk[superset_cnt == 0])
        reference_bitmasks = torch.cat(safe_bitmasks_list, dim=0)
        target_s_cnt = 0
    return new_combs, reference_bitmasks.cpu(), target_s_cnt

def prune(
    new_combs: torch.Tensor,
    reference_bitmasks: torch.Tensor,
    target_s_cnt: int,
    level: int
) -> torch.Tensor:
    bitmask_length = reference_bitmasks.shape[1]
    reference_bitmasks_chunks = reference_bitmasks.to(DEVICE).split(chunk_size*4)
    pruned_new_combs_list = []

    for i in range(0, new_combs.shape[0], chunk_size):
        new_bitmasks_chunk = combs2bitmasks(new_combs[i:i + chunk_size].to(DEVICE), bitmask_length)  # only the chunk is on GPU
        superset_cnt = torch.zeros(new_bitmasks_chunk.shape[0], dtype=torch.uint8, device=DEVICE)

        for reference_bitmasks_chunk in reference_bitmasks_chunks:
            superset_cnt += superset_sum(new_bitmasks_chunk, reference_bitmasks_chunk)

        # == 0: Filter out new_combs that are a subset of any safe_combs
        # == level+1: Only keep those new_combs who are the superset of level + 1 sets from the previous level
        valid_new_bm_chunk = new_bitmasks_chunk[superset_cnt == target_s_cnt]
        pruned_new_combs_list.append(bitmasks2combs(valid_new_bm_chunk, level + 1).cpu())

    return torch.cat(pruned_new_combs_list, dim=0)  # on CPU

def __estimate_SaP_speed_on_level(
    unk_combs: torch.Tensor,
    level: int,  # the size of each combination in unk_combs is `level`
    num_indices: int,  # Number of pixels in the image (e.g., 1024 for 32x32 images)
) -> torch.Tensor:
    last_item = unk_combs[:, -1] if level > 0 else torch.tensor([-1], dtype=torch.int64)
    num_children = num_indices - last_item - 1
    children_start_offset = num_children.cumsum(dim=0) - num_children
    total_children = int(children_start_offset[-1] + num_children[-1])

    split_start = time.time()
    new_combs, reference_bitmasks, target_s_cnt = prepare4pruning(unk_combs, num_indices)
    prune_start = time.time()
    prune(new_combs, reference_bitmasks, target_s_cnt, level)
    end = time.time()
    splitting_speed = total_children / (prune_start - split_start)
    pruning_speed = reference_bitmasks.shape[0] * total_children / (end - prune_start)
    return splitting_speed, pruning_speed

def batched_randsample(upper_bound: int, sample_size: int, batch_size: int) -> torch.Tensor:
    rand_combs = torch.rand(batch_size, upper_bound, device=DEVICE).topk(sample_size, dim=1, sorted=False).indices
    return rand_combs.sort().values.cpu()

def estimate_SaP_speed(epsilon:int, num_indices:int, show_progress=False, size_a: int=13740, size_b: int=53740):
    splitting_speed: dict[int, float] = {}
    pruning_speed: dict[int, float] = {}
    for level in range(2, epsilon):
        if show_progress: print(f" {level}", end="" if level != epsilon - 1 else "\n", flush=True)
        unk_combs = batched_randsample(num_indices, level, size_a).unique(dim=0)
        splitting_speed_a, pruning_speed_a = __estimate_SaP_speed_on_level(unk_combs, level=level, num_indices=num_indices)
        unk_combs = batched_randsample(num_indices, level, size_b).unique(dim=0)
        splitting_speed_b, pruning_speed_b = __estimate_SaP_speed_on_level(unk_combs, level=level, num_indices=num_indices)
        splitting_speed[level], pruning_speed[level] = max(splitting_speed_a, splitting_speed_b), max(pruning_speed_a, pruning_speed_b)
    return splitting_speed, pruning_speed

def estimate_SaP_time(num_unk: int, level: int, num_indices: int, splitting_speed: float, pruning_speed: float) -> float:
    estimated_num_children = num_unk * (num_indices - level) / (level + 1)
    estimated_splitting_time = estimated_num_children / splitting_speed
    if num_unk < math.comb(num_indices, level) * 0.5:
        return estimated_splitting_time + (num_unk * estimated_num_children) / pruning_speed
    else:  # prune by safe
        safe_combs_gen_time = (math.comb(num_indices, level) * num_unk) / pruning_speed
        return estimated_splitting_time + safe_combs_gen_time + ((math.comb(num_indices, level) - num_unk) * estimated_num_children) / pruning_speed