import warnings
import torch
from typing import List
from tqdm import tqdm
import bimm
from utils import DEVICE, tqdm_bar_format, num_features

chunk_size = 2000  # Adjust based on GPU memory availability
num_ints = 13
one_hot_feature_mask = torch.zeros((num_features, num_ints), dtype=torch.uint64, device=DEVICE)
"""
Precompute a one-hot bitmask for each feature as a tensor of shape (num_features, num_ints).
Each row has one nonzero element at the block corresponding to the feature.
"""
for idx in range(num_features):
    one_hot_feature_mask[idx, idx // 64] = (torch.tensor(1, dtype=torch.int64, device=DEVICE) << idx % 64)
one_hot_feature_mask = one_hot_feature_mask.to(torch.int64)

def comb2bitmask64(comb: List[int]) -> torch.Tensor:
    mask = torch.zeros(num_ints, dtype=torch.int64)
    for idx in comb:
        mask[idx // 64] |= (torch.tensor(1, dtype=torch.int64) << idx % 64)
    return mask

def split_and_prune(unk_combs: List[List[int]], safe_combs: List[List[int]], level: int, demo: bool=False) -> torch.Tensor:
    bitmask_list = []
    for comb in tqdm(unk_combs, desc="Bitmasking combs", bar_format=tqdm_bar_format, leave=False, disable=not demo):
        bitmask_list.append(comb2bitmask64(comb))
    unk_bitmasks = torch.stack(bitmask_list, dim=0).to(DEVICE)

    new_bm_list = []
    for i in range(num_features):
        new_bm_i = unk_bitmasks.clone()
        # filter out those combs that already have the i-th feature
        prune = new_bm_i.bitwise_and(one_hot_feature_mask[i]).any(dim=1)
        if prune.all() : continue
        new_bm_i = new_bm_i[~prune].bitwise_or(one_hot_feature_mask[i])
        new_bm_list.append(new_bm_i)

    try:
        new_bitmasks = torch.cat(new_bm_list, dim=0).unique(dim=0)
    except torch.OutOfMemoryError:
        warnings.warn("CUDA out of memory when concatenating new_bitmasks. Attempting to chunk process it.", UserWarning)
        new_bitmasks = torch.empty((0, num_ints), dtype=torch.int64, device=DEVICE)
        for i in range(0, len(new_bm_list), 100):
            new_chunk = new_bm_list[i:i + 100]
            new_bitmasks = torch.cat((new_bitmasks, torch.cat(new_chunk, dim=0)), dim=0)
            new_bitmasks = new_bitmasks.unique(dim=0)
            torch.cuda.empty_cache()
        new_bitmasks = new_bitmasks.unique(dim=0)
    del new_bm_list, bitmask_list
    torch.cuda.empty_cache()

    if len(unk_combs) < len(safe_combs):
        new_bitmasks_pruned_list = prune_by_unk(new_bitmasks, unk_bitmasks, level, demo)
    else:
        del unk_bitmasks
        torch.cuda.empty_cache()
        # Convert safe_combs to a tensor and precompute sums
        if safe_combs:
            safe_bitmasks = torch.stack([comb2bitmask64(s) for s in safe_combs], dim=0).to(DEVICE)
        else:
            safe_bitmasks = torch.empty((0, num_ints), dtype=torch.int64, device=DEVICE)

        new_bitmasks_pruned_list = prune_by_safe(new_bitmasks, safe_bitmasks, demo)

        del safe_bitmasks


    if new_bitmasks_pruned_list:
        combs_to_check_tensor = torch.cat(new_bitmasks_pruned_list, dim=0)
    else:
        combs_to_check_tensor = torch.empty((0, num_ints), dtype=torch.int64, device=DEVICE)

    del new_bitmasks_pruned_list
    torch.cuda.empty_cache()

    return bimm.bitmask2comb(combs_to_check_tensor, level + 1)

def prune_by_safe(new_bitmasks: torch.Tensor, safe_bitmasks: torch.Tensor, demo: bool=False) -> List[torch.Tensor]:
    new_bitmasks_pruned_list = []

    with tqdm(total=new_bitmasks.shape[0], desc="Pruning next level combs", disable=not demo,
                    bar_format=tqdm_bar_format) as pbar:
        for i in range(0, new_bitmasks.shape[0], chunk_size):
            new_bitmasks_ichunk = new_bitmasks[i:i + chunk_size]
            superset_cnt = torch.zeros(new_bitmasks_ichunk.shape[0], dtype=torch.uint8, device=DEVICE)

            for j in range(0, safe_bitmasks.shape[0], chunk_size*8):
                safe_bitmasks_jchunk = safe_bitmasks[j:j + chunk_size*8]
                superset_cnt += bimm.superset_sum(new_bitmasks_ichunk, safe_bitmasks_jchunk)

            # Filter out new_combs that are a subset of any safe_combs
            valid_new_combs = new_bitmasks_ichunk[superset_cnt == 0]
            if valid_new_combs.shape[0] > 0:
                new_bitmasks_pruned_list.append(valid_new_combs)
            pbar.update(new_bitmasks_ichunk.shape[0])

    return new_bitmasks_pruned_list


def prune_by_unk(new_bitmasks: torch.Tensor, unk_bitmasks: torch.Tensor, level: int, demo: bool=False) -> List[torch.Tensor]:
    new_bitmasks_pruned_list = []

    with tqdm(total=new_bitmasks.shape[0], desc="Pruning next level combs", disable=not demo,
                bar_format=tqdm_bar_format) as pbar:
        for i in range(0, new_bitmasks.shape[0], chunk_size):
            new_bitmasks_ichunk = new_bitmasks[i:i + chunk_size]
            superset_cnt = torch.zeros(new_bitmasks_ichunk.shape[0], dtype=torch.uint8, device=DEVICE)

            for j in range(0, unk_bitmasks.shape[0], chunk_size*8):
                unk_bitmasks_jchunk = unk_bitmasks[j:j + chunk_size*8]
                superset_cnt += bimm.superset_sum(new_bitmasks_ichunk, unk_bitmasks_jchunk)

            # Only keep those new_combs whose superset_cnt == level + 1
            valid_new_combs = new_bitmasks_ichunk[superset_cnt == level + 1]
            if valid_new_combs.shape[0] > 0:
                new_bitmasks_pruned_list.append(valid_new_combs)
            pbar.update(new_bitmasks_ichunk.shape[0])

    return new_bitmasks_pruned_list
