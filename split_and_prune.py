import torch
from tqdm import tqdm
import bimm
from utils import DEVICE, tqdm_bar_format
from data import DATASETS, DatasetName

chunk_size = 2000  # Adjust based on GPU memory availability

def comb2bitmask64(comb: list[int], dataset: DatasetName=DatasetName.MNIST) -> torch.Tensor:
    mask = torch.zeros(DATASETS[dataset].bitmask_length, dtype=torch.int64)
    for idx in comb:
        mask[idx // 64] |= (torch.tensor(1, dtype=torch.int64) << idx % 64)
    return mask

def split_and_prune(
    unk_combs: list[list[int]],
    safe_combs: list[list[int]],
    level: int,
    demo: bool=False,
    watchout: bool=False,
    dataset: DatasetName=DatasetName.MNIST
) -> torch.Tensor:
    unk_combs_tensor = torch.tensor(unk_combs, dtype=torch.int64, device=DEVICE)
    unk_bitmasks = bimm.comb2bitmask(unk_combs_tensor, DATASETS[dataset].bitmask_length)

    new_bm_list = []
    for i in range(DATASETS[dataset].num_pixels):
        new_bm_i = unk_bitmasks.clone()
        # filter out those combs that already have the i-th feature
        prune = new_bm_i.bitwise_and(DATASETS[dataset].one_hot_bitmasks_cuda[i]).any(dim=1)
        if prune.all() : continue
        new_bm_i = new_bm_i[~prune].bitwise_or(DATASETS[dataset].one_hot_bitmasks_cuda[i])
        new_bm_list.append(new_bm_i)

    new_bitmasks = DATASETS[dataset].get_empty_bitmask(DEVICE)
    while len(new_bm_list) > 0:
        new_chunk = torch.cat(new_bm_list[:100], dim=0)
        new_bitmasks = torch.cat((new_bitmasks, new_chunk), dim=0)
        new_bm_list[:100] = []
        del new_chunk
        torch.cuda.empty_cache()
        new_bitmasks = new_bitmasks.unique(dim=0)
    torch.cuda.empty_cache()

    if len(unk_combs) < len(safe_combs):
        new_bitmasks_pruned_list = prune_by_unk(new_bitmasks, unk_bitmasks, level, demo)
    else:
        del unk_bitmasks
        torch.cuda.empty_cache()
        if safe_combs:
            if watchout:
                safe_bitmasks = torch.stack([comb2bitmask64(s) for s in safe_combs], dim=0).to(DEVICE)
            else:
                safe_bitmasks = bimm.comb2bitmask(
                    torch.tensor(safe_combs, dtype=torch.int64, device=DEVICE),
                    DATASETS[dataset].bitmask_length)
        else:
            safe_bitmasks = DATASETS[dataset].get_empty_bitmask(DEVICE)

        new_bitmasks_pruned_list = prune_by_safe(new_bitmasks, safe_bitmasks, demo)

        del safe_bitmasks


    if new_bitmasks_pruned_list:
        combs_to_check_tensor = torch.cat(new_bitmasks_pruned_list, dim=0)
    else:
        combs_to_check_tensor = DATASETS[dataset].get_empty_bitmask(DEVICE)

    del new_bitmasks_pruned_list
    torch.cuda.empty_cache()

    return bimm.bitmask2comb(combs_to_check_tensor, level + 1)

def prune_by_safe(new_bitmasks: torch.Tensor, safe_bitmasks: torch.Tensor, demo: bool=False) -> list[torch.Tensor]:
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


def prune_by_unk(new_bitmasks: torch.Tensor, unk_bitmasks: torch.Tensor, level: int, demo: bool=False) -> list[torch.Tensor]:
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
