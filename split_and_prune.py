import torch
from tqdm import tqdm
from bimm import bitmasks2combs, superset_sum
from utils import DEVICE, TQDM_BAR_FORMAT
from data import DATASETS, DatasetName, DatasetUtils

chunk_size = 2000  # Adjust based on GPU memory availability

def canonical_split(prefix: torch.Tensor, N: int) -> torch.Tensor:
    # prefix are natrually sorted
    num_prefix, k = prefix.shape  # level != 0
    last_item = prefix[:, -1] if k > 0 else torch.Tensor([-1]).to(prefix.device, dtype=torch.int64)
    num_children = N - last_item - 1
    children_start_offset = num_children.cumsum(dim=0) - num_children
    total_children = int(children_start_offset[-1] + num_children[-1])
    prefix_idx = torch.arange(num_prefix, device=prefix.device).repeat_interleave(num_children)
    arange_within = torch.arange(total_children, device=prefix.device) - children_start_offset[prefix_idx]
    supersets = torch.empty((total_children, k + 1), dtype=torch.int64, device=prefix.device)
    supersets[:, :k] = prefix[prefix_idx]
    supersets[:, k] = last_item[prefix_idx] + arange_within + 1
    return supersets

def split_and_prune(
    unk_combs: list[list[int]],
    safe_combs: list[list[int]],
    level: int,
    demo: bool=False,
    watchout: bool=False,
    dataset_utils: DatasetUtils=DATASETS[DatasetName.MNIST]
) -> torch.Tensor:
    unk_combs_tensor = torch.tensor(unk_combs, dtype=torch.int64, device=DEVICE)
    new_bitmasks = dataset_utils.combs2bitmasks(canonical_split(unk_combs_tensor, dataset_utils.num_pixels))

    if len(unk_combs) < len(safe_combs):
        unk_bitmasks = dataset_utils.combs2bitmasks(unk_combs_tensor)
        del unk_combs_tensor
        new_bitmasks_pruned_list = prune_by_unk(new_bitmasks, unk_bitmasks, level, demo)
        del unk_bitmasks
    else:
        del unk_combs_tensor
        torch.cuda.empty_cache()
        if safe_combs:
            if watchout:
                safe_bitmasks = torch.stack([dataset_utils.comb2bitmask(s) for s in safe_combs], dim=0).to(DEVICE)
            else:
                safe_bitmasks = dataset_utils.combs2bitmasks(torch.tensor(safe_combs, dtype=torch.int64, device=DEVICE))
        else:
            safe_bitmasks = dataset_utils.get_empty_bitmask(DEVICE)

        new_bitmasks_pruned_list = prune_by_safe(new_bitmasks, safe_bitmasks, demo)
        del safe_bitmasks

    if new_bitmasks_pruned_list:
        combs_to_check_tensor = torch.cat(new_bitmasks_pruned_list, dim=0)
    else:
        combs_to_check_tensor = dataset_utils.get_empty_bitmask(DEVICE)

    del new_bitmasks_pruned_list, new_bitmasks
    torch.cuda.empty_cache()

    return bitmasks2combs(combs_to_check_tensor, level + 1)

def prune_by_safe(new_bitmasks: torch.Tensor, safe_bitmasks: torch.Tensor, demo: bool=False) -> list[torch.Tensor]:
    new_bitmasks_pruned_list = []

    with tqdm(total=new_bitmasks.shape[0], desc="Pruning next level combs", disable=not demo,
                    bar_format=TQDM_BAR_FORMAT) as pbar:
        for i in range(0, new_bitmasks.shape[0], chunk_size):
            new_bitmasks_ichunk = new_bitmasks[i:i + chunk_size]
            superset_cnt = torch.zeros(new_bitmasks_ichunk.shape[0], dtype=torch.uint8, device=DEVICE)

            for j in range(0, safe_bitmasks.shape[0], chunk_size*8):
                safe_bitmasks_jchunk = safe_bitmasks[j:j + chunk_size*8]
                superset_cnt += superset_sum(new_bitmasks_ichunk, safe_bitmasks_jchunk)

            # Filter out new_combs that are a subset of any safe_combs
            valid_new_combs = new_bitmasks_ichunk[superset_cnt == 0]
            if valid_new_combs.shape[0] > 0:
                new_bitmasks_pruned_list.append(valid_new_combs)
            pbar.update(new_bitmasks_ichunk.shape[0])

    return new_bitmasks_pruned_list


def prune_by_unk(new_bitmasks: torch.Tensor, unk_bitmasks: torch.Tensor, level: int, demo: bool=False) -> list[torch.Tensor]:
    new_bitmasks_pruned_list = []

    with tqdm(total=new_bitmasks.shape[0], desc="Pruning next level combs", disable=not demo,
                bar_format=TQDM_BAR_FORMAT) as pbar:
        for i in range(0, new_bitmasks.shape[0], chunk_size):
            new_bitmasks_ichunk = new_bitmasks[i:i + chunk_size]
            superset_cnt = torch.zeros(new_bitmasks_ichunk.shape[0], dtype=torch.uint8, device=DEVICE)

            for j in range(0, unk_bitmasks.shape[0], chunk_size*8):
                unk_bitmasks_jchunk = unk_bitmasks[j:j + chunk_size*8]
                superset_cnt += superset_sum(new_bitmasks_ichunk, unk_bitmasks_jchunk)

            # Only keep those new_combs whose superset_cnt == level + 1
            valid_new_combs = new_bitmasks_ichunk[superset_cnt == level + 1]
            if valid_new_combs.shape[0] > 0:
                new_bitmasks_pruned_list.append(valid_new_combs)
            pbar.update(new_bitmasks_ichunk.shape[0])

    return new_bitmasks_pruned_list
