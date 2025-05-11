import torch
from tqdm import tqdm
from bimm import bitmasks2combs, superset_sum
from utils import DEVICE, TQDM_BAR_FORMAT
from data import DATASETS, DatasetName, DatasetUtils

chunk_size = 4000  # Adjust based on GPU memory availability

def canonical_split(prefix: torch.Tensor, N: int) -> torch.Tensor:
    # prefix are naturally sorted in each row
    # Purely on CPU, in case of CUDA out of memory
    num_prefix, k = prefix.shape  # level != 0
    last_item = prefix[:, -1] if k > 0 else torch.tensor([-1], dtype=torch.int64, device=prefix.device)
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
    unk_combs_tensor = torch.tensor(unk_combs, dtype=torch.int64)
    new_combs = canonical_split(unk_combs_tensor, dataset_utils.num_pixels)
    if not safe_combs: return new_combs.to(DEVICE)  # no safe_combs, no pruning

    if len(unk_combs) < len(safe_combs):
        reference_bitmasks = dataset_utils.combs2bitmasks(unk_combs_tensor.to(DEVICE))
        target_s_cnt = level + 1
    else:
        if watchout:
            reference_bitmasks = torch.stack([dataset_utils.comb2bitmask(s) for s in safe_combs], dim=0).to(DEVICE)
        else:
            reference_bitmasks = dataset_utils.combs2bitmasks(torch.tensor(safe_combs, dtype=torch.int64).to(DEVICE))
        target_s_cnt = 0
    del unk_combs_tensor
    new_bitmasks_pruned_list = []

    pbar = tqdm(total=new_combs.shape[0], desc="Pruning next level combs", disable=not demo, bar_format=TQDM_BAR_FORMAT)
    while new_combs.shape[0] > 0:
        new_bitmasks_chunk = dataset_utils.combs2bitmasks(new_combs[:chunk_size].to(DEVICE))  # only the chunk is on GPU
        superset_cnt = torch.zeros(new_bitmasks_chunk.shape[0], dtype=torch.uint8, device=DEVICE)

        for j in range(0, reference_bitmasks.shape[0], chunk_size*4):
            superset_cnt += superset_sum(new_bitmasks_chunk, reference_bitmasks[j:j + chunk_size*4])

        # == 0: Filter out new_combs that are a subset of any safe_combs
        # == level+1: Only keep those new_combs who are the superset of level + 1 sets from the previous level
        valid_new_bm_chunk = new_bitmasks_chunk[superset_cnt == target_s_cnt]
        new_bitmasks_pruned_list.append(valid_new_bm_chunk.cpu())
        pbar.update(new_bitmasks_chunk.shape[0])
        new_combs = new_combs[chunk_size:]
    pbar.close()

    pruned_new_bitmasks = torch.cat(new_bitmasks_pruned_list, dim=0).to(DEVICE)
    del reference_bitmasks, new_bitmasks_chunk, valid_new_bm_chunk, new_bitmasks_pruned_list, superset_cnt
    torch.cuda.empty_cache()

    return bitmasks2combs(pruned_new_bitmasks, level + 1)
