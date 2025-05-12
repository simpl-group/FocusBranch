import torch
from tqdm import tqdm
from bimm import bitmasks2combs, superset_sum
from utils import DEVICE, TQDM_BAR_FORMAT
from data import DATASETS, DatasetName, DatasetUtils

chunk_size = 4000  # Adjust based on GPU memory availability
MEM_BUDGET = 128 * 1024**3  # 128 Giga Bytes

def split_and_prune(
    unk_combs: torch.Tensor,
    safe_bitmasks: torch.Tensor,
    level: int,
    demo: bool=False,
    dataset_utils: DatasetUtils=DATASETS[DatasetName.MNIST]
) -> torch.Tensor:
    # unk_combs are naturally sorted in each row
    last_item = unk_combs[:, -1] if level > 0 else torch.tensor([-1], dtype=torch.int64)
    num_children = dataset_utils.num_pixels - last_item - 1
    children_start_offset = num_children.cumsum(dim=0) - num_children
    total_children = int(children_start_offset[-1] + num_children[-1])
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

    if len(unk_combs) < len(safe_bitmasks):
        reference_bitmasks = dataset_utils.combs2bitmasks(unk_combs.to(DEVICE))
        target_s_cnt = level + 1
    else:
        reference_bitmasks = safe_bitmasks.to(DEVICE)
        target_s_cnt = 0
    del unk_combs
    pruned_new_combs_list = []

    pbar = tqdm(total=new_combs.shape[0], desc="Pruning next level combs", disable=not demo, bar_format=TQDM_BAR_FORMAT)
    while new_combs.shape[0] > 0:
        new_bitmasks_chunk = dataset_utils.combs2bitmasks(new_combs[:chunk_size].to(DEVICE))  # only the chunk is on GPU
        superset_cnt = torch.zeros(new_bitmasks_chunk.shape[0], dtype=torch.uint8, device=DEVICE)

        for j in range(0, reference_bitmasks.shape[0], chunk_size*4):
            superset_cnt += superset_sum(new_bitmasks_chunk, reference_bitmasks[j:j + chunk_size*4])

        # == 0: Filter out new_combs that are a subset of any safe_combs
        # == level+1: Only keep those new_combs who are the superset of level + 1 sets from the previous level
        valid_new_bm_chunk = new_bitmasks_chunk[superset_cnt == target_s_cnt]
        pruned_new_combs_list.append(bitmasks2combs(valid_new_bm_chunk, level + 1).cpu())
        pbar.update(new_bitmasks_chunk.shape[0])
        new_combs = new_combs[chunk_size:]
    pbar.close()

    return torch.cat(pruned_new_combs_list, dim=0)  # on CPU
