import torch
from torch.utils import benchmark
from l0bound import selected_top_k_sum_single, selected_top_k_sum_warped

def bench(shp, num_select, k, device="cuda"):
    x = torch.rand(shp, device=device)
    selected_idx = torch.randint(0, shp[-1], (shp[0], num_select), device=device)

    t_top_k_sum_single = benchmark.Timer(
        stmt=f"selected_topk_sum(x, selected_idx, {k})",
        globals={"x": x, "selected_idx": selected_idx, "selected_topk_sum": selected_top_k_sum_single}
    ).timeit(50)

    t_top_k_sum_warped = benchmark.Timer(
        stmt=f"selected_topk_sum(x, selected_idx, {k})",
        globals={"x": x, "selected_idx": selected_idx, "selected_topk_sum": selected_top_k_sum_warped}
    ).timeit(50)

    return t_top_k_sum_single, t_top_k_sum_warped

if __name__ == "__main__":
    from collections import defaultdict
    import logging
    import math
    from pathlib import Path
    result: defaultdict[tuple[int, int], list[int]] = defaultdict(lambda: [])
    # result[(num_select, k)] = []
    logger = logging.getLogger('warp_cmp_logger')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(Path(__file__).parent/'min_nv_to_warp_data.txt')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    for shp01 in [(8, 3136), (5, 8196), (2000, 256), (2000, 128)]:
        for _ in range(5):
            for eps in range(3, 11):
                for num_select in range(eps):
                    k = eps - num_select
                    for pixel in range(eps, 201):
                        shp = (*shp01, pixel)
                        t_top_k_sum_single, t_top_k_sum_warped = bench(shp, num_select, k, device="cuda")
                        if t_top_k_sum_single.mean > t_top_k_sum_warped.mean:
                            # logger.info(f"num_select: {num_select}, k: {k}, pixel: {pixel}")
                            result[(num_select, k)].append(pixel)
                            break

    avg_result: dict[tuple[int, int], int] = {}

    for (num_select, k), pixels in result.items():
        if pixels:
            # logger.info(f"num_select: {num_select}, k: {k}, avg_pixels: {sum(pixels) / len(pixels):.2f}, pixels: {pixels}")
            avg_result[(num_select, k)] = math.ceil(sum(pixels) / len(pixels))
    for (num_select, k), pixel in avg_result.items():
        logger.info(f"    {{{{{num_select}, {k}}}, {pixel}}},")