from multiprocessing import Process
from .covering_constructor import covering_generator

def launch_covering_generator(q: int, m: int, t: int, chosen_points: list[int], port: int, num_workers=1, worker_id=0) -> Process:
    process = Process(target=covering_generator, args=(q, m, t, chosen_points, port, num_workers, worker_id))
    process.start()
    return process