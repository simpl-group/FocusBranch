from coverd_finite_geometries import Geometry
from . import launch_covering_generator
from multiprocessing.connection import Listener
from scipy.stats import norm
import random
import math
import time


def estimate_generator_speed(geometry=Geometry(7,5,5), num_workers=4, num_pixels=784):
    num_of_blocks = math.floor(geometry.number_of_blocks() / num_workers)
    worker_id = 0
    if worker_id < geometry.number_of_blocks() % num_workers:
        num_of_blocks += 1
    chosen_indices = random.sample(range(geometry.number_of_points()), num_pixels)
    num_sizes = {}
    start_time = time.time()
    with Listener(('localhost', 0)) as blocks_listener:
            launch_covering_generator(
                q=geometry.q, m=geometry.m, t=geometry.t,
                port=blocks_listener.address[1],
                chosen_indices=chosen_indices,
                num_workers=num_workers,
                worker_id=worker_id
            )
            with blocks_listener.accept() as blocks_conn:
                for _ in range(num_of_blocks):
                    if 'iter_start_time' not in locals():
                        iter_start_time = time.time()
                        launch_time1 = iter_start_time - start_time
                    superset = blocks_conn.recv()  # assert superset is sorted and of type list[int]
                    setsize = len(superset)
                    num_sizes[setsize] = num_sizes.get(setsize, 0) + 1
                assert blocks_conn.recv() == 'done'
                iter_duration = time.time() - iter_start_time
                blocks_conn.send('done')
    skipping_iter_speed = num_of_blocks / iter_duration

    num_sizes = {}
    start_time2 = time.time()
    with Listener(('localhost', 0)) as blocks_listener:
            launch_covering_generator(
                q=geometry.q, m=geometry.m, t=geometry.t,
                port=blocks_listener.address[1],
                chosen_indices=chosen_indices,
                num_workers=1,
                worker_id=0
            )
            with blocks_listener.accept() as blocks_conn:
                for _ in range(geometry.number_of_blocks()):
                    if 'iter_start_time2' not in locals():
                        iter_start_time2 = time.time()
                        launch_time2 = iter_start_time2 - start_time2
                    superset = blocks_conn.recv()
                    setsize = len(superset)
                    num_sizes[setsize] = num_sizes.get(setsize, 0) + 1
                assert blocks_conn.recv() == 'done'
                iter_duration2 = time.time() - iter_start_time2
                blocks_conn.send('done')
    iter_speed = geometry.number_of_blocks() / iter_duration2
    return (launch_time1+launch_time2)*0.5, skipping_iter_speed, iter_speed

def estimate_extra_time(geometry: Geometry, num_workers, num_pixels, greater_batch, skipping_iter_speed, iter_speed):
    mean = geometry.block_size() * num_pixels / geometry.number_of_points()
    std  = math.sqrt(mean * (1 - mean + (geometry.block_size() - 1) * (num_pixels - 1) / (geometry.number_of_points() - 1)))
    num_of_blocks = math.floor(geometry.number_of_blocks() / num_workers)
    worker_id=0
    if worker_id < geometry.number_of_blocks() % num_workers:
        num_of_blocks += 1
    expected_number_of_blocks_of_mean_size = (norm.cdf(math.floor(mean)+1, mean, std) - norm.cdf(math.floor(mean), mean, std)).item() * num_of_blocks
    if greater_batch > expected_number_of_blocks_of_mean_size:
        return num_of_blocks / skipping_iter_speed
    prefilling_number_of_blocks = num_of_blocks * (greater_batch / expected_number_of_blocks_of_mean_size)
    rest_number_of_blocks = num_of_blocks - prefilling_number_of_blocks
    return prefilling_number_of_blocks / skipping_iter_speed + rest_number_of_blocks / iter_speed