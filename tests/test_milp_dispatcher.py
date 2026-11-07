"""The MILP dispatcher's deduplication of jobs. Run with `python -m pytest tests`."""
import queue
import pytest
import torch
from milp_dispatcher import MilpDispatcher


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deduplication_drops_repeats_and_jobs_already_sent():
    dispatcher = MilpDispatcher([], num_pixels=16, channel_size=1, eps=2, timeout=10.)
    dispatcher.jobs = queue.Queue()
    dispatcher.innocent, dispatcher.waiting = {(1, 2)}, {(3, 4)}
    dispatcher.jobs.put([3, 4]); dispatcher.jobs.get()  # sent to a MILP worker, its verdict is pending
    for job in ([7, 8], [5, 6], [1, 2], [3, 4], [5, 6]):  # jobs have ascending pixels
        dispatcher.jobs.put(job)
    dispatcher._MilpDispatcher__deduplicate()
    jobs = [dispatcher.jobs.get_nowait() for _ in range(dispatcher.jobs.qsize())]
    assert jobs == [[5, 6], [7, 8]] and all(isinstance(p, int) for job in jobs for p in job)
