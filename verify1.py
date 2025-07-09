if __name__ != "__main__": exit(0)

from utils import get2, add_shared_arguments
from utils.timeout import timeout_handler, TimeoutException
from utils.constant import TQDM_BAR_FORMAT
from time import perf_counter
import signal
import argparse
from json import dump
from pprint import pprint
from tqdm import tqdm
import torch

signal.signal(signal.SIGALRM, timeout_handler)
parser = argparse.ArgumentParser(
    description="Sample-wise (local) L0 robustness verification of neural networks"
)
add_shared_arguments(parser)
parser.add_argument("--max-k", type=int, default=200, help="Maximum k to sample from.")
parser.add_argument("-rk", "--rep-each-k", type=int, default=50, help="Number of repetitions for each k.")
parser.add_argument("-rl", "--rep-each-level", type=int, default=1000, help="Maximum number of repetitions for each level.")
parser.add_argument("--probe", action="store_true", help="Probe each level to estimate the time for verification.")
parser.add_argument("--partial", type=int, default=None, help="Maximum k to sample from.")
parser.add_argument("--sample-all", action="store_true", help="Sample all k from eps to max_k rep_each_k times and return the mean estimated verification time for each k")
args = parser.parse_args()

verifier, data_loader = get2(args)
verifier.max_k = args.max_k
verifier.rep_each_k = args.rep_each_k
verifier.rep_each_level = args.rep_each_level
partial_indices = None

total_time = 0.
num_verified = 0
record: list[tuple[bool, float]] = []
pbar = tqdm(total=args.test_batches, bar_format=TQDM_BAR_FORMAT, disable=args.demo)
for images, true_labels in data_loader:
    verifier.adapt_target_images(images, true_labels, target_label=args.target_label)

    if args.sample_all:
        t0 = perf_counter()
        pprint(verifier.sample_all_k())
        print(f"Time {perf_counter() - t0:.2f}s")
        exit(0)

    if isinstance(args.partial, int):
        partial_indices = torch.randperm(verifier.num_pixels)[:args.partial]
    if args.probe: verifier.probe(partial_indices)
    success = False
    elapsed_time = 0.
    try:
        signal.alarm(args.timeout)
        t0 = perf_counter()
        if verifier.verify1(partial_indices):
            elapsed_time = perf_counter() - t0
            total_time += elapsed_time
            success = True
            num_verified += 1
        signal.alarm(0)
    except TimeoutException:
        if args.demo: print("Timeout")
    except torch.OutOfMemoryError as e:
        if args.demo: raise(e)
    finally:
        verifier.bounded_x0.clear_cache()
        torch.cuda.empty_cache()

    if args.demo: break

    pbar.update(1)
    mean_time = total_time / num_verified if num_verified > 0 else 0
    pbar.set_postfix({"Verified": f"{num_verified}/{pbar.n}|{100*num_verified/pbar.n:.2f}%", "Avg":f"{mean_time:.2f}s/img"})

    record.append((success, elapsed_time))
    with open(args.output, "w") as f:  #TODO: use logging
        dump(record, f, indent=2)
pbar.close()
