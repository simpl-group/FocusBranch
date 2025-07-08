from time import perf_counter
from models import get_network_and_dataset_utils
from data.prepare import build_loader
from utils import *
from AbstractInterpreter import L0BoundedInput, SequentialAT
from tqdm import tqdm
import torch
import signal
import argparse
from json import dump
from verifier import Verifier
from pprint import pprint

if __name__ == "__main__":
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
    parser.add_argument("--sample-all", action="store_true", help="Sample all k from eps to max_k rep_each_k times and return the mean estimated verification time for each k as a dict, which will be dumped to output file.")
    args = parser.parse_args()
    if args.demo: args.test_batches = 1

    net, dataset_utils = get_network_and_dataset_utils(args.net, device=DEVICE)
    if args.data_source == 'pt':
        images, labels = dataset_utils.load_xy_from_pt()
    elif args.data_source == 'csv':
        images, labels = dataset_utils.load_xy_from_csv()
    data_loader = build_loader(
        images,
        labels,
        batch_size=1,
        target_label=args.target_label,
        num_batches=args.start_idx+args.test_batches
    )
    x_ = torch.zeros(1, *dataset_utils.image_shape).to(DEVICE)
    seqAT = SequentialAT.from_sequential(net, x_)
    bounded_x0 = L0BoundedInput(
        x=x_.reshape(1, -1, 1),
        eps=args.epsilon,
        channel_size=dataset_utils.channel_size,
        num_pixels=dataset_utils.num_pixels
    )
    partial_indices = None
    verifier = Verifier(
        seqAT,
        bounded_x0,
        batch_size=args.batch_size,
        targeted=args.target_label is not None,
        max_k=args.max_k,
        rep_each_k=args.rep_each_k,
        rep_each_level=args.rep_each_level,
        timeout=args.timeout,
        demo=args.demo
    )

    total_time = 0.
    num_verified = 0
    record: list[tuple[bool, float]] = []
    net.cpu()
    pbar = tqdm(total=args.test_batches, bar_format=TQDM_BAR_FORMAT, disable=args.demo)
    i = 0
    for images, pred_labels in data_loader:
        if i < args.start_idx:
            i += 1
            continue
        assert net(images).argmax(dim=1).eq(pred_labels).all()
        x0 = images.to(DEVICE).flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
        verifier.update_x0(x0, pred_labels, target_label=args.target_label)

        if args.sample_all:
            t0 = perf_counter()
            pprint(verifier.sample_all_k())
            print(f"Time {perf_counter() - t0:.2f}s")
            exit(0)

        if isinstance(args.partial, int):
            partial_indices = torch.randperm(verifier.num_pixels)[:args.partial]
        success = False
        elapsed_time = 0.
        try:
            if args.probe: verifier.probe(partial_indices)
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
            bounded_x0.clear_cache()
            torch.cuda.empty_cache()

        if args.demo: break

        pbar.update(1)
        mean_time = total_time / num_verified if num_verified > 0 else 0
        pbar.set_postfix({"Verified": f"{num_verified}/{pbar.n}|{100*num_verified/pbar.n:.2f}%", "Avg":f"{mean_time:.2f}s/img"})

        record.append((success, elapsed_time))
        with open(args.output, "w") as f:
            dump(record, f, indent=2)
        if pbar.n == args.test_batches: break
    pbar.close()
