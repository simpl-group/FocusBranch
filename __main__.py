# Reference: https://github.com/YuvShap/CoVerD/blob/main/tf_verify/coverd.py
import sys
import os
import argparse
from l0_robustness_analyzer import LZeroRobustnessAnalyzer
from multiprocessing.connection import Listener
from subprocess import Popen
import json
import itertools
import time
import random
import pandas as pd
from utils.shared import add_shared_arguments
from data.prepare import build_loader
from models import get_network_and_dataset_utils
from solver_check import launch_gurobi_worker
from verifier import __file__ as verifier_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample-wise (local) L0 robustness verification for neural networks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_shared_arguments(parser)
    parser.add_argument("--gpu_num", type=int, default=2, help="the number of GPUs used in the analysis")
    parser.add_argument("--milp_num", type=int, default=1, help="the number of MILP instances used in the analysis")
    parser.add_argument('--base_seed', type=int, default=42, help='the base seed')
    parser.add_argument("--detail", action="store_true", help="Output detailed superset statistics for each image in a separate excel file.")
    parser.add_argument("--plan_only", action="store_true", help="Only run the planning stage without actual verification.")
    args = parser.parse_args()
    args.output = args.output.split(".")[0] + f"{args.net}_t{args.epsilon}b{args.batch_size}g{args.gpu_num}s{args.start_idx}n{args.test_batches}"
    ilogname = args.output.rsplit("s", 1)[0]
    args.output = args.output + ".json"
    args.rep_each_k = max(args.rep_each_k, args.batch_size)

    address = ('localhost', 0)
    with Listener(address) as gpu_listener, Listener(address) as cpu_listener:
        for worker_id in range(args.gpu_num):
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = str(worker_id % args.gpu_num)
            Popen([
                sys.executable, verifier_file,
                "--worker_id", str(worker_id),
                "--num-workers", str(args.gpu_num),
                "--port", str(gpu_listener.address[1]),
                "--net",  args.net,
                "--batch-size", str(args.batch_size),
                "--epsilon", str(args.epsilon),
                "--max-k", str(args.max_k),
                "--rep-each-k", str(args.rep_each_k),
                "--rep-each-level", str(args.rep_each_level)
            ], env=my_env)

        for worker_id in range(args.milp_num):
            launch_gurobi_worker(netname=args.net, port=cpu_listener.address[1])

        print("Waiting for cpu workers")
        cpu_workers = [cpu_listener.accept() for _ in range(args.milp_num)]

        print("Waiting for gpu workers")
        gpu_workers = [gpu_listener.accept() for _ in range(args.gpu_num)]

    net, dataset_utils = get_network_and_dataset_utils(args.net)
    images, labels = dataset_utils.load_xy_from_csv(normalize=False)  # TODO: add option to normalize
    data_loader = build_loader(
        images,
        labels,
        batch_size=args.grain_size,
        target_label=args.target_label,
        start_idx=args.start_idx,
        num_batches=args.test_batches
    )
    analyzer = LZeroRobustnessAnalyzer(
        gpu_workers=gpu_workers,
        cpu_workers=cpu_workers,
        t=args.epsilon,
        timeout=args.timeout,
        max_k=args.max_k,
        num_pixels=dataset_utils.num_pixels,
        plan_only=args.plan_only
    )

    image_results_by_image_index = {}
    correctly_classified_images = 0
    verified_images = 0
    unsafe_images = 0
    cum_time = 0.
    for _i, (image, true_label) in enumerate(data_loader):
        i = _i + args.start_idx
        numpy_img = image.reshape(-1).numpy()  # shape (1, C, H, W) -> (C*H*W)
        label = true_label.item()
        start = time.time()
        is_correctly_classified = net(image).argmax(dim=1).eq(true_label).all().item()
        image_results = {}
        image_results['is_correctly_classified'] = is_correctly_classified
        image_results['true_label'] = label
        if is_correctly_classified:
            correctly_classified_images += 1
            print(f'Starting to analyze image {i}, label {label}')

            random.seed(args.base_seed + i)
            analyzer.update_image(numpy_img, label)
            print('*******************************************************')
            analysis_summary = analyzer.analyze()
            exec_details = analysis_summary.pop('exec_details', None)
            if exec_details is not None and args.detail:
                pd.DataFrame(exec_details).to_excel(ilogname + f"i{i}.xlsx", engine='openpyxl', index_label="Row")
            image_results['analysis_summary'] = analysis_summary

            if analysis_summary['verified']:
                verified_images += 1
            elif not analysis_summary['timed_out']:
                unsafe_images += 1

            end = time.time()
            cum_time += end - start  # only count samples where we did try to certify
        else:
            print("img", i, "not considered, incorrectly classified")
            end = time.time()

        image_results['running_time'] = end - start
        image_results_by_image_index[i] = image_results

        print(
            f"progress: {1 + i - args.start_idx}/{args.test_batches}, "
            f"correct:  {correctly_classified_images}/{1 + i - args.start_idx}, "
            f"verified: {verified_images}/{correctly_classified_images}, "
            f"unsafe: {unsafe_images}/{correctly_classified_images}, ",
            f"time: {end - start:.3f}; {0 if cum_time == 0 else cum_time / correctly_classified_images:.3f}; {cum_time:.3f}"
        )

    for worker in itertools.chain(gpu_workers, cpu_workers):
        worker.send('terminate')

    results = {
        'netname': args.net,
        't': args.epsilon,
        'rep_num': args.rep_each_k,
        'timeout': args.timeout,
        'start_idx': args.start_idx,
        'test_num': args.test_batches,
        'cumulative_time': cum_time,
        'images_results_by_index': image_results_by_image_index
    }

    print('analysis precision ', verified_images, '/ ', correctly_classified_images)
    with open(args.output, 'w') as fp:
        json.dump(results, fp, indent=4)
    print(f'Results saved to {args.output}')
