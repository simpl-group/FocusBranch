# Reference: https://github.com/YuvShap/CoVerD/blob/main/tf_verify/coverd.py
import sys
import os
import argparse
import warnings
from verifier import Verifier
from multiprocessing.connection import Listener
from subprocess import Popen
import threading
import json
import itertools
import time
import random
from inspect import signature
from torch.cuda import device_count
import pandas as pd
from utils.shared import add_shared_arguments, get2
from data.db import set_database, store_ours_result, ours_should_skip, ours_config_consistent, sigma0_min_eps, insert_ours
from solver_check import __file__ as solver_file
from gpu_worker import __file__ as gpu_worker_file


def accept_workers(listener: Listener, processes: list[Popen], kind: str, all_processes: list[Popen]) -> list:
    """One connection per worker process, in order of arrival. A worker that exits before every worker has connected
    (e.g. it cannot reach the solver's license server) stops the run with an error instead of a wait that never ends."""
    connections: list = []
    accepting = threading.Thread(target=lambda: connections.extend(listener.accept() for _ in processes), daemon=True)
    accepting.start()
    while accepting.is_alive():
        accepting.join(0.5)
        exited = [p for p in processes if p.poll() is not None]
        if exited and accepting.is_alive():
            for p in all_processes:
                if p.poll() is None: p.kill()
            sys.exit(f"\nA {kind} worker exited with code {exited[0].returncode} before every {kind} worker had connected, see its error above.")
    return connections

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample-wise (local) L0 robustness verification for neural networks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_shared_arguments(parser)
    parser.add_argument("-g", "--num-gpus", "--gpu_num", dest="gpu_num", type=int, default=4, help="The number of GPU workers. Normally, it should not exceed the number of GPUs on your machine.")
    parser.add_argument("-m", "--num-milps", "--milp_num", dest="milp_num", type=int, default=5, help="The number of MILP instances.")
    parser.add_argument("--base-seed", "--base_seed", type=int, default=42, help="The base seed.")
    parser.add_argument("--probe-thickness", "--probe_thickness", type=int, default=50, help="Maximum number of least safe combinations to track.")
    parser.add_argument("--detail", action="store_true", help="Output detailed superset statistics for each image in a separate excel file.")
    parser.add_argument("--plan-only", "--plan_only", action="store_true", help="Only run the planning stage without actual verification.")
    parser.add_argument("--until-timedout", "-ut", action="store_true", help="Continue running until timeout is reached, even if the estimated verification time is larger than twice the timeout.")
    parser.add_argument("--force-reestimate", action="store_true", help="Force a re-estimation of covering generator speed and SaP (concrete branching & pruning) speed even if existing estimates are found in the database.")
    args = parser.parse_args()
    set_database(args.database)
    args.strategy.assert_strategy_applicable(args.epsilon)
    config_params = signature(ours_config_consistent).parameters.keys()
    config_kwargs = {name: getattr(args, name) for name in config_params if hasattr(args, name)}
    consistent, warning_msg = ours_config_consistent(img=args.start_idx+args.test_batches-1, **config_kwargs)
    if not consistent:
        warnings.warn(f"Configuration inconsistent:\n{warning_msg}.")
        if not args.force_store and not args.demo:
            exit(1)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        available_gpus = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip().isdigit()]
    else:
        available_gpus = list(range(device_count()))
    if len(available_gpus) < args.gpu_num:
        warnings.warn(f"Requested {args.gpu_num} GPU workers but only {len(available_gpus)} GPU are available.")

    export_results = args.output != None
    ilogname = f"{args.output if args.output else ''}{args.net}e{args.epsilon}g{args.gpu_num}b{args.supercheck_batch}p{args.pool_size}"
    args.output = f"{args.output if args.output else ''}{args.net}e{args.epsilon}s{args.start_idx}_n{args.test_batches}g{args.gpu_num}m{args.milp_num}cb{args.concretecheck_batch}sb{args.supercheck_batch}ps{args.pool_size}"
    args.output = args.output + ".json"
    args.pool_size = min(max(args.pool_size, args.supercheck_batch), args.supercheck_batch * 10)

    address = ('localhost', 0)
    with Listener(address) as gpu_listener, Listener(address) as cpu_listener:
        gpu_processes, cpu_processes = [], []
        for worker_id in range(args.gpu_num):
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = str(available_gpus[worker_id % len(available_gpus)])
            gpu_processes.append(Popen([
                sys.executable, gpu_worker_file,
                "--strategy", args.strategy.name,
                "--worker_id", str(worker_id),
                "--num-workers", str(args.gpu_num),
                "--port", str(gpu_listener.address[1]),
                "--net",  args.net,
                "--concretecheck-batch", str(args.concretecheck_batch),
                "--supercheck-batch", str(args.supercheck_batch),
                "--pool-size", str(args.pool_size),
                "--epsilon", str(args.epsilon),
                "--max-k", str(args.max_k),
                "--rep-each-k", str(args.rep_each_k),
                "--rep-each-level", str(args.rep_each_level)
            ], env=my_env))

        for worker_id in range(args.milp_num):
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = str(available_gpus[worker_id % len(available_gpus)])
            cpu_processes.append(Popen([
                sys.executable, solver_file,
                "--solver", args.solver,
                "--port", str(cpu_listener.address[1]),
                "--net",  args.net,
            ], env=my_env))

        print("Waiting for cpu workers ... ", end='', flush=True)
        cpu_workers = accept_workers(cpu_listener, cpu_processes, "MILP", gpu_processes + cpu_processes)
        print("Done.")

        print("Waiting for gpu workers ... ", end='', flush=True)
        gpu_workers = accept_workers(gpu_listener, gpu_processes, "GPU", gpu_processes + cpu_processes)
        gpu_workers.sort(key=lambda w: w.recv())
        print("Done.")

    engine, data_loader = get2(args)
    verifier = Verifier(
        engine=engine,
        gpu_workers=gpu_workers,
        cpu_workers=cpu_workers,
        plan_only=args.plan_only,
        strategy=args.strategy,
        probe_thickness=args.probe_thickness,
        until_timedout=args.until_timedout,
        force_reestimate=args.force_reestimate
    )

    image_results_by_image_index = {}
    tried_images = 0
    verified_images = 0
    unsafe_images = 0
    cum_time = 0.
    for _i, (image, true_label) in enumerate(data_loader):
        i = _i + args.start_idx
        numpy_img = image.reshape(-1).numpy()  # shape (1, C, H, W) -> (C*H*W)
        label = true_label.item()
        start = time.time()
        is_correctly_classified = engine.torch_net(image).argmax(dim=1).eq(true_label).all().item()
        image_results = {}
        image_results['is_correctly_classified'] = is_correctly_classified
        image_results['true_label'] = label
        adv_min_eps = sigma0_min_eps(net=args.net, img=i)
        if adv_min_eps is not None and adv_min_eps <= args.epsilon:
            print(f"Image {i} already has a known adversarial example with eps={adv_min_eps} <= epsilon={args.epsilon}.")
            skip = True
            insert_ours(net=args.net, img=i, eps=args.epsilon, found_adv=True, use_cvd=-1)
        else: skip = ours_should_skip(net=args.net, img=i, eps=args.epsilon)
        if skip: print(f"Image {i} for net={args.net}, eps={args.epsilon} should be skipped.")
        if not skip or args.force_store or args.demo:
            if is_correctly_classified:
                tried_images += 1
                if verifier.mean_boundconcrete_time == None:
                    engine.adapt_target_images(image, label)
                    verifier.mean_boundconcrete_time = engine.estimate_mean_boundconcrete_time(netname=args.net, force_reestimate=args.force_reestimate)
                print(f'Starting to analyze image {i}, label {label}')

                random.seed(args.base_seed + i)
                verifier.update_image(numpy_img, label)
                start = time.time()
                analysis_summary = verifier.verify()
                exec_details = analysis_summary.pop('exec_details', None)
                if exec_details is not None and args.detail:
                    pd.DataFrame(exec_details).to_excel(ilogname + f"i{i}.xlsx", engine='openpyxl', index_label="Row")
                image_results['analysis_summary'] = analysis_summary

                if analysis_summary['verified']:
                    verified_images += 1
                if 'adversarial_results' in analysis_summary:
                    unsafe_images += 1

                image_results['running_time'] = analysis_summary['running_time']
                cum_time += image_results['running_time']  # only count samples where we did try to certify
            else:
                print("img", i, "not considered, incorrectly classified")
                image_results['running_time'] = round(time.time() - start, 3)

            if args.demo:
                print(f"Image {i} results NOT stored")
            else:
                store_ours_result(net=args.net, eps=args.epsilon, img=i, result=image_results, skip=not args.plan_only)
                print(f"Image {i} results stored")

        image_results_by_image_index[i] = image_results

        print(
            f"progress: {1 + i - args.start_idx}/{args.test_batches}, "
            f"correct:  {tried_images}/{1 + i - args.start_idx}, "
            f"verified: {verified_images}/{tried_images}, "
            f"unsafe: {unsafe_images}/{tried_images}, ",
            f"time: {round(time.time() - start, 3)}; {0 if cum_time == 0 else cum_time / tried_images:.3f}; {cum_time:.3f}"
        )
        print('*******************************************************', flush=True)

    for worker in itertools.chain(gpu_workers, cpu_workers):
        worker.send('terminate')

    print('analysis precision ', verified_images, '/ ', tried_images)

    if export_results:
        results = {
            'netname': args.net,
            'epsilon': args.epsilon,
            'rep_each_k': args.rep_each_k,
            'rep_each_level': args.rep_each_level,
            'concretecheck_batch': args.concretecheck_batch,
            'supercheck_batch': args.supercheck_batch,
            'pool_size': args.pool_size,
            'gpu_num': args.gpu_num,
            'milp_num': args.milp_num,
            'timeout': args.timeout,
            'start_idx': args.start_idx,
            'test_num': args.test_batches,
            'plan_only': args.plan_only,
            'cumulative_time': cum_time,
            'images_results_by_index': image_results_by_image_index
        }
        with open(args.output, 'w') as fp:
            json.dump(results, fp, indent=4)
        print(f'Results saved to {args.output}')
