import warnings
from utils import timeout_handler, TimeoutException, DEVICE, TQDM_BAR_FORMAT
from utils.shared import add_shared_arguments
from models import get_network_and_dataset_utils
from models.networks import make_marginal_tail, make_marginals_tail
from data.prepare import build_loader
from data.db import insert_lirpa, lirpa_should_skip, lirpa_config_consistent
from time import perf_counter
from warnings import filterwarnings
import signal
import argparse
from tqdm import tqdm
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationL0Norm
from auto_LiRPA.utils import stop_criterion_min
import torch

filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=ResourceWarning)

if __name__ == "__main__":
    signal.signal(signal.SIGALRM, timeout_handler)
    parser = argparse.ArgumentParser(
        description="Sample-wise (local) L0 robustness verification of neural networks. \
            Baseline appoach: LiRPA, backward pass with alpha-CROWN."
    )
    add_shared_arguments(parser)
    parser.add_argument("-mi", "--max-iteration", type=int, default=1000)
    parser.add_argument("--method", type=str, default="CROWN-Optimized", choices=["CROWN-Optimized", 'backward'])
    args = parser.parse_args()

    if not lirpa_config_consistent(net=args.net, eps=args.epsilon, img=args.start_idx+args.test_batches, method=args.method, timeout=args.timeout, max_iteration=args.max_iteration):
        warnings.warn(f"Configuration inconsistent: net={args.net}, img={args.start_idx+args.test_batches}, eps={args.epsilon}, timeout={args.timeout}, max_iteration={args.max_iteration}, method={args.method}.")
        if not args.force_store and not args.demo:
            exit(1)

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
    total_time = 0.
    num_verified = 0
    net.cpu()
    pbar = tqdm(total=args.test_batches, bar_format=TQDM_BAR_FORMAT, disable=args.demo)
    for _i, (images, pred_labels) in enumerate(data_loader):
        i = _i + args.start_idx
        skip = lirpa_should_skip(net=args.net, img=i, eps=args.epsilon)
        if skip:
            warnings.warn(f"Image {i} for net={args.net}, eps={args.epsilon} should be skipped.")
        if not skip or args.force_store or args.demo:
            if net(images).argmax(dim=1).eq(pred_labels).all():
                images = images.to(DEVICE)
                if args.target_label is not None:
                    marginal_tail = make_marginal_tail(pred_labels.item(), args.target_label)
                else:
                    marginal_tail = make_marginals_tail(pred_labels.item())
                lirpa_model = BoundedModule(net+marginal_tail, torch.empty_like(images)).to(DEVICE)
                lirpa_model.set_bound_opts({'optimize_bound_args':{
                    'iteration': args.max_iteration, "stop_criterion_func": stop_criterion_min()}})
                bounded_images = BoundedTensor(images, PerturbationL0Norm(eps=args.epsilon))

                success = False
                elapsed_time = 0.
                timed_out = False
                try:
                    signal.alarm(args.timeout)
                    t0 = perf_counter()
                    lb, _ = lirpa_model.compute_bounds(
                        x=(bounded_images,), bound_upper=False, method=args.method)
                    signal.alarm(0)
                    if (lb >= 0).all():
                        success = True
                        num_verified += 1
                except TimeoutException:
                    timed_out = True
                    if args.demo: print("Timed out")

                elapsed_time = perf_counter() - t0
                total_time += elapsed_time

                if args.demo:
                    print(f"Lower bounds: {lb[0].tolist()}")
                    print(f"Verified safe: {success}, Time: {elapsed_time:.2f}s, result NOT stored")
                    break

                insert_lirpa(
                    net=args.net, img=i, eps=args.epsilon, verified=success, timed_out=timed_out, total_duration=round(elapsed_time, 3)
                )

                del images, pred_labels, bounded_images, lirpa_model, marginal_tail
                net.cpu()
                torch.cuda.empty_cache()
            else:
                if args.demo:
                    print(f"Misclassified! Result NOT stored")
                    break
                insert_lirpa(net=args.net, img=i, eps=args.epsilon, cc=False)

        pbar.update(1)
        mean_time = total_time / num_verified if num_verified > 0 else 0
        pbar.set_postfix({"Verified": f"{num_verified}/{pbar.n}|{100*num_verified/pbar.n:.2f}%", "Avg":f"{mean_time:.2f}s/img"})

        if pbar.n == args.test_batches: break
    pbar.close()
