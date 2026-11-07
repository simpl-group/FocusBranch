import warnings
from utils import timeout_handler, TimeoutException, DEVICE, TQDM_BAR_FORMAT
from utils.shared import add_shared_arguments
from models import get_network_and_dataset_utils, MODEL_SPECS
from models.networks import make_marginals_tail
from data.prepare import build_loader
from data.db import insert_lirpa, lirpa_should_skip, lirpa_config_consistent, sigma0_min_eps
from time import perf_counter
from warnings import filterwarnings
import signal
import argparse
from tqdm import tqdm
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationL0Norm
def stop_criterion_min_ge0(): # from auto_LiRPA.utils import stop_criterion_min
    return lambda x: (x.min(1, keepdim=True).values >= 0)
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

    if not lirpa_config_consistent(net=args.net, eps=args.epsilon, first_img=args.start_idx, last_img=args.start_idx + args.test_batches - 1, method=args.method,
                                   timeout=args.timeout, max_iteration=args.max_iteration):
        warnings.warn(f"Configuration inconsistent: net={args.net}, images {args.start_idx}..{args.start_idx + args.test_batches - 1}, eps={args.epsilon}, timeout={args.timeout}, max_iteration={args.max_iteration}, method={args.method}.")
        if not args.force_store and not args.demo:
            exit(1)

    net, dataset_utils = get_network_and_dataset_utils(args.net, device=DEVICE, add_affine_normalization=False)
    need_normalization = MODEL_SPECS[args.net].need_normalization
    if args.data_source == 'pt':
        images, labels = dataset_utils.load_xy_from_pt(normalize=need_normalization)
    elif args.data_source == 'csv':
        images, labels = dataset_utils.load_xy_from_csv(normalize=need_normalization)
    data_loader = build_loader(
        images,
        labels,
        batch_size=1,
        start_idx=args.start_idx,
        num_batches=args.test_batches
    )
    img_shape = (1, *dataset_utils.image_shape)
    if need_normalization:
        mean = torch.as_tensor(dataset_utils.mean, device=DEVICE, dtype=images.dtype).view(1, -1, 1, 1)
        std = torch.as_tensor(dataset_utils.std, device=DEVICE, dtype=images.dtype).view(1, -1, 1, 1)
        img_LB = ((0.0 - mean) / std).expand(*img_shape)
        img_UB = ((1.0 - mean) / std).expand(*img_shape)
    else:
        img_LB = torch.zeros(img_shape).to(DEVICE)
        img_UB = torch.ones(img_shape).to(DEVICE)
    total_time = 0.
    num_verified = 0
    num_skipped = 0
    num_tried = 0
    net.cpu()
    pbar = tqdm(total=args.test_batches, bar_format=TQDM_BAR_FORMAT, disable=args.demo)
    for _i, (images, pred_labels) in enumerate(data_loader):
        i = _i + args.start_idx
        adv_min_eps = sigma0_min_eps(net=args.net, img=i)
        if adv_min_eps is not None and adv_min_eps <= args.epsilon:
            skip = True
            insert_lirpa(net=args.net, img=i, eps=args.epsilon, cc=True, verified=False, total_duration=0.0)
        else: skip = lirpa_should_skip(net=args.net, img=i, eps=args.epsilon)
        if skip:
            num_skipped += 1
        if not skip or args.force_store or args.demo:
            if net(images).argmax(dim=1).eq(pred_labels).all():
                num_tried += 1
                images = images.to(DEVICE)
                marginal_tail = make_marginals_tail(pred_labels.item())
                lirpa_model = BoundedModule(net+marginal_tail, torch.empty_like(images)).to(DEVICE)
                lirpa_model.set_bound_opts({
                    'optimize_bound_args':{
                        'iteration': args.max_iteration,
                        'stop_criterion_func': stop_criterion_min_ge0(),
                        'early_stop_patience': args.max_iteration,
                        }
                    }
                )
                bounded_images = BoundedTensor(images, PerturbationL0Norm(eps=args.epsilon, x_L=img_LB, x_U=img_UB))

                verified = False
                elapsed_time = 0.
                try:
                    signal.alarm(args.timeout)
                    t0 = perf_counter()
                    lb, _ = lirpa_model.compute_bounds(x=(bounded_images,), bound_upper=False, method=args.method)
                    signal.alarm(0)
                    if (lb >= 0).all():
                        verified = True
                        num_verified += 1
                except TimeoutException:
                    if args.demo: print("Timed out")

                elapsed_time = perf_counter() - t0
                total_time += elapsed_time

                if args.demo:
                    if 'lb' in locals(): print(f"Lower bounds: {lb[0].tolist()}")
                    print(f"Verified safe: {verified}, Time: {elapsed_time:.2f}s, result NOT stored")
                    break

                insert_lirpa(net=args.net, img=i, eps=args.epsilon, verified=verified, total_duration=round(elapsed_time, 3))

                del images, pred_labels, bounded_images, lirpa_model, marginal_tail
                net.cpu()
                torch.cuda.empty_cache()
            else:
                if args.demo:
                    print(f"Misclassified! Result NOT stored")
                    break
                insert_lirpa(net=args.net, img=i, eps=args.epsilon, cc=False)

        pbar.update(1)
        mean_time = f"{total_time / num_verified:.2f}s/img" if num_verified > 0 else "N/A"
        verified_percentage = f"{100*num_verified/num_tried:.2f}%" if num_tried > 0 else "N/A"
        pbar.set_postfix({"Skipped": num_skipped, "Verified": f"{num_verified}/{num_tried}|{verified_percentage}", "Avg": mean_time})

        if pbar.n == args.test_batches: break
    pbar.close()
