import argparse
import time
from utils.shared import add_shared_arguments
from data.db import insert_sigma0, sigma0_min_eps
from data.prepare import build_loader
from models import get_network_and_dataset_utils

# Reference: https://github.com/sigma0-advx/sigma-zero/blob/main/sigma_zero_attack.py
import torch
from adv_lib.utils.losses import difference_of_logits
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor, nn

def sigma_zero(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    steps: int = 100,
    lr: float = 1.0,
    sigma: float = 1e-3,
    threshold: float = 0.3,
    verbose: bool = False,
    epsilon_budget=None,
    grad_norm=torch.inf,
    t = 0.01
):
    clamp = lambda tensor: tensor.data.add_(inputs.data).clamp_(min=0, max=1).sub_(inputs.data)
    l0_approximation = lambda tensor, sigma: tensor.square().div(tensor.square().add(sigma)).sum(dim=1)
    batch_view = lambda tensor: tensor.view(tensor.shape[0], *[1] * (inputs.ndim - 1))
    normalize = lambda tensor: (
            tensor.flatten(1) / tensor.flatten(1).norm(p=grad_norm, dim=1, keepdim=True).clamp_(min=1e-12)).view(
        tensor.shape)

    device = next(model.parameters()).device
    batch_size, max_size = inputs.shape[0], torch.prod(torch.tensor(inputs.shape[1:]))

    delta = torch.zeros_like(inputs, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr / 10)
    best_delta = delta.clone()
    query_mask = torch.full((batch_size,), True, device=device)
    best_l0 = torch.full((batch_size,), max_size, device=device)
    is_adv_below_eps = torch.full((batch_size,), False, device=device)
    th = torch.ones(size=inputs.shape, device=device) * threshold

    for i in range(steps):
        optimizer.zero_grad()

        active_delta = delta[query_mask].clone().detach().requires_grad_(True)
        active_inputs = inputs[query_mask]
        active_labels = labels[query_mask]

        adv_inputs = active_inputs + active_delta

        # compute loss
        logits = model(adv_inputs)
        dl_loss = difference_of_logits(logits, active_labels).clip(0)
        l0_approx = l0_approximation(active_delta.flatten(1), sigma)
        l0_approx_normalized = l0_approx / active_delta.data.flatten(1).shape[1]

        # keep best solutions
        predicted_classes = (logits).argmax(1)
        true_l0 = active_delta.data.flatten(1).ne(0).sum(dim=1)
        is_not_adv = predicted_classes == active_labels
        is_smaller = true_l0 < best_l0[query_mask]
        is_both = ~is_not_adv & is_smaller
        best_l0[query_mask] = torch.where(is_both, true_l0.detach(), best_l0[query_mask])
        best_delta[query_mask] = torch.where(batch_view(is_both), active_delta.data.clone().detach(), best_delta[query_mask])
        is_adv_below_eps = best_l0 <= epsilon_budget if epsilon_budget is not None else is_adv_below_eps

        # update step
        adv_loss = (is_not_adv + dl_loss + l0_approx_normalized).mean()

        if verbose and i % 100 == 0:
            print(th.flatten(1).mean(dim=1), th.flatten(1).mean(dim=1).shape)
            print(is_not_adv)
            print(
                f"iter: {i}, dl loss: {dl_loss.mean().item():.4f}, l0 normalized loss: {l0_approx_normalized.mean().item():.4f}, current median norm: {delta.data.flatten(1).ne(0).sum(dim=1).median()}")

        adv_loss.backward()

        if delta.grad is None:
            delta.grad = torch.zeros_like(delta, device=device)
        # Copy gradients from active_delta.grad to delta.grad at the masked positions
        delta.grad[query_mask] += active_delta.grad

        delta.grad.data = normalize(delta.grad.data)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            # enforce box constraints
            clamp(delta.data)
            # dynamic thresholding step
            th_active = th[query_mask]

            th_active[is_not_adv, :, :, :] -= t * scheduler.get_last_lr()[0]
            th_active[~is_not_adv, :, :, :] += t * scheduler.get_last_lr()[0]
            th[query_mask] = th_active
            th.clamp_(0, 1)
            # filter components
            delta.data[delta.data.abs() < th] = 0
            # update active set
            query_mask[is_adv_below_eps] = False
            if not any(query_mask):
                break

    return best_delta.detach()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample-wise (local) L0 robustness verification for neural networks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_shared_arguments(parser)
    parser.add_argument('--base_seed', type=int, default=42, help='the base seed')
    parser.add_argument('--step', type=int, default=100, help='the number of steps for the attack')
    parser.add_argument("--verbose", '-v', action="store_true", help="Output verbose logs during the attack.")
    args = parser.parse_args()
    net, dataset_utils = get_network_and_dataset_utils(args.net)
    images, labels = dataset_utils.load_xy_from_csv(normalize=False)
    data_loader = build_loader(
        images,
        labels,
        batch_size=1,
        target_label=args.target_label,
        start_idx=args.start_idx,
        num_batches=args.test_batches
    )
    torch.manual_seed(args.base_seed)

    for _i, (image, true_label) in enumerate(data_loader):
        i = args.start_idx + _i
        print(f"{_i+1:3}/{args.test_batches:3} Image{i:3}", end=': ', flush=True)
        if not net(image).argmax(dim=1).eq(true_label).all().item():
            print(f"Misclassified, skipping.", end='\r', flush=True)
            continue
        previous_eps = sigma0_min_eps(net=args.net, img=i)
        if previous_eps is not None:
            print(f"Previous min_eps:{previous_eps:2}", end=', ', flush=True)
        start_time = time.time()
        delta = sigma_zero(
            model=net.cuda(), inputs=image.cuda(), labels=true_label.cuda(), epsilon_budget=args.epsilon, steps=args.step, verbose=args.verbose
        ).to('cpu')
        duration = round(time.time() - start_time, 3)
        min_eps = delta.flatten(1).ne(0).sum(dim=1).item()
        adv_image = image + delta
        net.to('cpu')
        adv_label = net(adv_image).argmax(dim=1).item()
        print(f"Min eps:{min_eps:2}, Attack time:{duration:>7}s, label: {true_label.item()}->{adv_label}", end=', ', flush=True)
        if args.verbose:
            print(f"Original indices and values: {[(idx.item(), image.flatten()[idx].item()) for idx in (delta.flatten(1).ne(0)).nonzero(as_tuple=True)[1]]}")
            print(f"Delta pixels indices and values: {[(idx.item(), delta.flatten()[idx].item()) for idx in (delta.flatten(1).ne(0)).nonzero(as_tuple=True)[1]]}")
            adversarial_pixel_values: dict[int, float] = {idx.item(): adv_image.flatten()[idx].item() for idx in (delta.flatten(1).ne(0)).nonzero(as_tuple=True)[1]}
            print(f"Adversarial indices and values: {adversarial_pixel_values}")
        if not args.force_store and (args.demo or previous_eps is not None and min_eps >= previous_eps):
            print("Results NOT stored", end='\r', flush=True)
        else:
            insert_sigma0(net=args.net, img=i, min_eps=min_eps, true_label=true_label.item(), adversarial_label=adv_label, total_duration=duration, seed=args.base_seed)
            print("Results stored    ")
    print(" "*120, end='\r', flush=True)