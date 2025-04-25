import time
from models import get_network
from data.prepare import get_loader
from data import DATASETS
from utils import get_args, TQDM_BAR_FORMAT, DEVICE, TimeoutException
from AbstractInterpreter import L0BoundedInput, SequentialAT
from tqdm import tqdm
import torch
from split_and_prune import split_and_prune
from signal import alarm
MAX_BATCH_SIZE = 25000

def check_and_branch_batch(seqAT: SequentialAT, bounded_x0: L0BoundedInput, batch_size: int=2000, demo: bool=False, timeout: int=0) -> bool:
    combs_to_check = torch.empty((1, 0), dtype=torch.int64)
    safe_combs: list[list[int]] = []
    level = 0
    watchout = False  # safe combs might contain combs that comes from different levels
    if demo:
        global branching_time, total_nodes, splits
    while level <= bounded_x0.eps:
        num_to_check = combs_to_check.shape[0]
        if num_to_check == 0: return True
        unk_combs: list[list[int]]  = []
        new_safe_combs: list[list[int]] = []
        if demo:
            min_value = float("inf")
            min_comb = None
        t0 = time.time()
        with tqdm(total=num_to_check, desc=f"Level {level}", bar_format=TQDM_BAR_FORMAT, disable=not demo) as pbar:
            for i in range(0, num_to_check, batch_size):
                combs_to_check_batch = combs_to_check[i:i + batch_size]

                current_batch_size = len(combs_to_check_batch)
                if bounded_x0.batch_size != current_batch_size:
                    bounded_x0.expand(current_batch_size)

                with bounded_x0.selecting(combs_to_check_batch):
                    seqAT.update_activation_rp(bounded_in=bounded_x0)
                    lrp = seqAT.tail.backsubstitution_lrp()
                    lb_l = bounded_x0.compute_lbound(lrp).reshape(-1).cpu()

                new_safe_combs += combs_to_check_batch[lb_l >= 0].tolist()
                unk_combs += combs_to_check_batch[lb_l < 0].tolist()

                if demo:
                    if lb_l.min() < min_value:
                        min_value = lb_l.min().item()
                        min_comb = combs_to_check_batch[lb_l.argmin()]
                    pbar.set_postfix({"Safe:UNK": f"{len(new_safe_combs)}:{len(unk_combs)}", "Min": f"{min_value:.2f}", "Comb": min_comb.tolist()})
                    pbar.update(current_batch_size)
                if timeout != 0 and i == batch_size * 100:  # calculate remaining time to decide if to early stop
                    if not demo: # calculate remaining time without pbar
                        rate = (time.time() - t0) / 100
                        if ((num_to_check - i) / batch_size) * rate > timeout:
                            return False
                    elif (pbar.total - pbar.n) / pbar.format_dict['rate'] > timeout:
                        alarm(0)
                        raise TimeoutException()

        if len(unk_combs) == 0: return True
        elif level == bounded_x0.eps: return False
        if safe_combs is not []:
            watchout = True
            safe_combs += new_safe_combs
        else:
            safe_combs = new_safe_combs
        del combs_to_check, combs_to_check_batch, lb_l, lrp, new_safe_combs
        torch.cuda.empty_cache()
        t0 = time.time()
        combs_to_check = split_and_prune(unk_combs, safe_combs, level, demo, watchout)
        if demo:
            branching_time += time.time() - t0
            total_nodes += num_to_check
            splits += len(unk_combs)
        del unk_combs
        torch.cuda.empty_cache()
        level += 1
    return False

def check_and_branch_batch_untargeted(seqAT: SequentialAT, bounded_x0: L0BoundedInput, batch_size: int=2000, demo: bool=False, timeout: int=0) -> bool:
    combs_to_check = torch.empty((1, 0), dtype=torch.int64)
    safe_combs: list[list[int]] = []
    level = 0
    watchout = False
    if demo:
        global branching_time, total_nodes, splits
    while level <= bounded_x0.eps:
        num_to_check = combs_to_check.shape[0]
        if num_to_check == 0: return True
        unk_combs: list[list[int]]  = []
        new_safe_combs: list[list[int]] = []
        t0 = time.time()
        with tqdm(total=num_to_check, desc=f"Level {level}", bar_format=TQDM_BAR_FORMAT, disable=not demo) as pbar:
            for i in range(0, num_to_check, batch_size):
                combs_to_check_batch = combs_to_check[i:i + batch_size]

                current_batch_size = len(combs_to_check_batch)
                if bounded_x0.batch_size != current_batch_size:
                    bounded_x0.expand(current_batch_size)

                with bounded_x0.selecting(combs_to_check_batch):
                    seqAT.update_activation_rp(bounded_in=bounded_x0)
                    lrp = seqAT.tail.backsubstitution_lrp()
                    lb_l = bounded_x0.compute_lbound(lrp).squeeze(2).cpu()

                safes = lb_l.ge(0).all(dim=1)
                new_safe_combs += combs_to_check_batch[safes].tolist()
                unk_combs += combs_to_check_batch[~safes].tolist()

                if demo:
                    pbar.set_postfix({"Safe:UNK": f"{len(new_safe_combs)}:{len(unk_combs)}"})
                    pbar.update(current_batch_size)
                if timeout != 0 and i == batch_size * 100:
                    if not demo:
                        rate = (time.time() - t0) / 100
                        if ((num_to_check - i) / batch_size) * rate > timeout:
                            return False
                    elif (pbar.total - pbar.n) / pbar.format_dict['rate'] > timeout:
                        alarm(0)
                        raise TimeoutException()

        if len(unk_combs) == 0: return True
        elif level == bounded_x0.eps: return False
        if safe_combs is not []:
            watchout = True
            safe_combs += new_safe_combs
        else:
            safe_combs = new_safe_combs
        del combs_to_check, combs_to_check_batch, lb_l, lrp, new_safe_combs
        torch.cuda.empty_cache()
        t0 = time.time()
        combs_to_check = split_and_prune(unk_combs, safe_combs, level, demo, watchout)
        if demo:
            branching_time += time.time() - t0
            total_nodes += num_to_check
            splits += len(unk_combs)
        del unk_combs
        torch.cuda.empty_cache()
        level += 1
    return False

if __name__ == "__main__":
    args = get_args()
    batch_size = args.batch_size if args.batch_size < MAX_BATCH_SIZE else MAX_BATCH_SIZE
    net, spec = get_network(args.net, device=DEVICE)
    data_loader = get_loader(spec.dataset_name, batch_size=1, target_label=args.target_label, num_batches=args.seed+1)
    x_ = torch.zeros(1, *DATASETS[spec.dataset_name].image_shape).to(DEVICE)
    seqAT = SequentialAT.from_sequential(net, x_)
    i = 0
    for images, pred_labels in data_loader:
        if i < args.seed:
            i += 1
            continue
        assert net.cpu()(images).argmax(dim=1).eq(pred_labels).all()
        x0 = images.to(DEVICE).flatten(start_dim=1).unsqueeze(2)  # (batch, d1, 1)
        bounded_x0 = L0BoundedInput(x0, args.epsilon)
        if args.target_label is not None:
            seqAT.tail.make_marginal(true_labels=pred_labels, target_label=args.target_label, device=DEVICE)
        else:
            seqAT.tail.make_marginals(true_labels=pred_labels, device=DEVICE)
        break
    del net

    if not args.mute:
        print("Preprocessing done")
        splits = 0
        total_nodes = 0
        branching_time = .0
    start = time.time()

    ret = False
    try:
        alarm(args.timeout)
        if args.target_label is not None:
            ret = check_and_branch_batch(seqAT, bounded_x0, batch_size, demo=not args.mute, timeout=args.timeout)
        else:
            ret = check_and_branch_batch_untargeted(seqAT, bounded_x0, batch_size, demo=not args.mute, timeout=args.timeout)
        alarm(0)
    except TimeoutException:
        print("Timeout")

    if not args.mute:
        print(f"Total: Splits {splits}; Nodes {total_nodes}; Time {time.time() - start:.2f}s; Branching Time {branching_time:.2f}s")
    else:
        print(f"Time {time.time() - start:.2f}s")
    print(f"Safe: {ret}")
