from time import perf_counter
from models import get_network
from data.prepare import get_loader
from data import DATASETS, DatasetName, DatasetUtils
from utils import *
from AbstractInterpreter import L0BoundedInput, SequentialAT
from tqdm import tqdm
import torch
from split_and_prune import split_and_prune
import signal
import argparse
from json import dump

def check_and_branch_batch(
    seqAT: SequentialAT,
    bounded_x0: L0BoundedInput,
    batch_size: int=2000,
    demo: bool=False,
    timeout: int=0,
    targeted: bool=False,
    dataset_utils: DatasetUtils=DATASETS[DatasetName.MNIST]
) -> bool:
    combs_to_check = torch.empty((1, 0), dtype=torch.int64)
    safe_bitmasks: torch.Tensor = torch.empty((0, dataset_utils.bitmask_length), dtype=torch.long)
    level = 0
    if demo:
        global branching_time, total_nodes, splits
    while level <= bounded_x0.eps:
        num_to_check = combs_to_check.shape[0]
        if num_to_check == 0: return True
        unk_combs_lst: list[torch.Tensor]  = []
        new_safe_bitmasks_lst: list[torch.Tensor] = [safe_bitmasks]
        if demo:
            safe_cnt, unk_cnt = 0, 0
            if targeted:
                min_value = float("inf")
        t0 = perf_counter()
        pbar = tqdm(total=num_to_check, desc=f"Level {level}", bar_format=TQDM_BAR_FORMAT, disable=not demo)
        with seqAT.working_on(DEVICE):
            for i in range(0, num_to_check, batch_size):
                combs_to_check_batch = combs_to_check[i:i + batch_size].to(DEVICE)

                current_batch_size = len(combs_to_check_batch)
                if bounded_x0.batch_size != current_batch_size:
                    bounded_x0.expand(current_batch_size)

                with bounded_x0.selecting(combs_to_check_batch):
                    seqAT.update_activation_rp(bounded_in=bounded_x0)
                    lrp = seqAT.tail.backsubstitution_lrp()
                    lb_l = bounded_x0.compute_lbound(lrp).squeeze(2)

                safes = lb_l.ge(0).all(dim=1)
                new_safe_bitmasks_lst.append(dataset_utils.combs2bitmasks(combs_to_check_batch[safes]).cpu())
                unk_combs_lst.append(combs_to_check_batch[~safes].cpu())

                if demo:
                    safe_cnt += new_safe_bitmasks_lst[-1].shape[0]
                    unk_cnt += unk_combs_lst[-1].shape[0]
                    posfix: dict = {"Safe:UNK": f"{safe_cnt}:{unk_cnt}"}
                    if targeted:
                        if lb_l.min() < min_value:
                            min_value = lb_l.min().item()
                            min_comb = combs_to_check_batch[lb_l.argmin()]
                        posfix["Min"] = f"{min_value:.2f}"
                        posfix["Comb"] = min_comb.tolist()
                    pbar.set_postfix(posfix)
                    pbar.update(current_batch_size)
                if timeout != 0 and i == batch_size * 100:  # calculate remaining time to decide if to early stop
                    if not demo: # calculate remaining time without pbar
                        rate = (perf_counter() - t0) / 100
                        if ((num_to_check - i) / batch_size) * rate > timeout:
                            return False
                    elif (pbar.total - pbar.n) / pbar.format_dict['rate'] > timeout:
                        signal.alarm(0)
                        raise TimeoutException()
        pbar.close()

        unk_combs = torch.cat(unk_combs_lst, dim=0)
        if unk_combs.shape[0] == 0: return True
        elif level == bounded_x0.eps: return False
        safe_bitmasks = torch.cat(new_safe_bitmasks_lst, dim=0)
        del combs_to_check, combs_to_check_batch, lb_l, lrp, new_safe_bitmasks_lst, unk_combs_lst, safes
        torch.cuda.empty_cache()
        t0 = perf_counter()
        combs_to_check = split_and_prune(unk_combs, safe_bitmasks, level, demo, dataset_utils)
        if demo:
            branching_time += perf_counter() - t0
            total_nodes += num_to_check
            splits += len(unk_combs)
        torch.cuda.empty_cache()
        level += 1
    return False

if __name__ == "__main__":
    signal.signal(signal.SIGALRM, timeout_handler)
    parser = argparse.ArgumentParser(
        description="Sample-wise (local) L0 robustness verification of neural networks"
    )
    add_shared_arguments(parser)
    args = parser.parse_args()
    if args.demo:
        args.test_batches = 1
        splits = 0
        total_nodes = 0
        branching_time = .0

    net, dataset_utils = get_network(args.net, device=DEVICE)
    data_loader = get_loader(
        dataset_utils.pt_file,
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
        bounded_x0.x = x0
        if args.target_label is not None:
            seqAT.tail.make_marginal(true_labels=pred_labels, target_label=args.target_label, device=DEVICE)
        else:
            seqAT.tail.make_marginals(true_labels=pred_labels, device=DEVICE)

        success = False
        elapsed_time = 0.
        try:
            signal.alarm(args.timeout)
            t0 = perf_counter()
            if check_and_branch_batch(
                seqAT,
                bounded_x0,
                args.batch_size,
                demo=args.demo,
                timeout=args.timeout,
                targeted=args.target_label is not None,
                dataset_utils=dataset_utils
            ):
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

        if args.demo:
            print(f"Total: Splits {splits}; Nodes {total_nodes}; Time {perf_counter() - t0:.2f}s; Branching Time {branching_time:.2f}s")
            print(f"Safe: {success}")
            break

        pbar.update(1)
        mean_time = total_time / num_verified if num_verified > 0 else 0
        pbar.set_postfix({"Verified": f"{num_verified}/{pbar.n}|{100*num_verified/pbar.n:.2f}%",
                            "Avg":f"{mean_time:.2f}s/img"})

        record.append((success, elapsed_time))
        with open(args.output, "w") as f:
            dump(record, f, indent=2)
        if pbar.n == args.test_batches: break
    pbar.close()
