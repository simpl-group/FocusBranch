from .constant import DEVICE
from models import MODEL_SPECS, get_network_and_dataset_utils
from data.prepare import build_loader
from AbstractInterpreter import L0BoundedInput, SequentialAT
from verifier import Verifier
import argparse
import torch

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame) -> None:
    raise TimeoutException()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TQDM_BAR_FORMAT = "{desc}: {n_fmt}/{total_fmt} |{percentage:3.2f}% {elapsed}<{remaining} {postfix} {rate_fmt}"

def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--net", type=str, choices=MODEL_SPECS.keys(), default="fc2",
        help="Neural network architecture",)
    parser.add_argument("-e", "--epsilon", type=int, required=True, help="L0 norm bound.")
    parser.add_argument("-bs", "--batch-size", type=int, default=200)
    parser.add_argument("-gs", "--grain-size", type=int, default=1)
    parser.add_argument("-tl", "--target-label", type=int, default=None)
    parser.add_argument("-to", "--timeout", type=int, default=0)
    parser.add_argument("-s", "--start-idx", type=int, default=0)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("-tb", "--test-batches", type=int, default=1)
    parser.add_argument("-o", "--output", type=str, default="output.json")
    parser.add_argument("-ds", "--data-source", type=str, choices=["pt", "csv"], default="csv",
        help="'pt' uses a .pt file preprocessed by `data.prepare`, 'csv' uses a .csv file with images and labels.")

def get4(args: argparse.Namespace) -> tuple[torch.nn.Module, SequentialAT, L0BoundedInput, torch.utils.data.DataLoader]:
    net, dataset_utils = get_network_and_dataset_utils(args.net, device=DEVICE)
    if args.data_source == 'pt':
        images, labels = dataset_utils.load_xy_from_pt()
    elif args.data_source == 'csv':
        images, labels = dataset_utils.load_xy_from_csv()
    data_loader = build_loader(
        images,
        labels,
        batch_size=args.grain_size,
        target_label=args.target_label,
        start_idx=args.start_idx,
        num_batches=args.test_batches
    )
    x_ = torch.zeros(args.grain_size, *dataset_utils.image_shape).to(DEVICE)
    seqAT = SequentialAT.from_sequential(net, x_)
    bounded_x0 = L0BoundedInput(
        x=x_.reshape(args.grain_size, -1, 1),
        eps=args.epsilon,
        channel_size=dataset_utils.channel_size,
        num_pixels=dataset_utils.num_pixels
    )
    return net, seqAT, bounded_x0, data_loader

def get2(args: argparse.Namespace) -> tuple[Verifier, torch.utils.data.DataLoader]:
    net, seqAT, bounded_x0, data_loader = get4(args)
    verifier = Verifier(
        net,
        seqAT,
        bounded_x0,
        batch_size=args.batch_size,
        targeted=args.target_label is not None,
        timeout=args.timeout,
        demo=args.demo
    )
    return verifier, data_loader
