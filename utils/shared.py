from models import MODEL_SPECS, get_network_and_dataset_utils
from data.prepare import build_loader
from data import Strategy
from data.db import DB_PATH
from AbstractInterpreter import L0BoundedInput, SequentialAT
from bounding_engine import BoundingEngine
from utils import DEVICE
import argparse
import torch

def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--net", type=str, choices=MODEL_SPECS.keys(), default="afc2", help="Neural network nickname",)
    parser.add_argument("-e", "--epsilon", type=int, default=3, help="Perturbation L0 norm bound (perturbation budget).")
    parser.add_argument("-bs", "--batch-size", type=int, default=200)
    parser.add_argument("-cb", "--concretecheck-batch", type=int, default=200, help="Batch size to bound concrete nodes. Defaults to the same as --batch-size if not specified.")
    parser.add_argument("-sb", "--supercheck-batch", type=int, default=200, help="Batch size to bound super nodes. Defaults to the same as --batch-size if not specified.")
    parser.add_argument("-to", "--timeout", type=int, default=1800, help="Time limit in seconds for each image verification.")
    parser.add_argument("-s", "--start-idx", type=int, default=0, help="The starting index of the image to verify.")
    parser.add_argument("--demo", action="store_true", help="If set, it bypass the local database and force a demo verification on a single image.")
    parser.add_argument("-fs", "--force-store", action="store_true", help="Force storing results into the database even if they already exist.")
    parser.add_argument("-tb", "-n", "--test-batches", "--num-images", dest="test_batches", type=int, default=1, help="The number of images to verify.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file name for storing results.")
    parser.add_argument("--data-source", type=str, choices=["pt", "csv"], default="csv",
        help="'pt' uses a .pt file preprocessed by `data.prepare`, 'csv' uses a .csv file with images and labels.")
    parser.add_argument("--max-k", type=int, default=200, help="Maximum size of focus sets to sample from.")
    parser.add_argument("-rk", "--rep-each-k", type=int, default=200, help="Sample size of super nodes of each focus set size, per GPU worker.")
    parser.add_argument("-rl", "--rep-each-level", type=int, default=5000, help="Sample size of each level's concrete nodes, per GPU worker.")
    parser.add_argument("-ps", "--pool-size", type=int, default=2000, help='Pool size to cache super nodes for each focus set size.')
    parser.add_argument("--strategy", type=lambda s: Strategy[s], choices=list(Strategy), default=Strategy.FocusBranch, help="Strategy to use for verification.")
    parser.add_argument("-db", "--database", type=str, default=DB_PATH, help="Path to the SQLite database file to use for storing results. If not specified, the default database will be used.")
    parser.add_argument("--solver", type=str, choices=["gurobi", "abcrown"], default="gurobi", help="The solver to use for optimization.")

def get4(args: argparse.Namespace) -> tuple[torch.nn.Module, SequentialAT, L0BoundedInput, torch.utils.data.DataLoader]:
    net, dataset_utils = get_network_and_dataset_utils(args.net, device=DEVICE)
    if args.data_source == 'pt':
        images, labels = dataset_utils.load_xy_from_pt()
    elif args.data_source == 'csv':
        images, labels = dataset_utils.load_xy_from_csv(normalize=False)  # TODO: add option to normalize
    data_loader = build_loader(images, labels, batch_size=1, start_idx=args.start_idx, num_batches=args.test_batches)
    x_ = torch.zeros(1, *dataset_utils.image_shape).to(DEVICE)
    seqAT = SequentialAT.from_sequential(net, x_)
    bounded_x0 = L0BoundedInput(
        x=x_.reshape(1, -1, 1),
        eps=args.epsilon,
        channel_size=dataset_utils.channel_size,
        num_pixels=dataset_utils.num_pixels
    )
    return net, seqAT, bounded_x0, data_loader

from inspect import signature

def get2(args: argparse.Namespace) -> tuple[BoundingEngine, torch.utils.data.DataLoader]:
    net, seqAT, bounded_x0, data_loader = get4(args)
    init_param_names = signature(BoundingEngine.__init__).parameters.keys()
    init_kwargs = {
        name: getattr(args, name) for name in init_param_names if hasattr(args, name)
    }
    return BoundingEngine(
        net.cpu(), seqAT, bounded_x0, **init_kwargs
    ), data_loader

def get_engine(args: argparse.Namespace) -> BoundingEngine:
    net, dataset_utils = get_network_and_dataset_utils(args.net, device=DEVICE)
    x_ = torch.zeros(1, *dataset_utils.image_shape).to(DEVICE)
    seqAT = SequentialAT.from_sequential(net, x_)
    bounded_x0 = L0BoundedInput(
        x=x_.reshape(1, -1, 1),
        eps=args.epsilon,
        channel_size=dataset_utils.channel_size,
        num_pixels=dataset_utils.num_pixels
    )
    init_param_names = signature(BoundingEngine.__init__).parameters.keys()
    init_kwargs = {
        name: getattr(args, name) for name in init_param_names if hasattr(args, name)
    }
    return BoundingEngine(
        net.cpu(), seqAT, bounded_x0, **init_kwargs
    )
