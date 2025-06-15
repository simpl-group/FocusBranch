import torch
from models import MODEL_SPECS
import argparse

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
    parser.add_argument("-bs", "--batch-size", type=int, default=5)
    parser.add_argument("-tl", "--target-label", type=int, default=None)
    parser.add_argument("-to", "--timeout", type=int, default=0)
    parser.add_argument("-s", "--start-idx", type=int, default=0)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("-tb", "--test-batches", type=int, default=500)
    parser.add_argument("-o", "--output", type=str, default="output.json")
    parser.add_argument("-ds", "--data-source", type=str, choices=["pt", "csv"], default="csv",
        help="'pt' uses a .pt file preprocessed by `data.prepare`, 'csv' uses a .csv file with images and labels.")
