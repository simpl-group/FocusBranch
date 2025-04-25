import torch
from models import MODEL_SPECS
import argparse
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame) -> None:
    raise TimeoutException()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TQDM_BAR_FORMAT = "{desc}: {n_fmt}/{total_fmt} |{percentage:3.2f}% {elapsed}<{remaining} {postfix} {rate_fmt}"

def get_args():
    parser = argparse.ArgumentParser(
        description="Verify L0 robustness of a neural network.",
    )
    parser.add_argument("--net", type=str, choices=MODEL_SPECS.keys(), default="fc2",
        help="Neural network architecture",)
    parser.add_argument("-e", "--epsilon", type=int, default=0, help="L0 norm bound.")
    parser.add_argument("-bs", "--batch-size", type=int, required=True)
    parser.add_argument("-t", "--target-label", type=int, default=None)
    parser.add_argument("-ntb", "--num-test-batches", type=int, default=100)
    parser.add_argument("-to", "--timeout", type=int, default=0)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-m", "--mute", action="store_true", help="Mute gurobi output.")
    parser.add_argument("-si", '--select-indices', type=int, nargs='*', default=[])
    parser.add_argument("-d", "--debug", action="store_true", help="Inspect a specific node.")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    signal.signal(signal.SIGALRM, timeout_handler)
    return args
