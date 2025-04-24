import torch
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.nn import Sequential
from data import DATASETS, DatasetName
from models import MODEL_SPECS
import argparse
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame) -> None:
    raise TimeoutException()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tqdm_bar_format = "{desc}: {n_fmt}/{total_fmt} |{percentage:3.2f}% {elapsed}<{remaining} {postfix} {rate_fmt}"
num_features = 28 * 28

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

def get_mnist_loader(args: argparse.Namespace) -> DataLoader:
    images, labels = torch.load(DATASETS[DatasetName.MNIST].pt_file, map_location="cpu")
    if args.target_label is not None:
        non_targeted_indices = (labels != args.target_label).nonzero().squeeze(1)
        images, labels = images[non_targeted_indices], labels[non_targeted_indices]
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

def preprocess(args: argparse.Namespace, net: Sequential, mean: float=0, sigma: float=1, demo: bool=True) -> DataLoader:
    """Filter and collect correctly classified target samples into a DataLoader"""
    test_dataset = MNIST(root='./data/test', train=False, transform=Compose([ToTensor(), Normalize(mean, sigma)]), download=True)
    if args.target_label is not None:
        non_targeted_indices = (test_dataset.targets != args.target_label).nonzero().squeeze(1)
        test_dataset = Subset(test_dataset, non_targeted_indices)

    correct_images = []
    correct_labels = []
    batch_size = min(256, len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
            batch_preds = net(batch_images).argmax(1)
            correct_idx = (batch_preds == batch_labels).nonzero().squeeze(1)

            correct_images.append(batch_images[correct_idx])
            correct_labels.append(batch_preds[correct_idx])

            if demo and sum(len(imgs) for imgs in correct_images) >= args.batch_size:
                break

    correct_dataset = TensorDataset(torch.cat(correct_images).cpu(), torch.cat(correct_labels).cpu())
    correct_loader = DataLoader(correct_dataset, batch_size=args.batch_size, shuffle=True)

    return correct_loader

