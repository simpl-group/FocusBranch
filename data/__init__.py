# -*- coding:utf-8 -*-

from covering_generator import CVD_MIN_T, CVD_MAX_T
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass
from functools import cached_property
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, VisionDataset
from torchvision.transforms import ToTensor, Compose, Normalize
import torch
import csv
import numpy as np

class Strategy(Enum):
    CheSaP = auto()
    CCSaP  = auto()
    SCSaP  = auto()
    CoVerD = auto()

    def __str__(self) -> str:
        return self.name

    def require_supercheck(self) -> bool:
        return self in {Strategy.SCSaP, Strategy.CoVerD}

    def assert_strategy_applicable(self, epsilon: int):
        if self == Strategy.SCSaP:
            assert epsilon >= CVD_MIN_T, f"Strategy {self} inapplicable for epsilon {epsilon}. Covering generator requires epsilon >= {CVD_MIN_T}."
        elif self == Strategy.CoVerD:
            assert CVD_MIN_T <= epsilon <= CVD_MAX_T, f"Strategy {self} inapplicable for epsilon {epsilon}. Covering generator requires {CVD_MIN_T} <= epsilon <= {CVD_MAX_T}."

class DatasetName(Enum):
    MNIST   = auto()
    CIFAR10 = auto()
    FASHIONMNIST = auto()

    def __str__(self) -> str:
        return self.name

BASE_DIR = Path(__file__).parent

@dataclass(frozen=True)
class DatasetUtils:
    csv_file: Path
    pt_file: Path
    classes: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    image_shape: tuple[int, int, int]
    dataset_class: type[VisionDataset]

    @property
    def channel_size(self) -> int:
        return self.image_shape[0]

    @cached_property
    def num_pixels(self) -> int:
        _, h, w = self.image_shape
        return h * w

    @property
    def raw_shape(self) -> tuple[int, ...]:
        return (*self.image_shape[1:], self.channel_size)

    def affine_normalization_layer(self) -> torch.nn.Module:
        normalization_layer = torch.nn.Conv2d(
            self.channel_size, self.channel_size, kernel_size=1, stride=1, padding=0
        )
        normalization_layer.weight.data.fill_(0.)
        normalization_layer.bias.data.fill_(0.)
        for i in range(self.channel_size):
            normalization_layer.weight.data[i, i, 0, 0] = 1.0 / self.std[i]
            normalization_layer.bias.data[i] = -self.mean[i] / self.std[i]
        return normalization_layer

    def load_xy_from_csv(self, normalize: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load dataset from a CSV file into PyTorch tensors.

        :param normalize: Whether to normalize the images using the mean and std of the dataset.
        :return: Tuple of images and labels as PyTorch tensors.
        """
        images, labels = [], []
        with open(self.csv_file, 'r') as reader:
            csv_reader = csv.reader(reader, delimiter=',')
            for line in csv_reader:
                label, image = int(line[0]), np.array(line[1:], dtype=np.uint8).reshape(self.raw_shape)
                images.append(image)
                labels.append(label)

        if normalize:
            transformation = Compose([ToTensor(), Normalize(self.mean, self.std)])
        else:
            transformation = ToTensor()

        return torch.stack([transformation(img) for img in images]), torch.tensor(labels, dtype=torch.int64)

    def load_xy_from_pt(self, normalize: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = torch.load(self.pt_file, map_location="cpu")
        if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise ValueError(f"Data in {self.pt_file} must be tensors.")
        if normalize:
            images = Normalize(self.mean, self.std)(images)
        return images, labels

    def sample_pixels(self, num: int, device: str = "cpu") -> torch.Tensor:
        indices = torch.randperm(self.num_pixels, device=device)[:num]
        if self.channel_size == 1: return indices
        indices = indices.repeat(self.channel_size, -1)
        for i in range(1, self.channel_size):
            indices[i] += i * self.num_pixels
        return indices.view(-1)


DATASETS: dict[DatasetName, DatasetUtils] = {
    DatasetName.MNIST: DatasetUtils(
        csv_file=BASE_DIR / "mnist_test.csv",
        pt_file=BASE_DIR / "mnist_test.pt",
        classes=tuple(str(i) for i in range(10)),
        mean=(0.1307,),
        std=(0.3081,),
        image_shape=(1, 28, 28),
        dataset_class=MNIST,
    ),
    DatasetName.CIFAR10: DatasetUtils(
        csv_file=BASE_DIR / "cifar10_test.csv",
        pt_file=BASE_DIR / "cifar10_test.pt",
        classes=(
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ),
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        image_shape=(3, 32, 32),
        dataset_class=CIFAR10,
    ),
    DatasetName.FASHIONMNIST: DatasetUtils(
        csv_file=BASE_DIR / "fashion-mnist_test100.csv",
        pt_file=BASE_DIR / "fashion-mnist_test.pt",
        classes=(
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ),
        mean=(0.5,),
        std=(0.5,),  # from coverd.py, not the actual dataset stats should be (0.2860,), (0.3530,)
        image_shape=(1, 28, 28),
        dataset_class=FashionMNIST,
    )
}

def load_covering(v: int, k: int, t: int) -> list[list[int]]:
    covering = []
    with open(BASE_DIR / f"refinement_coverings/{t}/({v},{k},{t}).txt", 'r') as f:
        for line in f:
            covering.append([int(item) for item in line.split(',')])
    return covering

def load_covering_sizes(t: int, clipping_size: int=500000) -> dict[tuple[int, int], int]:
    covering_table_file = np.genfromtxt(BASE_DIR / f'refinement_coverings/{t}-table-refinement.csv', delimiter=',')
    covering_sizes = {}
    for v in range(t + 1, covering_table_file.shape[0] + t - 1):
        for k in range(t, v):
            size = int(covering_table_file[v - t + 1][k - t + 1])
            if size < clipping_size:
                covering_sizes[(v, k)] = size
    return covering_sizes