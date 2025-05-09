# -*- coding:utf-8 -*-

from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass
from functools import cached_property
from torchvision.datasets import MNIST, CIFAR10, VisionDataset
import torch
from bimm import combs2bitmasks


class DatasetName(Enum):
    MNIST   = auto()
    CIFAR10 = auto()

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

    @cached_property
    def bitmask_length(self) -> int:
        return (self.num_pixels + 63) // 64

    def get_empty_bitmask(self, device: str = "cpu") -> torch.Tensor:
        return torch.empty((0, self.bitmask_length), dtype=torch.int64, device=device)

    def combs2bitmasks(self, combs: torch.Tensor) -> torch.Tensor:
        return combs2bitmasks(combs, self.bitmask_length)

    def comb2bitmask(self, comb: list[int]) -> torch.Tensor:
        mask = torch.zeros(self.bitmask_length, dtype=torch.int64)
        for idx in comb:
            mask[idx // 64] |= (torch.tensor(1, dtype=torch.int64) << idx % 64)
        return mask

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
}