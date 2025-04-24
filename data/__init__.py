# -*- coding:utf-8 -*-

from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass
from functools import cached_property
import torch


class DatasetName(Enum):
    MNIST   = auto()
    CIFAR10 = auto()

    def __str__(self) -> str:
        return self.name

BASE_DIR = Path(__file__).parent

@dataclass(frozen=True)
class DatasetConfig:
    csv_file: Path
    pt_file: Path
    classes: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    image_shape: tuple[int, int, int]

    @cached_property
    def num_pixels(self) -> int:
        _, h, w = self.image_shape
        return h * w

    @cached_property
    def bitmask_length(self) -> int:
        return (self.num_pixels + 63) // 64


DATASETS: dict[DatasetName, DatasetConfig] = {
    DatasetName.MNIST: DatasetConfig(
        csv_file=BASE_DIR / "mnist_test.csv",
        pt_file=BASE_DIR / "mnist_test.pt",
        classes=tuple(str(i) for i in range(10)),
        mean=(0.1307,),
        std=(0.3081,),
        image_shape=(1, 28, 28),
    ),
    DatasetName.CIFAR10: DatasetConfig(
        csv_file=BASE_DIR / "cifar10_test.csv",
        pt_file=BASE_DIR / "cifar10_test.pt",
        classes=(
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ),
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        image_shape=(3, 32, 32),
    ),
}