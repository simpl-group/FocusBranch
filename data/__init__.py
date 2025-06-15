# -*- coding:utf-8 -*-

from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass
from functools import cached_property
from torchvision.datasets import MNIST, CIFAR10, VisionDataset
from torchvision.transforms import ToTensor, Compose, Normalize
from bimm import combs2bitmasks
import torch
import csv
import numpy as np


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

    @property
    def raw_shape(self) -> tuple[int, ...]:
        return (*self.image_shape[1:], self.channel_size)

    def combs2bitmasks(self, combs: torch.Tensor) -> torch.Tensor:
        return combs2bitmasks(combs, self.bitmask_length)

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