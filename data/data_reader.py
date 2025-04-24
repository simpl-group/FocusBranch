# -*- coding:utf-8 -*-

import csv
import torch
import numpy as np
from data import *
from torchvision.transforms import ToTensor, Compose, Normalize


def load_mnist_from_csv(csv_path: str=MNIST_TEST_CSV, normalize: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load MNIST data from a CSV file into PyTorch tensors.

    :param csv_path: Path to the CSV file.
    :param normalize: Whether to normalize the images using the mean and standard deviation of the MNIST dataset.
    :return: Tuple of images and labels as PyTorch tensors.
    """
    images, labels = [], []
    with open(csv_path, 'r') as reader:
        csv_reader = csv.reader(reader, delimiter=',')
        for line in csv_reader:
            label, image = int(line[0]), np.array(line[1:], dtype=np.uint8).reshape([28, 28, 1])
            images.append(image)
            labels.append(label)
    if normalize:
        mnist_transformation = Compose([ToTensor(), Normalize(MNIST_MEAN, MNIST_STD)])
    else:
        mnist_transformation = Compose([ToTensor()])
    return torch.stack([mnist_transformation(img) for img in images]), torch.tensor(labels, dtype=torch.uint8)


def load_cifar10_from_csv(csv_path: str=CIFAR10_TEST_CSV, normalize: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load CIFAR-10 data from a CSV file into PyTorch tensors.

    :param csv_path: Path to the CSV file.
    :return: Tuple of images and labels as PyTorch tensors.
    """
    images, labels = [], []
    with open(csv_path, 'r') as reader:
        csv_reader = csv.reader(reader, delimiter=',')
        for line in csv_reader:
            label, image = int(line[0]), np.array(line[1:], dtype=np.uint8).reshape([32, 32, 3])
            images.append(image)
            labels.append(label)
    if normalize:
        cifar10_transformation = Compose([ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    else:
        cifar10_transformation = Compose([ToTensor()])
    return torch.stack([cifar10_transformation(img) for img in images]), torch.tensor(labels, dtype=torch.uint8)


if __name__ == '__main__':
    mnist_imgs, mnist_labels = load_mnist_from_csv()
    cifar10_imgs, cifar10_labels = load_cifar10_from_csv()

    import matplotlib.pyplot as plt
    # plot 20 images from mnist and cifar10
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for i in range(4):
        for j in range(5):
            axes[i, j].imshow(mnist_imgs[i * 5 + j].reshape(28, 28), cmap='gray')
            axes[i, j].set_title(f"Label: {mnist_labels[i * 5 + j]}")
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig("mnist.png")
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for i in range(4):
        for j in range(5):
            axes[i, j].imshow(cifar10_imgs[i * 5 + j])
            axes[i, j].set_title(f"Label: {CIFAR10_CLASSES[cifar10_labels[i * 5 + j]]}")
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig("cifar10.png")
