# -*- coding:utf-8 -*-

from pathlib import Path

DIR = Path(__file__).parent
MNIST_TEST_CSV = str(DIR / "mnist_test.csv")
CIFAR10_TEST_CSV = str(DIR / "cifar10_test.csv")
CIFAR10_CLASSES = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

MNIST_TEST_PT = str(DIR / "mnist_test.pt")
CIFAR10_TEST_PT = str(DIR / "cifar10_test.pt")
