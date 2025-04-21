# -*- coding:utf-8 -*-

from pathlib import Path

DIR = Path(__file__).parent
MNIST_TEST_CSV = str(DIR / "mnist_test.csv")
CIFAR10_TEST_CSV = str(DIR / "cifar10_test.csv")
CIFAR10_CLASSES = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]

MNIST_TEST_PT = str(DIR / "mnist_test.pt")
CIFAR10_TEST_PT = str(DIR / "cifar10_test.pt")
