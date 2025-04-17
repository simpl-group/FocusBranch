# -*- coding:utf-8 -*-

import os

DIR = os.path.dirname(__file__)
MNIST_FILE = os.path.join(DIR, "mnist_test.csv")
CIFAR10_FILE = os.path.join(DIR, "cifar10_test.csv")
CIFAR10_CLASSES = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]

MNIST_PT_FILE = os.path.join(DIR, "mnist_test.pt")
CIFAR10_PT_FILE = os.path.join(DIR, "cifar10_test.pt")
