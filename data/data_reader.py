# -*- coding:utf-8 -*-

import csv
import numpy as np
from data import MNIST_TEST_CSV, CIFAR10_TEST_CSV, CIFAR10_CLASSES


def load_mnist_data():
    images, labels = [], []
    with open(MNIST_TEST_CSV, 'r') as reader:
        csv_reader = csv.reader(reader, delimiter=',')
        for line in csv_reader:
            label, image = int(line[0]), np.array(line[1:], dtype=np.int32).reshape([28, 28, 1])
            images.append(image)
            labels.append(label)
    images, labels = np.array(images, dtype=np.int32), np.array(labels, dtype=np.int32)
    return images, labels

def load_cifar10_data():
    labels, images = [], []
    with open(CIFAR10_TEST_CSV, 'r') as reader:
        csv_reader = csv.reader(reader, delimiter=',')
        for line in csv_reader:
            label, image = int(line[0]), np.array(line[1:], dtype=np.int32).reshape([32, 32, 3])
            labels.append(label)
            images.append(image)
    labels, images = np.array(labels, dtype=np.int32), np.array(images, dtype=np.int32)
    return labels, images


if __name__ == '__main__':
    mnist_data = load_mnist_data()
    cifar10_data = load_cifar10_data()

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    img1, label1 = mnist_data[0]
    img1 = np.asarray(img1[0, ..., 0] * 255, np.int_)
    ax1.imshow(img1, cmap='gray')
    ax1.title.set_text(label1)

    ax2 = fig.add_subplot(122)
    img2, label2 = cifar10_data[0]
    img2 = np.asarray(img2[0, ..., 0] * 255, np.int_)
    ax2.imshow(img2)
    ax2.title.set_text(CIFAR10_CLASSES[label2])
    plt.show()
