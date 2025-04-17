# -*- coding:utf-8 -*-

import os
from models import MODEL_DIR, ONNX_FILES
from networks import FCN, CNN
from data import MNIST_PT_FILE
from data.data_reader import load_mnist_data
import torch

mnist_data = load_mnist_data()
images, labels = torch.from_numpy(mnist_data[0]) / 255., torch.from_numpy(mnist_data[1])

dataset = "mnist"
archs = ONNX_FILES[dataset]

indices = torch.ones_like(labels, dtype=torch.bool)
for arch, feature_sizes in archs:
    pt_file = os.path.join(MODEL_DIR, dataset, arch)[:-len('.onnx')] + ".pt"
    if all(len(sizes) == 2 for sizes in feature_sizes):
        net = FCN(feature_sizes)
    else:
        net = CNN(feature_sizes)
    net.load_state_dict(torch.load(pt_file))
    indices = torch.logical_and(indices, torch.argmax(net(images), dim=-1) == labels)

torch.save([images[indices], labels[indices]], MNIST_PT_FILE)
print(f"found {indices.sum().item()} images from MNIST dataset")
