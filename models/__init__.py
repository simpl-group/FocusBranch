# -*- coding:utf-8 -*-

import os

MODEL_DIR = os.path.dirname(__file__)

ONNX_FILES = {
    "mnist": [
        # FC
        ("mnist-net_256x2.onnx", [
            {"in_features": 784, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 10},
        ]),
        ("mnist-net_256x4.onnx", [
            {"in_features": 784, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 10},
        ]),
        ("mnist-net_256x6.onnx", [
            {"in_features": 784, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 10},
        ]),
        # ConV
        # ("mnist_0.1.onnx", [
        #     {"in_channels": 1, "out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
        #     {"in_channels": 32, "out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 1},
        #     {"in_features": 3136, "out_features": 100},
        #     {"in_features": 100, "out_features": 10},
        # ]),
        # ("mnist_0.3.onnx", [
        #     {"in_channels": 1, "out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
        #     {"in_channels": 32, "out_channels": 128, "kernel_size": 4, "stride": 2, "padding": 1},
        #     {"in_features": 6272, "out_features": 250},
        #     {"in_features": 250, "out_features": 10},
        # ]),
    ],
    "cifar10": [
        # FC
        # ConV
        # "cifar10_2_255.onnx",
        # "cifar10_8_255.onnx",
    ],
}

FCN_SPEC = {
    'fc1': (ONNX_FILES["mnist"][0][-1], os.path.join(MODEL_DIR, "mnist", "mnist-net_256x2.pt")),
    'fc2': (ONNX_FILES["mnist"][1][-1], os.path.join(MODEL_DIR, "mnist", "mnist-net_256x4.pt")),
    'fc3': (ONNX_FILES["mnist"][2][-1], os.path.join(MODEL_DIR, "mnist", "mnist-net_256x6.pt")),
}
CNN_SPEC = {
    # 'conv1': ONNX_FILES["mnist"]['mnist_0.1.onnx'][-1],
    # 'conv2': ONNX_FILES["mnist"]['mnist_0.3.onnx'][-1],
}
