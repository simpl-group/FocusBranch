# -*- coding:utf-8 -*-

from pathlib import Path

MODEL_DIR = Path(__file__).parent

MODEL_CONFIGS = {
    str(MODEL_DIR/"mnist/mnist-net_256x2"): [
            {"in_features": 784, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 10},
        ],
    str(MODEL_DIR/"mnist/mnist-net_256x4"): [
            {"in_features": 784, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 10},
        ],
    str(MODEL_DIR/"mnist/mnist-net_256x6"): [
            {"in_features": 784, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 10},
        ],
    str(MODEL_DIR/"mnist/mnist_0.1"): [
        {"in_channels": 1, "out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
        {"in_channels": 32, "out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 1},
        {"in_features": 3136, "out_features": 100},
        {"in_features": 100, "out_features": 10},
    ],
    str(MODEL_DIR/"mnist/mnist_0.3"): [
        {"in_channels": 1, "out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
        {"in_channels": 32, "out_channels": 128, "kernel_size": 4, "stride": 2, "padding": 1},
        {"in_features": 6272, "out_features": 250},
        {"in_features": 250, "out_features": 10},
    ],
    str(MODEL_DIR/"cifar10/cifar10_2_255"): [
        {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
        {"in_channels": 32, "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1},
        {"in_channels": 32, "out_channels": 128, "kernel_size": 4, "stride": 2, "padding": 1},
        {"in_features": 8192, "out_features": 250},
        {"in_features": 250, "out_features": 10},
    ],
    str(MODEL_DIR/"cifar10/cifar10_8_255"): [
        {"in_channels": 3, "out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
        {"in_channels": 32, "out_channels": 128, "kernel_size": 4, "stride": 2, "padding": 1},
        {"in_features": 8192, "out_features": 250},
        {"in_features": 250, "out_features": 10},
    ]
}

__keys = list(MODEL_CONFIGS.keys())

def __idx2spec(idx: int) -> tuple[list[dict[str, int]], str]:
    if idx < 0 or idx >= len(__keys):
        raise ValueError(f"Invalid index {idx}.")
    return MODEL_CONFIGS[__keys[idx]], __keys[idx]

FCN_SPEC = {
    'fc1': (*__idx2spec(0), "MNIST", False),
    'fc2': (*__idx2spec(1), "MNIST", False),
    'fc3': (*__idx2spec(2), "MNIST", False),
}
CNN_SPEC = {
    'conv1': (*__idx2spec(3), "MNIST", True),
    'conv2': (*__idx2spec(4), "MNIST", True),
    'conv3': (*__idx2spec(5), "CIFAR10", True),
    'conv4': (*__idx2spec(6), "CIFAR10", True)
}
