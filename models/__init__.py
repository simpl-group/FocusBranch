# -*- coding:utf-8 -*-

from pathlib import Path
from data import DatasetName
from dataclasses import dataclass
from typing import Callable
from torch.nn import Sequential
from .networks import FCN, CNN, FLN, get_affine_normalization
import torch

BASE_DIR = Path(__file__).parent

@dataclass(frozen=True)
class ModelSpec:
    model_path: str
    constructor: Callable[[list[dict]], Sequential]
    layer_params: list[dict[str, int]]
    dataset_name: DatasetName
    need_normalization: bool
    debug: bool = False

    @property
    def onnx_path(self) -> str:
        return self.model_path + ".onnx"

    @property
    def pt_path(self) -> str:
        return self.model_path + ".pt"


MODEL_SPECS: dict[str, ModelSpec] = {
    'fcbase': ModelSpec(
        model_path=str(BASE_DIR / "fcbase"),
        constructor=FLN,
        layer_params=[
            {"in_features": 784, "out_features": 50},
            {"in_features": 50, "out_features": 50},
            {"in_features": 50, "out_features": 50},
            {"in_features": 50, "out_features": 10},
        ],
        dataset_name=DatasetName.MNIST,
        need_normalization=False,
        debug=True
    ),
    'fc1': ModelSpec(
        model_path=str(BASE_DIR / "mnist/mnist-net_256x2"),
        constructor=FCN,
        layer_params=[
            {"in_features": 784, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 10},
        ],
        dataset_name=DatasetName.MNIST,
        need_normalization=False
    ),
    'fc2': ModelSpec(
        model_path=str(BASE_DIR / "mnist/mnist-net_256x4"),
        constructor=FCN,
        layer_params=[
            {"in_features": 784, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 10},
        ],
        dataset_name=DatasetName.MNIST,
        need_normalization=False
    ),
    'fc3': ModelSpec(
        model_path=str(BASE_DIR / "mnist/mnist-net_256x6"),
        constructor=FCN,
        layer_params=[
            {"in_features": 784, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 256},
            {"in_features": 256, "out_features": 10},
        ],
        dataset_name=DatasetName.MNIST,
        need_normalization=False
    ),
    'conv1': ModelSpec(
        model_path=str(BASE_DIR / "mnist/mnist_0.1"),
        constructor=CNN,
        layer_params=[
        {"in_channels": 1, "out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
        {"in_channels": 32, "out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 1},
        {"in_features": 3136, "out_features": 100},
        {"in_features": 100, "out_features": 10},
    ],
        dataset_name=DatasetName.MNIST,
        need_normalization=True
    ),
    'conv2': ModelSpec(
        model_path=str(BASE_DIR / "mnist/mnist_0.3"),
        constructor=CNN,
        layer_params=[
            {"in_channels": 1, "out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
            {"in_channels": 32, "out_channels": 128, "kernel_size": 4, "stride": 2, "padding": 1},
            {"in_features": 6272, "out_features": 250},
            {"in_features": 250, "out_features": 10},
        ],
        dataset_name=DatasetName.MNIST,
        need_normalization=True
    ),
    'conv3': ModelSpec(
        model_path=str(BASE_DIR / "cifar10/cifar10_2_255"),
        constructor=CNN,
        layer_params=[
            {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
            {"in_channels": 32, "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1},
            {"in_channels": 32, "out_channels": 128, "kernel_size": 4, "stride": 2, "padding": 1},
            {"in_features": 8192, "out_features": 250},
            {"in_features": 250, "out_features": 10},
        ],
        dataset_name=DatasetName.CIFAR10,
        need_normalization=True
    ),
    'conv4': ModelSpec(
        model_path=str(BASE_DIR / "cifar10/cifar10_8_255"),
        constructor=CNN,
        layer_params=[
            {"in_channels": 3, "out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
            {"in_channels": 32, "out_channels": 128, "kernel_size": 4, "stride": 2, "padding": 1},
            {"in_features": 8192, "out_features": 250},
            {"in_features": 250, "out_features": 10},
        ],
        dataset_name=DatasetName.CIFAR10,
        need_normalization=True
    )
}


def get_network(name: str, device: str="cpu", add_affine_normalization: bool=True) -> Sequential:
    if name in MODEL_SPECS:
        model_spec = MODEL_SPECS[name]
    else:
        assert False, f"Invalid network name: {name}"
    model = model_spec.constructor(model_spec.layer_params)
    if model_spec.need_normalization and add_affine_normalization:
        normalization_layer = get_affine_normalization[model_spec.dataset_name]
        model.insert(0, normalization_layer)
    model.load_state_dict(torch.load(model_spec.pt_path, weights_only=True))
    model.eval()

    return model.to(device=device)