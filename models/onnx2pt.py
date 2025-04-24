# -*- coding:utf-8 -*-

import os
import onnx
from models import MODEL_SPECS
from models.networks import FCN, CNN, get_affine_normalization
from collections import defaultdict
import torch
import itertools
from torch import nn
from data import DatasetName
from data.data_reader import load_mnist_from_csv, load_cifar10_from_csv

def test_accuracy(net: nn.Sequential, images: torch.Tensor, labels: torch.Tensor) -> float:
    preds: torch.Tensor = net(images).argmax(dim=1)
    return (preds == labels).sum().item() / labels.size(0)

def load_from_onnx(onnx_path: str, torch_net: nn.Sequential) -> None:
    try:
        onnx_net = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_net)
    except:
        raise ValueError(f"Ill-formed networks: {onnx_path}")

    wb_data = defaultdict(lambda: {"bias": None, "weight": None})
    for initializer in onnx_net.graph.initializer:
        type = initializer.name.split('.')[-1]
        if type not in ["weight", "bias"]:
            warnings.warn(f"Unknown type {type} in {initializer.name}", UserWarning)
            continue
        else:
            idx = [int(n) for n in initializer.name.split('.') if str.isdigit(n)][0]
            nparray = onnx.numpy_helper.to_array(initializer)
            wb_data[idx][type] = torch.from_numpy(nparray)
    # param_list can be generated here
    wb_data = [[wb_data[idx]["weight"], wb_data[idx]["bias"]] for idx in sorted(wb_data.keys())]
    wb_data = list(itertools.chain(*wb_data))

    for (k, w_or_b), param in zip(torch_net.state_dict().items(), wb_data):
        assert w_or_b.shape == param.shape, (w_or_b.shape, param.shape)
        w_or_b.data.copy_(param)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert ONNX models to PyTorch models",
    )
    parser.add_argument("--save", action="store_true", help="Save the converted models")
    parser.add_argument("-n", "--normalize", action="store_true", help="Normalize the images")
    parser.add_argument("-anl", "--add-normalization-layer", action="store_true",
        help="Add affine layer at the beginning of the network to normalize the input equivalently")
    parser.add_argument("-cnn", "--cnn-only", action="store_true", help="Only convert CNN models")
    parser.add_argument("-fcn", "--fcn-only", action="store_true", help="Only convert FCN models")
    parser.add_argument("-mnist", "--mnist-only", action="store_true", help="Only convert MNIST models")
    parser.add_argument("-cifar10", "--cifar10-only", action="store_true", help="Only convert CIFAR10 models")
    parser.add_argument("-f", "--force", action="store_true", help="Force conversion even if the model is already converted")
    args = parser.parse_args()

    if not args.cifar10_only:
        mnist_imgs, mnist_labels = load_mnist_from_csv(normalize=args.normalize)
    if not args.mnist_only:
        cifar10_imgs, cifar10_labels = load_cifar10_from_csv(normalize=args.normalize)
    for model_spec in MODEL_SPECS.values():
        if not os.path.exists(model_spec.onnx_path): continue
        if args.cnn_only and model_spec.constructor == FCN: continue
        if args.fcn_only and model_spec.constructor == CNN: continue
        torch_net = model_spec.constructor(model_spec.layer_params)
        if model_spec.dataset_name == DatasetName.MNIST:
            if args.cifar10_only: continue
            imgs, labels = mnist_imgs, mnist_labels
        elif model_spec.dataset_name == DatasetName.CIFAR10:
            if args.mnist_only: continue
            imgs, labels = cifar10_imgs, cifar10_labels
        else: assert False, "Unknown dataset"

        load_from_onnx(model_spec.onnx_path, torch_net)

        if args.add_normalization_layer and model_spec.need_normalization:
            normalization_layer = get_affine_normalization[model_spec.dataset_name]
            torch_net.insert(0, normalization_layer)

        acc = test_accuracy(torch_net.eval(), imgs, labels)
        print(f"The accuracy of {os.path.basename(model_spec.model_path)} on {model_spec.dataset_name} testset is {acc:.2%}")

        if args.save:
            if not args.force and os.path.exists(model_spec.pt_path):
                print(f"Model {model_spec.pt_path} already exists. Use --force to overwrite.")
                continue
            torch.save(torch_net.state_dict(), model_spec.pt_path)
            print(f"Model {model_spec.pt_path} saved.")