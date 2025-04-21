# -*- coding:utf-8 -*-

import os
import onnx
from models import ONNX_FILES
from networks import FCN, CNN
from collections import defaultdict
import torch
import itertools
from data.data_reader import load_mnist_data, load_cifar10_data
import warnings

def test_model(file: str, net: torch.nn.Sequential) -> None:
    dataset = "MNIST" if "mnist" in file else "CIFAR10"
    if dataset == "MNIST":
        images, labels = mnist_imgs, mnist_labels
    else:
        images, labels = cifar10_imgs, cifar10_labels
    if isinstance(net, CNN):
        images = torch.permute(images, dims=[0, 3, 1, 2])
    net.eval()
    preds = net(images)
    acc = torch.sum(torch.argmax(preds, dim=-1) == labels).item() / labels.size(0)
    print(f"The accuracy of {os.path.basename(file)} on {dataset} testset is {acc:.2%}")


def onnx2pt(onnx_path: str, param_list: list[dict[str, int]]) -> torch.nn.Sequential:
    onnx_fullpath = onnx_path + ".onnx"
    try:
        model = onnx.load(onnx_fullpath)
        onnx.checker.check_model(model)
    except:
        raise ValueError(f"Ill-formed networks: {onnx_fullpath}")

    wb_data = defaultdict(lambda: {"bias": None, "weight": None})
    for initializer in model.graph.initializer:
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
    if len(param_list[0]) == 2:
        net = FCN(param_list)
    else:
        net = CNN(param_list)
    for (k, w_or_b), param in zip(net.state_dict().items(), wb_data):
        assert w_or_b.shape == param.shape, (w_or_b.shape, param.shape)
        w_or_b.data.copy_(param)

    return net

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert ONNX models to PyTorch models",
    )
    parser.add_argument("--save", action="store_true", help="Save the converted models")
    args = parser.parse_args()

    mnist_imgs, mnist_labels = load_mnist_data()
    mnist_imgs, mnist_labels = torch.from_numpy(mnist_imgs) / 255., torch.from_numpy(mnist_labels)
    cifar10_imgs, cifar10_labels = load_cifar10_data()
    cifar10_imgs, cifar10_labels = torch.from_numpy(cifar10_imgs) / 255., torch.from_numpy(cifar10_labels)

    for model_path, param_list in ONNX_FILES.items():
        net = onnx2pt(model_path, param_list)
        test_model(model_path, net)
        if args.save:
            torch.save(net.state_dict(), model_path + ".pt")
            print(f"Saved {model_path}.pt")