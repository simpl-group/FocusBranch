# -*- coding:utf-8 -*-

import os
import onnx
from models import MODEL_DIR, ONNX_FILES
from networks import FCN, CNN
from collections import defaultdict
import torch
import itertools
from data.data_reader import load_mnist_data

mnist_data = load_mnist_data()
mnist_data = [torch.from_numpy(mnist_data[0]) / 255., torch.from_numpy(mnist_data[1])]


def test_model(file, model):
    images, labels = mnist_data
    if isinstance(model, CNN):
        images = torch.permute(images, dims=[0, 3, 1, 2])
    model.eval()
    preds = model(images)
    acc = torch.sum(torch.argmax(preds, dim=-1) == labels).item() / labels.size(0)
    print(f"The accuracy of {os.path.basename(file)} on MNIST testset is {acc:.2%}")


def layer_func():
    return {"bias": None, "weight": None}


for dataset, archs in ONNX_FILES.items():
    for arch, feature_sizes in archs:
        src_file = os.path.join(MODEL_DIR, dataset, arch)
        dst_file = src_file[:str.rfind(src_file, '.onnx')] + ".pt"

        # load parameters from onnx files
        try:
            model = onnx.load(src_file)
            onnx.checker.check_model(model)
        except:
            raise ValueError(f"Ill-formed networks: {src_file}")

        parameters = defaultdict(layer_func)
        for initializer in model.graph.initializer:
            name = initializer.name
            idx = [int(n) for n in name.split('.') if str.isdigit(n)][0]
            type = name.split('.')[-1]
            if str.endswith(name, "deepz_lambda"):
                continue
            else:
                tensor = onnx.numpy_helper.to_array(initializer)
                parameters[idx][type] = torch.from_numpy(tensor)
        parameters = [[parameters[idx]["weight"], parameters[idx]["bias"]] for idx in sorted(parameters.keys())]
        parameters = list(itertools.chain(*parameters))

        # load parameters from numpy file
        if all(len(sizes) == 2 for sizes in feature_sizes):
            net = FCN(feature_sizes)
        else:
            net = CNN(feature_sizes)
        for (name, layer), param in zip(net.state_dict().items(), parameters):
            assert layer.shape == param.shape, (layer.shape, param.shape)
            avg_param = param.data.mean().item()
            layer.data.copy_(param)
            avg_layer_param = layer.data.mean().item()
            assert avg_layer_param == avg_param

        # evaluate model on testset to manually check their parameters are well transfered from onnx file
        test_model(src_file, net)

        # save network as pt files
        torch.save(net.state_dict(), dst_file)
