import torch
import torch.nn as nn
import os
from models import FCN_SPEC, CNN_SPEC, MODEL_DIR
from data import MNIST_MEAN, MNIST_STD, CIFAR10_MEAN, CIFAR10_STD


class Normalization(nn.Module):

    def __init__(self, mean=0.1307, sigma=0.3081):
        super(Normalization, self).__init__()
        self.mean = mean
        self.sigma = sigma

    def forward(self, x):
        return (x - self.mean) / self.sigma


class FCN(nn.Sequential):

    def __init__(self, args):
        layers = [nn.Flatten(start_dim=1)]
        for idx, kwargs in enumerate(args):
            layers += [nn.Linear(**kwargs)]
            if idx < len(args) - 1:
                layers += [nn.ReLU()]
        super().__init__(*layers)


class CNN(nn.Sequential):

    def __init__(self, args):
        layers = []
        conv_kwargs = list(filter(lambda xs: len(xs) > 2, args))
        for idx, kwargs in enumerate(conv_kwargs):
            layers += [nn.Conv2d(**kwargs), nn.ReLU(), ]
        layers += [nn.Flatten(start_dim=1)]
        fc_kwargs = list(filter(lambda xs: len(xs) == 2, args))
        for idx, kwargs in enumerate(fc_kwargs):
            layers += [nn.Linear(**kwargs)]
            if idx < len(fc_kwargs) - 1:
                layers += [nn.ReLU()]
        super().__init__(*layers)


class FLN(nn.Sequential):  # Fully linear network
    def __init__(self, input_size, hdims):
        layers = [nn.Flatten()]
        prev_fc_size = input_size * input_size
        for d in hdims:
            layers.append(nn.Linear(prev_fc_size, d))
            prev_fc_size = d
        super().__init__(*layers)


def get_affine_normalization_mnist(mean: float=MNIST_MEAN, sigma: float=MNIST_STD) -> nn.Module:
    normalization_layer = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
    normalization_layer.weight.data.fill_(1.0 / sigma)
    normalization_layer.bias.data.fill_(-mean / sigma)
    return normalization_layer


def get_affine_normalization_cifar10(mean: tuple[float, float, float]=CIFAR10_MEAN, sigma: tuple[float, float, float]=CIFAR10_STD) -> nn.Module:
    normalization_layer = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
    normalization_layer.weight.data.fill_(0.)
    normalization_layer.bias.data.fill_(0.)
    normalization_layer.weight.data[0, 0, 0, 0] = 1.0 / sigma[0]
    normalization_layer.weight.data[1, 1, 0, 0] = 1.0 / sigma[1]
    normalization_layer.weight.data[2, 2, 0, 0] = 1.0 / sigma[2]
    normalization_layer.bias.data[0] = -mean[0] / sigma[0]
    normalization_layer.bias.data[1] = -mean[1] / sigma[1]
    normalization_layer.bias.data[2] = -mean[2] / sigma[2]
    return normalization_layer


def get_network(name: str, folder: str=MODEL_DIR, device: str="cpu", add_affine_normalization: bool=True) -> nn.Sequential:
    if name in FCN_SPEC:
        layer_params, model_path, dataset, need_normalization = FCN_SPEC[name]
        model = FCN(layer_params)
    elif name in CNN_SPEC:
        layer_params, model_path, dataset, need_normalization = CNN_SPEC[name]
        model = CNN(layer_params)
    elif name == "fcbase":
        model = FLN(28, [50, 50, 50, 10])
        model_path = str(MODEL_DIR/"fcbase")
        dataset = "MNIST"
        need_normalization = False
    else:
        assert False, f"Invalid network name: {name}"

    if need_normalization and add_affine_normalization:
        if dataset == "MNIST":
            normalization_layer = get_affine_normalization_mnist()
        elif dataset == "CIFAR10":
            normalization_layer = get_affine_normalization_cifar10()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        model.insert(0, normalization_layer)
    model.load_state_dict(torch.load(model_path + ".pt", weights_only=True))
    model.eval()

    return model.to(device=device)
