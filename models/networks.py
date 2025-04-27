import torch
import torch.nn as nn
from data import DatasetName, DATASETS


def FCN(layer_params: list[dict[str, int]]) -> nn.Sequential:
    """
    Fully connected network with ReLU activations.
    :param args: List of dictionaries containing layer parameters.
    :return: A Sequential model.
    """
    layers = [nn.Flatten(start_dim=1)]
    for idx, kwargs in enumerate(layer_params):
        layers += [nn.Linear(**kwargs)]
        if idx < len(layer_params) - 1:
            layers += [nn.ReLU()]
    return nn.Sequential(*layers)


def CNN(layer_params: list[dict[str, int]]) -> nn.Sequential:
    """
    Convolutional network with ReLU activations.
    :param args: List of dictionaries containing layer parameters.
    :return: A Sequential model.
    """
    layers = []
    conv_kwargs = list(filter(lambda xs: len(xs) > 2, layer_params))
    for idx, kwargs in enumerate(conv_kwargs):
        layers += [nn.Conv2d(**kwargs), nn.ReLU()]
    layers += [nn.Flatten(start_dim=1)]
    fc_kwargs = list(filter(lambda xs: len(xs) == 2, layer_params))
    for idx, kwargs in enumerate(fc_kwargs):
        layers += [nn.Linear(**kwargs)]
        if idx < len(fc_kwargs) - 1:
            layers += [nn.ReLU()]
    return nn.Sequential(*layers)


def FLN(layer_params: list[dict[str, int]]) -> nn.Sequential:
    """
    Fully linear network without activations. Used for testing purposes.
    :param args: List of dictionaries containing layer parameters.
    :return: A Sequential model.
    """
    layers = [nn.Flatten()]
    for idx, kwargs in enumerate(layer_params):
        layers += [nn.Linear(**kwargs)]
    return nn.Sequential(*layers)


def get_affine_normalization_1c(mean: float, sigma: float) -> nn.Module:
    normalization_layer = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
    normalization_layer.weight.data.fill_(1.0 / sigma)
    normalization_layer.bias.data.fill_(-mean / sigma)
    return normalization_layer


def get_affine_normalization_3c(mean: tuple[float, float, float], sigma: tuple[float, float, float]) -> nn.Module:
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


get_affine_normalization = {
    DatasetName.MNIST: get_affine_normalization_1c(
        mean=DATASETS[DatasetName.MNIST].mean[0],
        sigma=DATASETS[DatasetName.MNIST].std[0]),
    DatasetName.CIFAR10: get_affine_normalization_3c(
        mean=DATASETS[DatasetName.CIFAR10].mean,
        sigma=DATASETS[DatasetName.CIFAR10].std)
}


def make_marginal_tail(true_label: int, target_label: int, base_out_dim: int=10) -> nn.Sequential:
    tail = nn.Linear(base_out_dim, 1, bias=False)
    tail.weight.requires_grad = False
    tail.weight.data = torch.zeros(1, base_out_dim)
    tail.weight.data[0, true_label] = 1
    tail.weight.data[0, target_label] = -1
    return nn.Sequential(tail)

def make_marginals_tail(true_label: int, base_out_dim: int=10) -> nn.Sequential:
    tail = nn.Linear(base_out_dim, 1, bias=False)
    tail.weight.requires_grad = False
    tail.weight.data = -torch.eye(base_out_dim)
    tail.weight.data[:, true_label] += 1
    return nn.Sequential(tail)
