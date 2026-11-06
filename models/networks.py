import torch
import torch.nn as nn


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


def make_marginals_tail(true_label: int, base_out_dim: int=10) -> nn.Sequential:
    tail = nn.Linear(base_out_dim, 1, bias=False)
    tail.weight.requires_grad = False
    tail.weight.data = -torch.eye(base_out_dim)
    tail.weight.data[:, true_label] += 1
    return nn.Sequential(tail)
