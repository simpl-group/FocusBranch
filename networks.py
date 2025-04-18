import torch
import torch.nn as nn
import os

# fcn_spec = {
#     'fc1': [50, 10],
#     'fc2': [100, 50, 10],
#     'fc3': [100, 100, 10],
#     'fc4': [100, 100, 50, 10],
#     'fc5': [100, 100, 100, 10],
#     'fc6': [100, 100, 100, 100, 10],
#     'fc7': [100, 100, 100, 100, 100, 10]
# }
# cnn_spec = {
#     'conv1': ([(16, 3, 2, 1)], [100, 10]),
#     'conv2': ([(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10]),
#     'conv3': ([(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10])
# }

from models import FCN_SPEC, CNN_SPEC


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
            if idx < len(args) - 1:
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


def get_network(name: str, folder: str, device: str = "cpu") -> nn.Sequential:
    if name in FCN_SPEC:
        feature_sizes, pt_path = FCN_SPEC[name]
        model = FCN(feature_sizes)
    elif name in CNN_SPEC:
        feature_sizes, pt_path = CNN_SPEC[name]
        model = CNN(feature_sizes)
    elif name == "fcbase":
        raise NotImplementedError
        model = FLN(28, [50, 50, 50, 10])
    else:
        assert False, f"Invalid network name: {name}"

    model.load_state_dict(torch.load(pt_path, weights_only=True))
    model.eval()

    return model.to(device=device)
