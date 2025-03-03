import torch
import torch.nn as nn
import os

fcn_spec = {
    'fc1': [50, 10],
    'fc2': [100, 50, 10],
    'fc3': [100, 100, 10],
    'fc4': [100, 100, 50, 10],
    'fc5': [100, 100, 100, 10],
    'fc6': [100, 100, 100, 100, 10],
    'fc7': [100, 100, 100, 100, 100, 10]
}
cnn_spec = {
    'conv1': ([(16, 3, 2, 1)], [100, 10]),
    'conv2': ([(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10]),
    'conv3': ([(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10])
}

class Normalization(nn.Module):

    def __init__(self, mean=0.1307, sigma=0.3081):
        super(Normalization, self).__init__()
        self.mean = mean
        self.sigma = sigma

    def forward(self, x):
        return (x - self.mean) / self.sigma


class FCN(nn.Sequential):

    def __init__(self, input_size, fc_layers):
        layers = [Normalization(), nn.Flatten()]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.ReLU()]
            prev_fc_size = fc_size
        super().__init__(*layers)


class CNN(nn.Sequential):

    def __init__(self, input_size, conv_layers, fc_layers):
        layers = [Normalization()]
        prev_channels = 1
        img_dim = input_size

        for n_channels, kernel_size, stride, padding in conv_layers:
            layers += [
                nn.Conv2d(prev_channels, n_channels, kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
            ]
            prev_channels = n_channels
            img_dim = img_dim // stride
        layers += [nn.Flatten()]

        prev_fc_size = prev_channels * img_dim * img_dim
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.ReLU()]
            prev_fc_size = fc_size

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
    if name in fcn_spec:
        model = FCN(28, fcn_spec[name])
    elif name in cnn_spec:
        model = CNN(28, *cnn_spec[name])
    elif name == "fcbase":
        model = FLN(28, [50, 50, 50, 10])
    else:
        assert False, f"Invalid network name: {name}"

    model.load_state_dict(torch.load(os.path.join(folder, f"{name}.pt"), weights_only=True))
    model.eval()

    return model.to(device=device)
