from torch import nn

class PendulumModel(nn.Module):
    def __init__(
            self,
            input_size: int = 5,
            hidden_size: int = 64,
            output_size: int = 2,
            num_hidden_layers: int = 2
    ):
        super(PendulumModel, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend((nn.Linear(hidden_size, hidden_size), nn.ReLU()))
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)