from torch import nn

class PendulumModel(nn.Module):
    def __init__(
            self,
            input_size: int = 5,
            hidden_size: int = 64,
            output_size: int = 2,
            num_hidden_layers: int = 1
    ):
        super(PendulumModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *[nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)],
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)