import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in: int, m_out: int, hidden_nodes: int, num_layers: int):
        super().__init__()
        layers = []
        in_dim = d_in
        for _ in range(num_layers - 1):      # hidden layers
            layers.append(nn.Linear(in_dim, hidden_nodes))
            layers.append(nn.ReLU())
            in_dim = hidden_nodes
        # final layer to m_out logits
        layers.append(nn.Linear(in_dim, m_out))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
