import torch.nn as nn
from .model import Model

class MLP(Model):
    
    def __init__(self, n_features, n_classes):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x