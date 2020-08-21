import torch.nn as nn

class FineTuner(nn.Module):

    def __init__(self, n_features, n_classes, mlp):
        super(FineTuner, self).__init__()
        if not mlp:
            self.model = nn.Sequential(nn.Linear(n_features, n_classes),)
        else:
            self.model = nn.Sequential(
                nn.Linear(n_features, n_features),
                nn.ReLU(),
                nn.Linear(n_features, n_classes)
            )

    def forward(self, x):
        return self.model(x)