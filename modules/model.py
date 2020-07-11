import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def initialize(self, m):
        if isinstance(m, (nn.Conv1d)):
            # nn.init.xavier_uniform_(m.weight)
            # if m.bias is not None:
            #     nn.init.xavier_uniform_(m.bias)

            nn.init.kaiming_uniform_(
                m.weight, mode="fan_in", nonlinearity="relu"
            )

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return f"{self}\n\nNumber of trainable parameters: {params}"
