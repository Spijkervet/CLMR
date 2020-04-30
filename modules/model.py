import torch.nn as nn
import numpy as np

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return f"{self}\n\nNumber of trainable parameters: {params}"