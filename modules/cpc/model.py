import torch
from modules import Model
from .cpc import CPC


class CPCModel(Model):
    def __init__(
        self, args, strides, filter_sizes, padding, genc_hidden, gar_hidden,
    ):
        super(CPCModel, self).__init__()

        self.args = args
        self.strides = strides
        self.filter_sizes = filter_sizes
        self.padding = padding
        self.genc_input = 1
        self.genc_hidden = genc_hidden
        self.gar_hidden = gar_hidden
        self.n_features = self.gar_hidden

        self.model = CPC(
            args,
            strides,
            filter_sizes,
            padding,
            self.genc_input,
            genc_hidden,
            gar_hidden,
        )

    def forward(self, x, y):
        """Forward through the network"""

        loss, output, z, c = self.model(x, y)
        return loss, output, z, c
