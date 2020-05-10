import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder module, consiting of Convolutional blocks with ReLU activation layers
    Add a convolutional block for each stride / filter (kernel) / padding.
    """

    # for 22050 Hz
    def __init__(self, input_dim, hidden_dim, strides, filter_sizes, padding):
        super(Encoder, self).__init__()

        assert (
            len(strides) == len(filter_sizes) == len(padding)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.seq = nn.Sequential()
        for idx, (s, f, p) in enumerate(zip(strides, filter_sizes, padding)):
            block = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, f, stride=s, padding=p), nn.ReLU()
            )
            self.seq.add_module("layer-{}".format(idx), block)
            input_dim = hidden_dim

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 50)
        

    def forward(self, x):
        out = self.seq(x)


        # CLMR!!!
        # out = self.avgpool(out)
        # out = out.reshape(x.shape[0], out.size(1) * out.size(2))
        return out
