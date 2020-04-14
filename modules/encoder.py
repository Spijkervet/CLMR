import torch
import torch.nn as nn


class WaveEncoder(nn.Module):
    """Function implementing the front-end proposed by Lee et al. 2017.
    Lee, et al. "Sample-level Deep Convolutional Neural Networks for Music
    Auto-tagging Using Raw Waveforms."
    arXiv preprint arXiv:1703.01789 (2017).
    - 'x': input waveform
    - 'is_training': placeholder indicating weather it is training or test
    phase, for dropout or batch norm.
    """

    expansion = 1

    def __init__(self, input_dim, num_classes=256): # TODO num_classes makes no sense now.
        super(WaveEncoder, self).__init__()


        hidden_dims = [64, 64, 64, 128, 128, 128, 256, 512, 512]
        kernel_sizes = [3] * len(hidden_dims)
        strides = [3, 1, 1, 1, 1, 1, 1, 1, 1]
        padding = [1] * len(hidden_dims) # TODO

        assert (
            len(hidden_dims) == len(kernel_sizes) == len(strides) == len(padding)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.seq = nn.Sequential()
        for idx, (h, k, s, p) in enumerate(zip(hidden_dims, kernel_sizes, strides, padding)):
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=h,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    bias=False,
                ),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=k, stride=s, padding=p)
            )

            self.seq.add_module("layer-{}".format(idx), block)
            input_dim = h

        # TODO
        # self.last_conv = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512 * self.expansion, num_classes)


    def forward(self, x):
        out = self.seq(x)

        # out = self.last_conv(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
