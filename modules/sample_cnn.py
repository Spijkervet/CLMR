import torch
import torch.nn as nn
from .model import Model

class SampleCNN(Model):

    def __init__(self, args, strides):
        super(SampleCNN, self).__init__()

        self.supervised = args.supervised
        self.strides = strides
        self.sequential = [nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )]
        
        # CLMR
        self.hidden = [
            [128, 128],
            [128, 128],
            [128, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 512],
        ]
        
        # CLMR-large
        # self.hidden = [
        #     [128, 128],
        #     [128, 128],
        #     [128, 256],
        #     [256, 256],
        #     [256, 256],
        #     [256, 512],
        #     [512, 512],
        #     [512, 512],
        #     [512, 1024],
        # ]

        assert len(self.hidden) == len(self.strides), "Number of hidden layers and strides are not equal"
        for stride, (h_in, h_out) in zip(self.strides, self.hidden):
            self.sequential.append(
                nn.Sequential(
                    nn.Conv1d(h_in, h_out, kernel_size=stride, stride=1, padding=1),
                    nn.BatchNorm1d(h_out),
                    nn.ReLU(),
                    nn.MaxPool1d(stride, stride=stride)
                )
            )

        # 1 x 512
        self.sequential.append(
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            )
        )

        # CLMR-large
        # self.sequential.append(
        #     nn.Sequential(
        #         nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding=1),
        #         nn.BatchNorm1d(1024),
        #         nn.ReLU(),
        #     )
        # )

        self.sequential = nn.Sequential(*self.sequential)

        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        if self.supervised:
            self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(512, 50)

    def forward(self, x):
        # input x : B x 59049 x 1
        out = self.sequential(x)

            
        # out = self.avgpool(out)
        if self.supervised:
            out = self.dropout(out)

        out = out.reshape(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)
        return logit
