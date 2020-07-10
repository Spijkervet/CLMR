import torch
import torch.nn as nn
import torchvision
from .model import Model
from .sample_cnn_59049 import SampleCNN59049
from .cpc.encoder import Encoder as CPCEncoder

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class CLMR(Model):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, args):
        super(CLMR, self).__init__()

        self.args = args

        input_dim = 1
        if args.domain == "audio":
            if args.sample_rate == 22050:
                self.encoder = SampleCNN59049(args)
            print(f"### {self.encoder.__class__.__name__} ###")

        elif args.domain == "scores":
            self.encoder = self.get_resnet(args.resnet)  # resnet
            self.encoder.conv1 = nn.Conv2d(
                args.image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            raise NotImplementedError

        self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        if not args.supervised:
            self.encoder.fc = Identity()  # remove fully-connected layer after pooling layer for contrastive learning

            if args.projector_layers == 0:
                self.projector = Identity()
            elif args.projector_layers == 1:
                self.projector = nn.Linear(self.n_features, args.projection_dim, bias=False)
            else:
                # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
                self.projector = nn.Sequential(
                    nn.Linear(self.n_features, self.n_features, bias=False),
                    nn.ReLU(),
                    nn.Linear(self.n_features, args.projection_dim, bias=False),
                )

    def get_resnet(self, name):
        resnets = {
            "resnet18": torchvision.models.resnet18(),
            "resnet50": torchvision.models.resnet50(),
        }
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid ResNet version")
        return resnets[name]

    def get_latent_size(self, input_size):
        x = torch.zeros(input_size).to(self.args.device)
        h, z = self.get_latent_representations(x)
        return h.size(1)

    def get_latent_representations(self, x):
        h = self.encoder(x)

        if not self.args.supervised:
            z = self.projector(h)

            if self.args.normalize:
                z = nn.functional.normalize(z, dim=1)
        else:
            z = None

        return h, z

    def forward(self, x):
        h, z = self.get_latent_representations(x)
        return h, z
