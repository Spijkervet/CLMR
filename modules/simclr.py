import torch
import torch.nn as nn
import torchvision
from .encoder import WaveEncoder
from .sample_cnn_16000 import SampleCNN16000
from .sample_cnn_59049 import SampleCNN59049
from .sample_cnn_8000 import SampleCNN8000


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, args):
        super(SimCLR, self).__init__()

        self.args = args

        input_dim = 1
        # self.encoder = WaveEncoder(input_dim)
        # self.encoder = SampleCNN42848()
        if args.domain == "audio":
            if args.sample_rate == 16000:
                self.encoder = SampleCNN16000(args)
            elif args.sample_rate == 22050:
                self.encoder = SampleCNN59049(args)
            elif args.sample_rate == 8000:
                self.encoder = SampleCNN8000(args)
            print(f"### {self.encoder.__class__.__name__} ###")

        elif args.domain == "scores":
            self.encoder = self.get_resnet(args.resnet)  # resnet
            self.encoder.conv1 = nn.Conv2d(
                args.image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            raise NotImplementedError

        self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer
        self.encoder.fc = Identity()  # remove fully-connected layer after pooling layer

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

        if self.args.fp16:
            x = x.half()

        h, z = self.get_latent_representations(x)
        return z.size(1)

    def get_latent_representations(self, x):
        h = self.encoder(x)
        z = self.projector(h)

        if self.args.normalize:
            z = nn.functional.normalize(z, dim=1)

        return h, z

    def forward(self, x):
        h, z = self.get_latent_representations(x)
        return h, z
