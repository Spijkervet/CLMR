import torch
import torch.nn as nn
import torchvision
from .nt_xent import NT_Xent
from .model import Model
from .sample_cnn_59049 import SampleCNN59049
from .identity import Identity

class SimCLR(Model):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, args, encoder, n_features, projection_dim):
        super(SimCLR, self).__init__()

        self.args = args
        self.encoder = encoder
        self.n_features = n_features
        self.projection_dim = projection_dim

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        # loss function
        if not args.supervised:
            self.criterion = NT_Xent(args.batch_size, args.temperature, args.device)

            self.encoder.fc = (
                Identity()
            )  # remove fully-connected layer after pooling layer for contrastive learning
            
            # construct projection head
            self.projector = [] if args.projector_layers else [Identity()]
            for p in range(args.projector_layers - 1):
                block = [
                    nn.Linear(self.n_features, self.n_features, bias=False),
                    nn.ReLU(),
                ]
                self.projector.extend(block)

            self.projector = nn.Sequential(
                *self.projector,
                nn.Linear(self.n_features, self.projection_dim, bias=False)
            )
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()

    def get_latent_size(self, x):
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

    def forward(self, x_i, x_j):
        h_i, z_i = self.get_latent_representations(x_i)
        h_j, z_j = self.get_latent_representations(x_j)
        loss = self.criterion(z_i, z_j)
        return loss
