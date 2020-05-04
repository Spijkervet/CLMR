import torch
from .encoder import Encoder
from .autoregressor import Autoregressor
from .infonce import InfoNCE
# from testing.loss.genre_loss import GenreLoss

class CPC(torch.nn.Module):
    def __init__(
        self, args, strides, filter_sizes, padding, genc_input, genc_hidden, gar_hidden,
    ):
        super(CPC, self).__init__()

        self.args = args

        """
        First, a non-linear encoder genc maps the input sequence of observations xt to a
        sequence of latent representations zt = genc(xt), potentially with a lower temporal resolution.
        """
        # self.encoder = SampleCNN20480()
        # self.encoder = SampleCNN42848()

        # if args.batch_norm:
        #     self.encoder = SampleCNN59049BatchNorm(args)
        # else:
        #     self.encoder = SampleCNN59049(args)
        self.encoder = Encoder(genc_input, genc_hidden, strides, filter_sizes, padding,)

        """
        We then use a GRU RNN [17] for the autoregressive part of the model, gar with 256 dimensional hidden state.
        """
        self.autoregressor = Autoregressor(
            args, input_dim=genc_hidden, hidden_dim=gar_hidden
        )

        if args.loss == 0:
            print("### InfoNCE Loss ###")
            self.loss = InfoNCE(args, gar_hidden, genc_hidden)
        elif args.loss == 1:
            print("### Supervised Loss ###")
            self.loss = GenreLoss(args, gar_hidden, args.n_classes)
        else:
            raise NotImplementedError

    def get_latent_size(self, input_size):
        x = torch.zeros(input_size).to(self.args.device)

        z, c = self.get_latent_representations(x)
        return c.size(2) * c.size(1) # c.size(2), c.size(1)

    def get_latent_representations(self, x):
        """
        Calculate latent representation of the input with the encoder and autoregressor
        :param x: inputs (B x C x L)
        :return: loss - calculated loss
                accuracy - calculated accuracy
                z - latent representation from the encoder (B x L x C)
                c - latent representation of the autoregressor  (B x C x L)
        """

        # calculate latent represention from the encoder
        z = self.encoder(x)
        z = z.permute(0, 2, 1)  # swap L and C

        # calculate latent representation from the autoregressor
        c = self.autoregressor(z)
        return z, c

    def forward(self, x, y):
        z, c = self.get_latent_representations(x)
        loss, output = self.loss.get(x, z, c, y)
        return loss, output, z, c
