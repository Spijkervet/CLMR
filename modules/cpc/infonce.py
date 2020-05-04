"""
InfoNCE
Calculates the 'Info Noise-Contrastive-Estimation' as explained by Van den Oord et al. (2018),
implementation by Bas Veeling & Sindy Lowe
"""

import torch
import torch.nn as nn
import numpy as np

class InfoNCE(nn.Module):
    def __init__(self, args, gar_hidden, genc_hidden):
        super(InfoNCE, self).__init__()

        self.args = args
        self.gar_hidden = gar_hidden
        self.genc_hidden = genc_hidden
        self.negative_samples = self.args.negative_samples

        # predict |prediction_step| timesteps into the future
        self.predictor = nn.Linear(
            gar_hidden, genc_hidden * self.args.prediction_step, bias=False
        )

        if self.args.subsample:
            self.subsample_win = 81 # TODO

        self.loss = nn.LogSoftmax(dim=1)

    def get(self, x, z, c, y=None):
        full_z = z

        if self.args.subsample:
            """ 
            positive samples are restricted to this subwindow to reduce the number of calculations for the loss, 
            negative samples can still come from any point of the input sequence (full_z)
            """
            if c.size(1) > self.subsample_win:
                seq_begin = np.random.randint(0, c.size(1) - self.subsample_win)
                c = c[:, seq_begin : seq_begin + self.subsample_win, :]
                z = z[:, seq_begin : seq_begin + self.subsample_win, :]

        Wc = self.predictor(c)
        return self.infonce_loss(Wc, z, full_z)

    def broadcast_batch_length(self, input_tensor):
        """
        broadcasts the given tensor in a consistent way, such that it can be applied to different inputs and
        keep their indexing compatible
        :param input_tensor: tensor to be broadcasted, generally of shape B x L x C
        :return: reshaped tensor of shape (B*L) x C
        """

        assert input_tensor.size(0)
        assert len(input_tensor.size()) == 3

        return input_tensor.reshape(-1, input_tensor.size(2))

    def get_pos_sample_f(self, Wc_k, z_k):
        """
        calculate the output of the log-bilinear model for the positive samples, i.e. where z_k is the actual
        encoded future that had to be predicted
        :param Wc_k: prediction of the network for the encoded future at time-step t+k (dimensions: (B*L) x C)
        :param z_k: encoded future at time-step t+k (dimensions: (B*L) x C)
        :return: f_k, output of the log-bilinear model (without exp, as this is part of the log-softmax function)
        """
        Wc_k = Wc_k.unsqueeze(1)
        z_k = z_k.unsqueeze(2)
        f_k = torch.squeeze(torch.matmul(Wc_k, z_k), 1)
        return f_k

    def get_neg_z(self, z):
        """
        scramble z to retrieve negative samples, i.e. z values that should not be predicted by the model
        :param z: unshuffled z as output by the model
        :return: z_neg - shuffled z to be used for negative sampling
                shuffling params rand_neg_idx, rand_offset for testing this function
        """

        """ randomly selecting from all z values; 
            can cause positive samples to be selected as negative samples as well 
            (but probability is <0.1% in our experiments)
            done once for all time-steps, much faster                
        """
        z = self.broadcast_batch_length(z)
        z_neg = torch.stack(
            [
                torch.index_select(z, 0, torch.randperm(z.size(0)).to(z.get_device()))
                for i in range(self.negative_samples)
            ],
            2,
        )
        rand_neg_idx = None
        rand_offset = None
        return z_neg, rand_neg_idx, rand_offset

    def get_neg_samples_f(self, Wc_k, z_k, z_neg=None, k=None):
        """
        calculate the output of the log-bilinear model for the negative samples. For this, we get z_k_neg from z_k
        by randomly shuffling the indices.
        :param Wc_k: prediction of the network for the encoded future at time-step t+k (dimensions: (B*L) x C)
        :param z_k: encoded future at time-step t+k (dimensions: (B*L) x C)
        :return: f_k, output of the log-bilinear model (without exp, as this is part of the log-softmax function)
        """
        Wc_k = Wc_k.unsqueeze(1)

        """
            by shortening z_neg from the front, we get different negative samples
            for every prediction-step without having to re-sample;
            this might cause some correlation between the losses within a batch
            (e.g. negative samples for projecting from z_t to z_(t+k+1) 
            and from z_(t+1) to z_(t+k) are the same)                
        """

        z_k_neg = z_neg[z_neg.size(0) - Wc_k.size(0) :, :, :]

        f_k = torch.squeeze(torch.matmul(Wc_k, z_k_neg), 1)

        return f_k

    def infonce_loss(self, Wc, z, full_z):
        """
        calculate the loss based on the model outputs Wc (the prediction) and z (the encoded future)
        :param Wc: output of the predictor, where W are the weights for the different timesteps and
        c the latent representation (B, L, C * self.args.prediction_step)
        :param z: encoded future - output of the encoder (B, L, C)
        :return: loss - average loss over all samples, timesteps and prediction steps in the batch
                accuracy - average accuracies over all samples, timesteps and predictions steps in the batch
        """
        seq_len = z.size(1)

        cur_device = Wc.get_device()

        total_loss = 0
        accuracies = torch.zeros(self.args.prediction_step, 1).to(cur_device)
        true_labels = torch.zeros((seq_len * self.args.batch_size,)).long().to(cur_device)

        # Which type of method to use for negative sampling:
        # 0 - inside the loop for the prediction time-steps. Slow, but samples from all but the current pos sample
        # 1 - outside the loop for prediction time-steps
        #   Low probability (<0.1%) of sampling the positive sample as well.
        # 2 - outside the loop for prediction time-steps. Sampling only within the current sequence
        #   Low probability of sampling the positive sample as well.

        # sampling method 1 / 2
        z_neg, _, _ = self.get_neg_z(full_z)
        for k in range(1, self.args.prediction_step + 1):
            z_k = z[:, k:, :]
            Wc_k = Wc[:, :-k, (k - 1) * self.genc_hidden : k * self.genc_hidden]

            z_k = self.broadcast_batch_length(z_k)
            Wc_k = self.broadcast_batch_length(Wc_k)

            pos_samples = self.get_pos_sample_f(Wc_k, z_k)

            neg_samples = self.get_neg_samples_f(Wc_k, z_k, z_neg, k)

            # concatenate positive and negative samples
            results = torch.cat((pos_samples, neg_samples), 1)
            loss = self.loss(results)[:, 0]

            total_samples = (seq_len - k) * self.args.batch_size
            loss = -loss.sum() / total_samples
            total_loss += loss


        total_loss /= self.args.prediction_step
        accuracies = torch.mean(accuracies)

        # total_loss = total_loss.unsqueeze(0)
        return total_loss, None #accuracies

