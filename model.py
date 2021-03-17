import torch
import torch.nn as nn
import torchaudio
from pytorch_lightning import LightningModule
from pytorch_lightning import metrics
from collections import defaultdict

from simclr import SimCLR
from simclr.modules import NT_Xent

from sklearn.metrics import roc_auc_score, average_precision_score


class ContrastiveLearning(LightningModule):
    def __init__(self, args, encoder):
        super().__init__()
        self.hparams = args
        self.save_hyperparameters(self.hparams)

        # initialize ResNet
        self.encoder = encoder
        self.n_features = (
            self.encoder.fc.in_features
        )  # get dimensions of last fully-connected layer
        self.model = SimCLR(self.encoder, self.hparams.projection_dim, self.n_features)
        self.criterion = self.configure_criterion()

    def forward(self, x_i, x_j):
        h_i, h_j, z_i, z_j = self.model(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_i = x[:, 0, :].unsqueeze(dim=1)
        x_j = x[:, 1, :].unsqueeze(dim=1)
        loss = self.forward(x_i, x_j)
        self.log("Train/loss", loss)
        return loss

    def configure_criterion(self):
        # PT lightning aggregates differently in DP mode
        if self.hparams.accelerator == "dp" and self.hparams.gpus:
            batch_size = int(self.hparams.batch_size / self.hparams.gpus)
        else:
            batch_size = self.hparams.batch_size

        criterion = NT_Xent(batch_size, self.hparams.temperature, world_size=1)
        return criterion

    def configure_optimizers(self):
        scheduler = None
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        elif self.hparams.optimizer == "LARS":
            # optimized using LARS with linear learning rate scaling
            # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
            learning_rate = 0.3 * args.batch_size / 256
            optimizer = LARS(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=args.weight_decay,
                exclude_from_weight_decay=["batch_normalization", "bias"],
            )

            # "decay the learning rate with the cosine decay schedule without restarts"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.epochs, eta_min=0, last_epoch=-1
            )
        else:
            raise NotImplementedError

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}


class LinearEvaluation(LightningModule):
    def __init__(self, args, encoder, hidden_dim, output_dim):
        super().__init__()

        self.hparams = args
        # self.save_hyperparameters()

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))
        self.criterion = self.configure_criterion()

        self.accuracy = metrics.Accuracy()
        self.average_precision = metrics.AveragePrecision(pos_label=1)

    def forward(self, x, y):
        with torch.no_grad():
            h0 = self.encoder(x)

        preds = self.model(h0)
        loss = self.criterion(preds, y)
        return loss, preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, preds = self.forward(x, y)

        # self.log("Train/accuracy", self.accuracy(preds, y))
        self.log("Train/pr_auc", self.average_precision(preds, y))
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        x, y = batch
        loss, preds = self.forward(x, y)
        # self.log("Test/accuracy", self.accuracy(preds, y))
        self.log("Test/pr_auc", self.average_precision(preds, y))
        self.log("Test/loss", loss)
        self.model.train()
        return loss

    def configure_criterion(self):
        if self.hparams.dataset in ["magnatagatune", "msd"]:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion

    def configure_optimizers(self):
        scheduler = None
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}


class SupervisedBaseline(LightningModule):
    def __init__(self, args, encoder, output_dim):
        super().__init__()
        self.hparams = args
        # self.save_hyperparameters()
        self.encoder = encoder

        self.encoder.fc.out_features = output_dim
        self.output_dim = output_dim
        self.model = self.encoder
        self.criterion = self.configure_criterion()

        self.accuracy = metrics.Accuracy()
        self.roc = metrics.ROC(pos_label=1)
        self.average_precision = metrics.AveragePrecision(pos_label=1)

    def forward(self, x, y):
        preds = self.model(x)
        loss = self.criterion(preds, y)
        return loss, preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, preds = self.forward(x, y)

        # roc_auc, _, _ = self.roc(y, preds)

        self.log("Train/accuracy", self.accuracy(preds, y))
        # self.log("Train/roc_auc", roc_auc)
        # self.log("Train/pr_auc", self.average_precision(preds, y))
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        x, y = batch
        loss, preds = self.forward(x, y)
        self.log("Valid/accuracy", self.accuracy(preds, y))
        # self.log("Valid/pr_auc", self.average_precision(preds, y))
        self.log("Valid/loss", loss)
        self.model.train()
        return loss

    def configure_criterion(self):
        if self.hparams.dataset in ["magnatagatune"]:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion

    def configure_optimizers(self):
        scheduler = None
        if self.hparams.optimizer == "Adam":
            # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=1e-6,
                nesterov=True,
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.2, patience=5, verbose=True
            )

        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "Valid/loss",
            }
        else:
            return {"optimizer": optimizer}
