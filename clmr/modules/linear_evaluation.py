import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning import metrics


class LinearEvaluation(LightningModule):
    def __init__(self, args, encoder, hidden_dim, output_dim):
        super().__init__()

        self.hparams = args
        # self.save_hyperparameters()

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if self.hparams.finetuner_mlp:
            self.model = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.output_dim))
        else:
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
        x, y = batch
        loss, preds = self.forward(x, y)
        # self.log("Test/accuracy", self.accuracy(preds, y))
        self.log("Test/pr_auc", self.average_precision(preds, y))
        self.log("Test/loss", loss)
        return loss

    def configure_criterion(self):
        if self.hparams.dataset in ["magnatagatune", "msd"]:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion

    def configure_optimizers(self):
        scheduler = None
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.finetuner_learning_rate, weight_decay=self.hparams.weight_decay)
        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}
