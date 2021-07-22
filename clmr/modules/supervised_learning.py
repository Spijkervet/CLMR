import torch
import torchmetrics
import torch.nn as nn
from pytorch_lightning import LightningModule


class SupervisedLearning(LightningModule):
    def __init__(self, args, encoder, output_dim):
        super().__init__()
        self.save_hyperparameters(args)
        self.encoder = encoder

        self.encoder.fc.out_features = output_dim
        self.output_dim = output_dim
        self.model = self.encoder
        self.criterion = self.configure_criterion()

        self.average_precision = torchmetrics.AveragePrecision(pos_label=1)

    def forward(self, x, y):
        x = x[:, 0, :]  # we only have 1 sample, no augmentations
        preds = self.model(x)
        loss = self.criterion(preds, y)
        return loss, preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, preds = self.forward(x, y)
        self.log("Train/pr_auc", self.average_precision(preds, y))
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, preds = self.forward(x, y)
        self.log("Valid/pr_auc", self.average_precision(preds, y))
        self.log("Valid/loss", loss)
        return loss

    def configure_criterion(self):
        if self.hparams.dataset in ["magnatagatune"]:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion

    def configure_optimizers(self):
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
