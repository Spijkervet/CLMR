import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning import metrics


class SupervisedLearning(LightningModule):
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
