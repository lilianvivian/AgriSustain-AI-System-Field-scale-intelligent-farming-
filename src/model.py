import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from torchmetrics.classification import Accuracy


class PlantDiseaseModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        # EfficientNet backbone
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=num_classes
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer