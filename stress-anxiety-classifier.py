"""
Stress & Anxiety Facial‑Expression Classifier (Lightning v2.x)
──────────────────────────────────────────────────────────────
* Fine‑tunes ResNet‑50 on four stress‑level classes.
* **Automatically logs:**
  – train & validation loss/metrics (TensorBoard scalars)
  – a **confusion‑matrix heat‑map** at the end of each validation epoch
  – a **loss‑curve figure** (train vs val) once training finishes.

All plots are pushed to the default Lightning logger (TensorBoard). Open with:

The command to run this script in powershell is:
python stress-anxiety-classifier.py fit --data.data_root=output-stress-split
"""
from pathlib import Path
from typing import Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from torchmetrics import Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef, ConfusionMatrix

import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────── Globals
CLASS_MAPPING: Dict[int, str] = {0: "relaxed", 1: "mild_stress", 2: "anxious", 3: "ptsd"}
NUM_CLASSES = len(CLASS_MAPPING)
IMAGE_SIZE  = 224

# ─────────────────────────────── Transforms
def build_transforms(train: bool):
    basic = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomRotation(12),
            *basic,
        ])
    return transforms.Compose(basic)

# ─────────────────────────────── DataModule
class StressDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        self.ds_train = datasets.ImageFolder(self.data_root / "train", transform=build_transforms(True))
        self.ds_val   = datasets.ImageFolder(self.data_root / "test",  transform=build_transforms(False))

    def train_dataloader(self):
        return DataLoader(self.ds_train, self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

# ─────────────────────────────── LightningModule
class StressClassifier(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, freeze_backbone: bool = False):
        super().__init__()
        self.save_hyperparameters()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
        backbone.fc = nn.Linear(backbone.fc.in_features, NUM_CLASSES)
        self.model = backbone
        self.loss_fn = nn.CrossEntropyLoss()

        self._y_true: List[int] = []
        self._y_pred: List[int] = []
        self.train_loss_epoch: List[float] = []
        self.val_loss_epoch:   List[float] = []

        self.val_acc  = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.val_prec = Precision(task="multiclass", num_classes=NUM_CLASSES, average="weighted")
        self.val_rec  = Recall(task="multiclass", num_classes=NUM_CLASSES, average="weighted")
        self.val_f1   = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="weighted")
        self.val_mcc  = MatthewsCorrCoef(task="multiclass", num_classes=NUM_CLASSES)
        self.cm_metric= ConfusionMatrix(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        tl = self.trainer.callback_metrics.get("train_loss")
        if tl is not None:
            self.train_loss_epoch.append(tl.item())

    def on_validation_epoch_start(self):
        for m in (self.val_acc, self.val_prec, self.val_rec, self.val_f1, self.val_mcc):
            m.reset()
        self._y_true.clear()
        self._y_pred.clear()
        self.cm_metric.reset()

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, 1)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.val_acc(preds, y); self.val_prec(preds, y); self.val_rec(preds, y)
        self.val_f1(preds, y);  self.val_mcc(preds, y)
        self.cm_metric.update(preds, y)

        self._y_true.extend(y.tolist())
        self._y_pred.extend(preds.tolist())

    def on_validation_epoch_end(self):
        self.log_dict({
            "val_acc": self.val_acc.compute(),
            "val_precision": self.val_prec.compute(),
            "val_recall":    self.val_rec.compute(),
            "val_f1":        self.val_f1.compute(),
            "val_mcc":       self.val_mcc.compute(),
        }, prog_bar=True)

        self.val_loss_epoch.append(self.trainer.callback_metrics["val_loss"].item())

        cm = self.cm_metric.compute().cpu().numpy()
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=list(CLASS_MAPPING.values()),
                    yticklabels=list(CLASS_MAPPING.values()), ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        self.logger.experiment.add_figure("confusion_matrix", fig, self.current_epoch)
        plt.close(fig)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return {"optimizer": opt, "lr_scheduler": sch}

    def on_fit_end(self):
        if not self.train_loss_epoch:
            return
        epochs = range(len(self.train_loss_epoch))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(epochs, self.train_loss_epoch, label="Train Loss", marker="o")
        ax.plot(epochs, self.val_loss_epoch,   label="Val Loss",   marker="s")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
        ax.set_title("Loss curves"); ax.legend(); ax.grid(True); fig.tight_layout()
        self.logger.experiment.add_figure("loss_curves", fig, global_step=self.current_epoch)
        plt.close(fig)

# ─────────────────────────────── CLI Entrypoint
if __name__ == "__main__":
    LightningCLI(StressClassifier, StressDataModule,
                 seed_everything_default=42,
                 save_config_callback=None)
