"""
project/experiment.py

PyTorch Lightning LightningModule wrapping the model + training logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl

from project.configs.config import TrainingConfig
from project.models.cnn import UltMel2DCNN


class Ult2MelExperiment(pl.LightningModule):
    def __init__(self, model: UltMel2DCNN, cfg: TrainingConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loss_fn = nn.MSELoss()
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        ult, mel = batch
        pred = self(ult)
        mse = self.loss_fn(pred, mel)
        loss = mse
        if stage == "train":
            loss = mse + self.model.l1_penalty()
        self.log(f"{stage}_mse", mse, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.learning_rate,
            momentum=self.cfg.momentum,
            nesterov=self.cfg.nesterov,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.cfg.lr_reduce_factor,
            patience=self.cfg.lr_reduce_patience,
            min_lr=self.cfg.lr_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mse",
                "frequency": 1,
                "interval": "epoch",
            },
        }
