"""
main.py — unified pipeline entry point.

Steps
-----
1. Load config.yml
2. Build HDF5 file for each speaker (skips if already exists)
3. Build train/val DataLoaders from HDF5
4. Train the model with PyTorch Lightning
5. Save the mel scaler alongside the checkpoint
6. Test the best checkpoint on the test set, using the saved scaler

"""

from __future__ import annotations

import datetime
import logging
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


from project.configs.config import load_config
from project.datasets.dataset import build_dataloaders
from project.experiment import Ult2MelExperiment
from project.models import MODEL_REGISTRY
from project.preprocessing.synced_h5 import build_h5, h5_exists

torch.set_float32_matmul_precision("high")

# Suppress the LitLogger tip — we're using CSVLogger and MLFlowLogger intentionally
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# Store MLflow runs in a 'mlruns' folder next to main.py,
# regardless of the working directory when the script is invoked
tracking_uri=f"file:{Path(__file__).parent / 'mlruns'}"

console = Console()

def print_config_summary(cfg) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[dim]speakers[/dim]", ", ".join(cfg.data.speakers))
    table.add_row("[dim]data_dir[/dim]", cfg.data.data_dir)
    table.add_row("[dim]h5_dir[/dim]", cfg.data.h5_dir)
    table.add_row("[dim]test ID range[/dim]", f"[{cfg.data.test_suffix_range[0]}, {cfg.data.test_suffix_range[1]})")
    table.add_row("[dim]model[/dim]", "UltMel2DCNN")
    table.add_row("[dim]epochs[/dim]", str(cfg.training.epochs))
    table.add_row("[dim]batch_size[/dim]", str(cfg.training.batch_size))
    table.add_row("[dim]model saved to:[/dim]", str(cfg.training.output_dir))
    console.print(Panel(table, title="[bold]ult-to-mel[/bold]", border_style="cyan"))


def main(hparams) -> None:
    cfg = load_config(hparams.config)
    print_config_summary(cfg)

    for speaker in cfg.data.speakers:
        console.rule(f"[bold cyan]Speaker: {speaker}[/bold cyan]")
        
        # Temporarily override speakers to just this one
        cfg.data.speakers = [speaker]

        if hparams.force_preprocess:
            h5 = cfg.data.h5_path / f"{speaker}.h5"
            if h5.exists():
                h5.unlink()
                console.print(f"  deleted {h5}")

        if not h5_exists(cfg.data, speaker):
            build_h5(cfg.data)
        else:
            console.print(f"[green] H5 present for {speaker} - skipping preprocessing [/green]")

        console.rule("[bold]Building DataLoaders[/bold]")
        train_loader, val_loader, test_loader, scaler = build_dataloaders(cfg.data, cfg.training)
        console.print(f"  train batches: {len(train_loader):,}  |  val batches: {len(val_loader):,}")

        console.rule("[bold]Building 2DCNN model[/bold]")
        ModelClass = MODEL_REGISTRY["UltMel2DCNN"]
        model = ModelClass(cfg.model, cfg.data)
        experiment = Ult2MelExperiment(model, cfg.training)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"UltMel2DCNN_{speaker}_{timestamp}"

        os.makedirs(cfg.training.output_dir, exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_mse",
                min_delta=cfg.training.early_stop_min_delta,
                patience=cfg.training.early_stop_patience,
                mode="min",
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath=cfg.training.output_dir,
                filename=run_name + "_{epoch:02d}_{val_mse:.4f}",
                monitor="val_mse",
                save_top_k=1,
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            RichProgressBar(leave=True),
        ]

        csv_logger = CSVLogger(save_dir=cfg.training.output_dir, name=run_name)
        mlflow_logger = MLFlowLogger(
            experiment_name="ult2mel-2DCNN",
            run_name=run_name,
            tracking_uri = tracking_uri,
        )
        
        loggers = [csv_logger, mlflow_logger]

        console.rule("[bold]Training[/bold]")
        trainer = pl.Trainer(
            max_epochs=cfg.training.epochs,
            callbacks=callbacks,
            logger=loggers,
            accelerator=hparams.accelerator,
            devices=hparams.devices,
            log_every_n_steps=50,
        )
        trainer.fit(experiment, train_loader, val_loader)

        scaler_path = os.path.join(cfg.training.output_dir, run_name + "_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        console.print(f"\n[green]Scaler saved → {scaler_path}[/green]")
        console.print(f"[green]Best checkpoint → {cfg.training.output_dir}/[/green]")

        console.rule("[bold]Testing[/bold]")
        trainer.test(experiment, dataloaders=test_loader)

if __name__ == "__main__":
    parser = ArgumentParser(description="Run a local training and testing pipeline for speaker-specific UltMel2DCNN.")
    parser.add_argument("--config", default="config.yml", help="Path to config YAML")
    parser.add_argument("--data-dir", default=None, help="Override data_dir from config")
    parser.add_argument("--h5-dir", default=None, help="Override h5_dir from config")
    parser.add_argument("--force-preprocess", action="store_true", help="Delete and rebuild the H5 file before training")
    parser.add_argument("--accelerator", default="gpu", help="PyTorch Lightning accelerator to use (e.g. 'cpu', 'gpu', 'cuda', 'mps')")
    parser.add_argument("--devices", default="1", help="PyTorch Lightning devices to use (e.g. '1', '0,1', 'auto')")
    args = parser.parse_args()

    main(args)