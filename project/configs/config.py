"""
Typed configuration using Pydantic v2 + YAML.

"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import yaml
from pydantic import BaseModel, Field, computed_field, model_validator, field_validator

# validators can be changed and made into more comprehensive checks

class DataConfig(BaseModel):
    model_config = {"frozen": False}
    # Raw TaL corpus location
    data_dir: str = "./TaL80/core"
    # Where preprocessed HDF5 files are written (one per speaker)
    h5_dir: str = "./h5_TaL"
    speakers: List[str] = Field(default_factory=lambda: ["01fi"])

    # Ultrasound geometry
    n_lines: int = 64
    n_pixels: int = 842
    n_pixels_reduced: int = 128

    # Audio / mel
    n_melband: int = 80
    sample_rate: int = 22050
    n_fft: int = 1024
    win_size: int = 1024
    fmin: float = 0.0
    fmax: float = 8000.0

    # TaL-specific: suffix range identifying test session files
    seed: int = 17
    test_suffix_range: Tuple[int, int] = (4, 14)
    train_split: float = Field(default=0.9, ge=0.0, le=1.0)

    @computed_field
    @property
    def frames_per_sec(self) -> float:
        """Standard TaL corpus frame rate."""
        return 81.5

    @computed_field
    @property
    def n_hop(self) -> int:
        """Hop size in samples, derived from sample_rate and frames_per_sec."""
        return int(self.sample_rate / self.frames_per_sec)

    @computed_field
    @property
    def h5_path(self) -> Path:
        return Path(self.h5_dir)
    
    @field_validator("data_dir", "h5_dir", mode="after")
    @classmethod
    def resolve_path(cls, v: str) -> str:
        return str(Path(v).resolve())    


class ModelConfig(BaseModel):
    conv_filters: List[int] = Field(default_factory=lambda: [30, 60, 90, 120])
    kernel_size: int = 13
    dense_units: int = 1000
    dropout_rate: float = Field(default=0.2, ge=0.0, le=1.0)
    l1_reg: float = 1e-5
    l1_reg_dense: float = 5e-6


class TrainingConfig(BaseModel):
    epochs: int = Field(default=50, gt=0)
    batch_size: int = Field(default=128, gt=0)
    learning_rate: float = Field(default=0.1, gt=0.0)
    momentum: float = Field(default=0.1, ge=0.0, le=1.0)
    nesterov: bool = True

    early_stop_patience: int = 3
    early_stop_min_delta: float = 1e-4

    lr_reduce_patience: int = 2
    lr_reduce_factor: float = Field(default=0.5, gt=0.0, le=1.0)
    lr_min: float = 1e-4

    output_dir: str = "models"

class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @model_validator(mode="after")
    def check_h5_dir_not_same_as_data_dir(self) -> "Config":
        if self.data.h5_dir == self.data.data_dir:
            raise ValueError("h5_dir must be different from data_dir")
        return self

def load_config(path: str = "config.yml") -> Config:
    """Load and validate a YAML config file, returning a fully typed Config."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)