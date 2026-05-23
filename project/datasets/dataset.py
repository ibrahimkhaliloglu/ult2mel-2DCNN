"""
PyTorch Dataset backed by HDF5 (.h5) files.

Each speaker has one file on disk:
  {speaker}.h5
    /ult          (N, n_lines, n_pixels_reduced)  float32  values in [-1, 1]
    /mel          (N, n_melband)                  float32
    /boundaries   (R, 2)                          int64
    /filenames    (R,)                            str

h5py opens the file and reads individual frames on demand — the OS pages
data from disk as needed. DataLoader workers each open their own h5py handle
(opened lazily in __getitem__) which is safe for multiprocessing.
"""

from __future__ import annotations

import random
from typing import Tuple

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from project.configs.config import DataConfig, TrainingConfig
from project.preprocessing.synced_h5 import h5_paths

class UltMelDataset(Dataset):
    def __init__(self, h5_path, indices, scaler=None):
        self.h5_path = str(h5_path)
        self.indices = indices
        self.scaler  = scaler
        self._handles: dict = {}

    def _get_handle(self, h5_path: str) -> h5py.File:
        if h5_path not in self._handles:
            self._handles[h5_path] = h5py.File(h5_path, "r")
        return self._handles[h5_path]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        f = self._get_handle(self.h5_path)
        ult = f["ult"][global_idx].astype(np.float32) / 255.0 * 2.0 - 1.0
        mel = f["mel"][global_idx].astype(np.float32)

        if self.scaler is not None:
            mel = self.scaler.transform(mel.reshape(1, -1)).squeeze(0).astype(np.float32)

        return torch.from_numpy(ult).unsqueeze(0), torch.from_numpy(mel)


def _is_test_utterance(filename: str, suffix_range: Tuple[int, int]) -> bool:
    prefix = filename.split("_")[0]  # "004"
    try:
        session_num = int(prefix)
        return suffix_range[0] <= session_num < suffix_range[1]
    except ValueError:
        return False


def build_dataloaders(
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    assert len(data_cfg.speakers) == 1, "Single-speaker training only"
    speaker = data_cfg.speakers[0]
    h5_path = str(h5_paths(data_cfg, speaker))

    with h5py.File(h5_path, "r") as f:
        boundaries = f["boundaries"][:]
        filenames  = [fn.decode() if isinstance(fn, bytes) else fn
                      for fn in f["filenames"][:]]
        mel_data   = f["mel"][:]

    is_test = [_is_test_utterance(fn, data_cfg.test_suffix_range)
               for fn in filenames]

    test_utts     = [i for i, t in enumerate(is_test) if t]
    non_test_utts = [i for i, t in enumerate(is_test) if not t]

    rng = random.Random(data_cfg.seed)
    rng.shuffle(non_test_utts)
    n_train = max(1, int(len(non_test_utts) * data_cfg.train_split))
    train_utts = non_test_utts[:n_train]
    val_utts   = non_test_utts[n_train:]

    def frames(utts):
        if not utts:
            return np.array([], dtype=np.int64)
        return np.concatenate([
            np.arange(boundaries[u][0], boundaries[u][1], dtype=np.int64)
            for u in utts
        ])

    train_frames = frames(train_utts)

    scaler = StandardScaler()
    scaler.fit(mel_data[train_frames]) 

    def make_loader(indices, shuffle):
        return DataLoader(
            UltMelDataset(h5_path, indices, scaler=scaler),
            batch_size=train_cfg.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )

    return (
        make_loader(train_frames,        shuffle=True),
        make_loader(frames(val_utts),    shuffle=False),
        make_loader(frames(test_utts),   shuffle=False),
        scaler,
    )