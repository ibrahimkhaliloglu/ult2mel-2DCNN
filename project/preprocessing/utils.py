"""
Low-level signal processing helpers for ult-to-mel mapping.

"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.signal as sps
import soundfile as sf
import torch
import torchaudio.functional as F
import cv2

def read_ult(path: str | Path, n_lines: int, n_pixels: int) -> np.ndarray:
    """
    Read a raw ultrasound binary file. 
    Returns (n_frames, n_lines, n_pixels) uint8.
    """
    raw = np.fromfile(str(path), dtype=np.uint8)
    n_frames = len(raw) // (n_lines * n_pixels)
    return raw[: n_frames * n_lines * n_pixels].reshape(n_frames, n_lines, n_pixels)


def reduce_pixels(ult: np.ndarray, n_lines: int, n_pixels_reduced: int) -> np.ndarray:
    """Resize ultrasound frames with bicubic interpolation. """
    ult_resized = np.empty((ult.shape[0], n_lines, n_pixels_reduced), dtype=np.float32)
    for i in range(ult.shape[0]):
        ult_resized[i] = cv2.resize(
            ult[i],
            (n_pixels_reduced, n_lines),
            interpolation=cv2.INTER_CUBIC,
        )
    return ult_resized

def read_wav(path: str | Path) -> Tuple[np.ndarray, int]:
    """Return (samples float64, sample_rate) as mono."""
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr

def resample(wav: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return wav
    n_samples = round(len(wav) * float(target_sr) / original_sr)
    return sps.resample(wav, n_samples)

def read_param(path: str | Path) -> dict:
    """
    Parse a TaL .param file into {key: {"value": ..., "unit": ...}}.
    """
    params: dict = {}
    with open(str(path), "r") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            key, rest = line.split("=", 1)
            parts = rest.strip().split()
            if not parts:
                continue
            try:
                val = float(parts[0])
            except ValueError:
                val = parts[0]
            unit = parts[1] if len(parts) > 1 else None
            params[key.strip()] = {"value": val, "unit": unit}
    return params


_mel_basis: dict = {}


def _get_mel_basis(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
    device: torch.device,
) -> torch.Tensor:
    key = (n_fft, n_mels, sample_rate, fmin, fmax)
    if key not in _mel_basis:
        fb = F.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        _mel_basis[key] = fb
    return _mel_basis[key].to(device)


def compute_mel(
    wav: np.ndarray,
    sample_rate: int,
    n_fft: int,
    n_melband: int,
    n_hop: int,
    win_size: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """
    Compute mel spectrogram matching HiFi-GAN's mel_spectrogram() exactly.
    https://github.com/jik876/hifi-gan
    Returns shape (T, n_melband) — time-first, ready to pair with ULT frames.
    """
    wav_tensor = torch.FloatTensor(wav).unsqueeze(0)  # (1, T)

    wav_padded = torch.nn.functional.pad(
        wav_tensor,
        (n_fft // 2, n_fft // 2),
        mode="reflect",
    ).squeeze(0) 

    window = torch.hann_window(win_size)

    stft = torch.stft(
        wav_padded,
        n_fft=n_fft,
        hop_length=n_hop,
        win_length=win_size,
        window=window,
        center=False,
        return_complex=True,
    )  

    magnitude = torch.abs(stft) 

    mel_fb = _get_mel_basis(n_fft, n_melband, sample_rate, fmin, fmax, magnitude.device)
    mel = torch.matmul(mel_fb.T, magnitude)  
    mel = torch.log(torch.clamp(mel, min=1e-5))

    return mel.T.numpy() 

def sync_ult_wav(
    ult: np.ndarray,
    wav: np.ndarray,
    start_time: float,
    frame_per_sec: float,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trim both ULT and WAV to share the same time window,
    cutting the pre-recording beep indicated by start_time.
    Returns (ult_synced, wav_synced).
    """
    end_time = len(wav) / sample_rate
    frame_end = int((end_time - start_time) * frame_per_sec)
    ult_synced = ult[:frame_end]

    sample_start = int(start_time * sample_rate)
    wav_synced = wav[sample_start:]

    return ult_synced, wav_synced