"""
project/preprocessing/synced_h5.py

Preprocesses raw TaL corpus files and saves one HDF5 (.h5) file per speaker.

H5 file layout (h5_dir/)
--------------------------
{speaker}.h5
  /ult          dataset  shape (N, n_lines, n_pixels_reduced)  dtype uint8
  /mel          dataset  shape (N, n_melband)                  dtype float32
  /boundaries   dataset  shape (R, 2)                          dtype int64
  /filenames    dataset  shape (R,)                            dtype variable-length str

HDF5 datasets are written with chunking and gzip compression.
Random-access reads remain O(1): h5py returns a NumPy array slice without
loading the full dataset, so RAM usage stays proportional to what the
DataLoader actually accesses.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from project.configs.config import DataConfig
from project.preprocessing.utils import (
    compute_mel,
    read_param,
    read_ult,
    read_wav,
    reduce_pixels,
    resample,
    sync_ult_wav,
)

console = Console()

def _h5_path(cfg: DataConfig, speaker: str) -> Path:
    return Path(cfg.h5_dir) / f"{speaker}.h5"

def h5_exists(cfg: DataConfig, speaker: str) -> bool:
    """True only if the .h5 file is present."""
    return _h5_path(cfg, speaker).exists()

def h5_paths(cfg: DataConfig, speaker: str) -> Path:
    """Return the .h5 path for a speaker."""
    return _h5_path(cfg, speaker)


def _collect_files(data_dir: Path) -> List[str]:
    """Return sorted list of base paths (no extension) for ULT sessions."""
    stems = []
    for file in sorted(os.listdir(data_dir)):
        if (file.endswith("aud.ult") or file.endswith("spo.ult")):
            stems.append(str(data_dir / file[:-4]))
    return stems


def build_speaker_h5(cfg: DataConfig, speaker: str) -> None:
    """
    Preprocess all recordings for one speaker and write a single HDF5 file:
      {speaker}.h5
        /ult          (N, n_lines, n_pixels_reduced)  uint8
        /mel          (N, n_melband)                  float32
        /boundaries   (R, 2)                          int64 *boundary of utterances
        /filenames    (R,)                            variable-length str

    Skips silently if the h5 file already exists.
    """
    if h5_exists(cfg, speaker):
        console.print(
            f"[dim]  ↳ H5 file already exists for [bold]{speaker}[/bold], skipping.[/dim]"
        )
        return

    h5_dir = Path(cfg.h5_dir)
    h5_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(cfg.data_dir) / speaker
    file_stems = _collect_files(data_dir)

    if not file_stems:
        console.print(f"[yellow] No ULT files found for speaker {speaker}[/yellow]")
        return

    console.print(
        f"[cyan]  Processing [bold]{speaker}[/bold] — "
        f"{len(file_stems)} recording(s)[/cyan]"
    )

    ult_chunks: List[np.ndarray] = []
    mel_chunks: List[np.ndarray] = []
    boundaries: List[Tuple[int, int]] = []
    filenames: List[str] = []
    frame_cursor = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task(f"  {speaker}", total=len(file_stems))

        for stem in file_stems:
            fname = os.path.basename(stem)
            progress.update(task, description=f"  {speaker} › {fname}")

            try:
                ult_raw          = read_ult(stem + ".ult", cfg.n_lines, cfg.n_pixels)
                params           = read_param(stem + ".param")
                wav_raw, orig_sr = read_wav(stem + ".wav")

                frame_per_sec: float = params["FramesPerSec"]["value"]
                start_time: float    = params["TimeInSecsOfFirstFrame"]["value"]

                wav_rs = resample(wav_raw, orig_sr, cfg.sample_rate)

                ult_synced, wav_synced = sync_ult_wav(
                    ult_raw, wav_rs, start_time, frame_per_sec, cfg.sample_rate
                )

                ult_reduced = reduce_pixels(ult_synced, cfg.n_lines, cfg.n_pixels_reduced)

                n_hop = int(cfg.sample_rate / frame_per_sec)
                mel = compute_mel(
                    wav_synced,
                    cfg.sample_rate,
                    cfg.n_fft,
                    cfg.n_melband,
                    n_hop,
                    cfg.win_size,
                    cfg.fmin,
                    cfg.fmax,
                )

                n           = min(len(ult_reduced), len(mel))
                ult_reduced = ult_reduced[:n]
                mel         = mel[:n]

                ult_chunks.append(ult_reduced.astype(np.uint8))
                mel_chunks.append(mel.astype(np.float32))
                boundaries.append((frame_cursor, frame_cursor + n))
                filenames.append(fname)
                frame_cursor += n

            except Exception as exc:
                console.print(f"[red]Failed on {fname}: {exc}[/red]")

            progress.advance(task)

    if not ult_chunks:
        console.print(
            f"[red]No valid data for {speaker}, skipping write.[/red]"
        )
        return

    ult_all = np.concatenate(ult_chunks, axis=0)   # (N, 64, 128)  uint8
    mel_all = np.concatenate(mel_chunks, axis=0)   # (N, 80)       float32

    h5_path = _h5_path(cfg, speaker)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            "ult",
            data=ult_all,
            chunks=(1, cfg.n_lines, cfg.n_pixels_reduced),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "mel",
            data=mel_all,
            chunks=(1, cfg.n_melband),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "boundaries",
            data=np.array(boundaries, dtype=np.int64),
        )
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(
            "filenames",
            data=np.array(filenames, dtype=object),
            dtype=dt,
        )

    size_mb = h5_path.stat().st_size / 1024 ** 2
    console.print(
        f"[green] {speaker}: {frame_cursor:,} frames "
        f"| ULT {ult_all.shape} | MEL {mel_all.shape} "
        f"| {size_mb:.1f} MB on disk → {h5_path}[/green]"
    )

def build_h5(cfg: DataConfig) -> None:
    """Build .h5 per speaker with synced ult and mel data."""
    console.rule("[bold]Preprocessing — building H5 data[/bold]")
    for speaker in cfg.speakers:
        build_speaker_h5(cfg, speaker)
    console.rule("[bold green]H5 ready[/bold green]")