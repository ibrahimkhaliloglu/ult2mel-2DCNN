"""
predict.py - run UltMel2DCNN on a test utterance.

Supports two checkpoint sources:
  --source local      load from a local .ckpt file (requires --checkpoint and --scaler)
  --source hf         download from HuggingFace (only --speaker needed)

Examples
--------
    # HuggingFace (interactive utterance selection)
    python predict.py --source hf --speaker 01fi

    # HuggingFace (direct)
    python predict.py --source hf --speaker 01fi --utterance 004_aud --plot

    # Local (interactive)
    python predict.py --source local --speaker 01fi --checkpoint runs/01fi/model.ckpt --scaler runs/01fi/scaler.pkl

    # Local (direct)
    python predict.py --source local --speaker 01fi --checkpoint runs/01fi/model.ckpt --scaler runs/01fi/scaler.pkl --utterance 004_aud --plot
"""

from __future__ import annotations

import pickle
import warnings
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from huggingface_hub import hf_hub_download

from project.configs.config import load_config
from project.datasets.dataset import _is_test_utterance
from project.experiment import Ult2MelExperiment
from project.models.cnn import UltMel2DCNN
from project.preprocessing.synced_h5 import h5_paths


# ---------------------------------------------------------------------------
HF_REPO = "ibrahimkhaliloglu/ult2mel_2DCNN"
HF_REVISION: str | None = None
SPEAKERS = ["01fi", "02fe", "03mn", "04me"]
# ---------------------------------------------------------------------------


def resolve_weights(args) -> tuple[Path, Path]:
    """Return (ckpt_path, scaler_path) from either HF or local source."""
    if args.source == "hf":
        ckpt = Path(hf_hub_download(
            repo_id=HF_REPO, filename=f"{args.speaker}/model.ckpt", revision=HF_REVISION,
        ))
        scaler = Path(hf_hub_download(
            repo_id=HF_REPO, filename=f"{args.speaker}/scaler.pkl", revision=HF_REVISION,
        ))
        print(f"[hf] checkpoint  → {ckpt}")
        print(f"[hf] scaler      → {scaler}")
        return ckpt, scaler
    else:
        if not args.checkpoint or not args.scaler:
            raise SystemExit("--source local requires --checkpoint and --scaler.")
        return args.checkpoint, args.scaler


def load_model(ckpt_path: Path, cfg, device: torch.device) -> Ult2MelExperiment:
    from project.configs.config import TrainingConfig, ModelConfig, DataConfig
    safe_globals = [TrainingConfig, ModelConfig, DataConfig]
    try:
        with torch.serialization.safe_globals(safe_globals):
            exp = Ult2MelExperiment.load_from_checkpoint(
                str(ckpt_path),
                model=UltMel2DCNN(cfg.model, cfg.data),
                cfg=cfg.training,
                map_location=device,
            )
    except Exception:
        warnings.warn(
            "safe_globals allowlist incomplete — retrying with weights_only=False. "
            "Only do this with checkpoints from a trusted source.",
            stacklevel=2,
        )
        exp = Ult2MelExperiment.load_from_checkpoint(
            str(ckpt_path),
            model=UltMel2DCNN(cfg.model, cfg.data),
            cfg=cfg.training,
            map_location=device,
            weights_only=False,
        )
    exp.eval().to(device)
    return exp


def list_test_utterances(h5_path: Path, cfg) -> list[tuple[str, int, int]]:
    with h5py.File(h5_path, "r") as f:
        filenames = [fn.decode() if isinstance(fn, bytes) else fn
                     for fn in f["filenames"][:]]
        boundaries = f["boundaries"][:]
    return [
        (filenames[i], int(boundaries[i][0]), int(boundaries[i][1]))
        for i in range(len(filenames))
        if _is_test_utterance(filenames[i], cfg.data.test_suffix_range)
    ]


def pick_utterance(h5_path: Path, cfg, utterance: str | None,
                   list_only: bool) -> tuple[str, int, int]:
    test_utts = list_test_utterances(h5_path, cfg)
    if not test_utts:
        raise SystemExit(f"No test utterances in {h5_path}.")

    if list_only or utterance is None:
        print(f"\nTest utterances in {h5_path.name} "
              f"(range {cfg.data.test_suffix_range}):")
        for n, (fn, s, e) in enumerate(test_utts):
            print(f"  [{n:2d}] {fn:<20s}  frames={e - s}")
        if list_only:
            raise SystemExit(0)

        choice = input("\nSelect utterance by number or filename: ").strip()
        if choice.isdigit():
            n = int(choice)
            if not (0 <= n < len(test_utts)):
                raise SystemExit(f"Invalid index {n}.")
            return test_utts[n]
        utterance = choice

    for fn, s, e in test_utts:
        if fn == utterance:
            return fn, s, e
    available = ", ".join(fn for fn, _, _ in test_utts)
    raise SystemExit(f"'{utterance}' not in test set. Available: {available}")


@torch.no_grad()
def predict_utterance(exp, h5_path: Path, start: int, end: int,
                      scaler, device: torch.device,
                      batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        ult_raw = f["ult"][start:end].astype(np.float32) / 255.0 * 2.0 - 1.0
        mel_gt  = f["mel"][start:end].astype(np.float32)

    preds = []
    for i in range(0, len(ult_raw), batch_size):
        batch = torch.from_numpy(ult_raw[i:i + batch_size]).unsqueeze(1).to(device)
        preds.append(exp(batch).cpu().numpy())
    pred_scaled = np.concatenate(preds, axis=0)
    pred_mel = scaler.inverse_transform(pred_scaled).astype(np.float32)
    return pred_mel, mel_gt


def save_plot(pred_mel: np.ndarray, gt_mel: np.ndarray, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].imshow(gt_mel.T, aspect="auto", origin="lower")
    axes[0].set_title("Ground truth mel")
    axes[1].imshow(pred_mel.T, aspect="auto", origin="lower")
    axes[1].set_title("Predicted mel")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"plot         → {out_png}")


def main(args):
    cfg = load_config(args.config)
    cfg.data.speakers = [args.speaker]

    h5_path = h5_paths(cfg.data, args.speaker)
    if not h5_path.exists():
        raise SystemExit(
            f"HDF5 cache not found at {h5_path}.\n"
            f"Run `python main.py` once to build the cache for speaker {args.speaker}."
        )

    utt, start, end = pick_utterance(h5_path, cfg, args.utterance, args.list)
    ckpt_path, scaler_path = resolve_weights(args)

    device = torch.device(args.device)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    exp = load_model(ckpt_path, cfg, device)

    pred_mel, gt_mel = predict_utterance(exp, h5_path, start, end, scaler, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.speaker}_{utt}"
    np.save(args.out_dir / f"{stem}_pred.npy", pred_mel)
    np.save(args.out_dir / f"{stem}_gt.npy",   gt_mel)
    print(f"\npred mel     → {args.out_dir / (stem + '_pred.npy')}  shape={pred_mel.shape}")
    print(f"gt mel       → {args.out_dir / (stem + '_gt.npy')}    shape={gt_mel.shape}")

    if args.plot:
        save_plot(pred_mel, gt_mel, args.out_dir / f"{stem}.png")


if __name__ == "__main__":
    p = ArgumentParser(description="Run UltMel2DCNN on a test utterance (local or HuggingFace).")
    p.add_argument("--source",     required=True, choices=["local", "hf"],
                   help="'local' to use a local checkpoint, 'hf' to download from HuggingFace.")
    p.add_argument("--speaker",    required=True, choices=SPEAKERS,
                   help="Speaker ID, e.g. 01fi.")
    # local-only args
    p.add_argument("--checkpoint", default=None, type=Path,
                   help="[local] Path to a .ckpt file.")
    p.add_argument("--scaler",     default=None, type=Path,
                   help="[local] Path to the matching *_scaler.pkl.")
    # shared args
    p.add_argument("--utterance",  default=None,
                   help="Utterance tag like '004_aud' (omit to pick interactively).")
    p.add_argument("--list",       action="store_true",
                   help="List available test utterances and exit.")
    p.add_argument("--plot",       action="store_true",
                   help="Save a PNG of the predicted vs ground-truth mel spectrogram.")
    p.add_argument("--config",     default="config.yml")
    p.add_argument("--out-dir",    default="predictions", type=Path)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    main(args)