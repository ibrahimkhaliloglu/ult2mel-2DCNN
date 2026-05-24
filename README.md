# ult2mel-2DCNN
A PyTorch + PyTorch Lightning based implementation of 2D CNN pipeline for mapping ultrasound (ULT) tongue imaging frames to mel spectrogram frames, trained on the [TaL corpus](https://ultrasuite.github.io/papers/tal_corpus_SLT2021.pdf).
The model structure utilized here is based on the [Keras baseline](https://github.com/BME-SmartLab/UTI-to-STFT/blob/master/UTI_to_STFT_CNN_train.py) originally developed at BME SmartLab by [Tamás Gábor Csapó](https://scholar.google.com/citations?user=ivoOEbkAAAAJ&hl=hu).


## Project Structure

```
ult2mel-2DCNN/
├── main.py                        # Pipeline entry point (preprocess → train → test)
├── predict.py                     # Run inference on a test utterance (local or HuggingFace)
├── config.yml                     # All hyperparameters and paths
├── pixi.toml                      # Reproducible environment (pixi)
├── requirement.txt                # pip fallback
└── project/
    ├── configs/config.py          # Pydantic v2 typed config
    ├── datasets/dataset.py        # HDF5-backed Dataset + DataLoader builders
    ├── models/cnn.py              # UltMel2DCNN architecture
    ├── preprocessing/
    │   ├── synced_h5.py           # Builds per-speaker HDF5 files
    │   └── utils.py               # Signal processing helpers
    └── experiment.py              # LightningModule (training / val / test logic)
```

## 1. Set up the environment

**With pixi:** install pixi via [the official instructions](https://pixi.prefix.dev/latest/installation/), then:
> **Note:** This environment is configured for `linux-64`. If you are on a different platform,
> pixi will notify you during `pixi install` and you may need to add your platform to `pixi.toml`
> under `platforms` before proceeding.

```bash
git clone https://github.com/ibrahimkhaliloglu/ult2mel-2DCNN.git
cd ult2mel-2DCNN
pixi install
pixi s
```

## 2. Download the dataset

More info on [how to download TaL80 corpus](https://ultrasuite.github.io/data/tal_corpus/#download). Chosen speaker IDs: `01fi`, `02fe`, `03mn`, `04me`. For training, we only utilize `aud`, `xaud`, `spo` tagged utterances.

```bash
rsync -av ultrasuite-rsync.inf.ed.ac.uk::tal-corpus/TaL80/core/{speaker_id} .
```

> **Note:** Instead of building the HDF5 file from raw TaL corpus files, you can download
> the preprocessed `.h5` files directly from
> [ibrahimkhaliloglu/TaL80-UTI-mel-hdf5](https://huggingface.co/datasets/ibrahimkhaliloglu/TaL80-UTI-mel-hdf5)
> on Hugging Face.

## 3. Configure the run

Edit `config.yml` to point at your data and pick speakers:

```yaml
data:
  data_dir: "./TaL80/core"     # where you downloaded the speakers
  h5_dir:   "./h5_TaL"      # where preprocessed h5 files will be written
  speakers: ["01fi", "02fe", "03mn", "04me"]
training:
  epochs: 50
  batch_size: 128
  output_dir: "models"         # where checkpoints + scalers go
```

Or you can use `--data_dir` flag for overriding config file from command line. Additionally, you can use `--h5_dir` for defining preprocessed H5 file directly using command line. For additional info about possible flags for the demonstrated scripts below, you can use `--help`.

## 4. Train and test

A single command runs the full pipeline (preprocess → train → test) for every speaker listed in `config.yml`:

```bash
python main.py --config config.yml
```

Useful flags:

```bash
python main.py --accelerator cpu --devices auto    # no GPU
python main.py --force-preprocess                  # rebuild the HDF5 file
```

For each speaker you'll get a best checkpoint, a fitted scaler, and CSV logs under `models/`, plus an MLflow run in `./mlruns/`. The test set is evaluated automatically on the best checkpoint after training finishes.

## 5. Inspect runs

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open `http://localhost:5000` to compare train/val/test MSE across speakers and runs.

## 6. Predict on a test utterance

A single script handles both local and HuggingFace checkpoints via `--source`:

```bash
# From a local training run
python predict.py \
  --source local \
  --speaker 01fi \
  --checkpoint models/UltMel2DCNN_01fi_xxx.ckpt \
  --scaler    models/UltMel2DCNN_01fi_xxx_scaler.pkl \
  --utterance 004_xaud \
  --plot

# From Hugging Face (only --speaker required)
python predict.py --source hf --speaker 01fi
python predict.py --source hf --speaker 01fi --utterance 004_xaud --plot
```

Avoid using `--utterance` to pick interactively from the test set, or pass `--list` to print available utterances and exit.

Pretrained per-speaker checkpoints are hosted at [`ibrahimkhaliloglu/ult2mel_2DCNN`](https://huggingface.co/ibrahimkhaliloglu/ult2mel_2DCNN).

Outputs saved in `predictions/`:

```
predictions/01fi_004_xaud_pred.npy    # predicted mel,    shape (T, n_melband)
predictions/01fi_004_xaud_gt.npy      # ground-truth mel, shape (T, n_melband)
predictions/01fi_004_xaud.png         # if --plot was set
```

## Results

Test-set Mean-MSE on the TaL80 held-out sessions (`test_suffix_range = [4, 14)`):

| Speaker | Test Mean-MSE | Notes |
|---------|---------:|-------|
| 01fi    | *0.37*  |       |
| 02fe    | *0.49*  |    Due to few utterances   |
| 03mn    | *0.36*  |       |
| 04me    | *0.39*  |       |

Side Note: Comparing to [INTERSPEECH 2025 Conformer paper baseline](https://www.isca-archive.org/interspeech_2025/ibrahimov25_interspeech.html): *The results are much better here as I utilized more efficient synchronization technique between ultrasound and mel frames. That section of the pipeline will be introduced in depth soon.*


## Citation
 
If you use this code, please cite the following works.
 
**This work**
```bibtex
@inproceedings{ibrahimov25_interspeech,
  title     = {{Conformer-based Ultrasound-to-Speech Conversion}},
  author    = {Ibrahim Ibrahimov and Csaba Zainkó and Gábor Gosztolya},
  booktitle = {{Interspeech 2025}},
  year      = {2025},
  pages     = {5578--5582},
  doi       = {10.21437/Interspeech.2025-2147},
  issn      = {2958-1796},
}
```
 
**CNN baseline (Csapó et al.)**
```bibtex
@inproceedings{csapo20b_interspeech,
  title     = {{Ultrasound-Based Articulatory-to-Acoustic Mapping with WaveGlow Speech Synthesis}},
  author    = {Tamás Gábor Csapó and Csaba Zainkó and László Tóth and Gábor Gosztolya and Alexandra Markó},
  booktitle = {{Interspeech 2020}},
  year      = {2020},
  pages     = {2727--2731},
  doi       = {10.21437/Interspeech.2020-1031},
  issn      = {2958-1796},
}
```
 
**TaL corpus**
```bibtex
@inproceedings{ribeiro21_slt,
  title     = {{TaL: a synchronised multi-speaker corpus of ultrasound tongue imaging, audio, and lip videos}},
  author    = {Manuel Sam Ribeiro and Jennifer Sanger and Jing-Xuan Zhang and Aciel Eshky and Alan Wrench and Korin Richmond and Steve Renals},
  booktitle = {{2021 IEEE Spoken Language Technology Workshop (SLT)}},
  year      = {2021},
  pages     = {1109--1116},
  doi       = {10.1109/SLT48900.2021.9383619},
  isbn      = {978-1-7281-7066-4},
}
```