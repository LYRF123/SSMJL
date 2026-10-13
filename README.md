# SSMJL

SSMJL contains multispectral soybean remote-sensing training code for self-supervised pretraining and downstream classification.

The repository currently focuses on three workflows:

- SupCon supervised contrastive pretraining
- ConvMAE multispectral masked autoencoder pretraining, fine-tuning, and linear probing
- Single-encoder and dual-encoder downstream classification fine-tuning

Object detection examples have been removed. The repository does not include raw datasets, model checkpoints, or personal path configuration.

## Project Layout

```text
Pretrain/
  SupCon/                         SupCon multispectral pretraining
  MCMAE-Multispectral/            ConvMAE pretraining, fine-tuning, linear probing
Finetune/
  finetune_classifier_head.py     Single-encoder ViT classifier fine-tuning
  train_dual_classifier.py        ConvMAE + SupCon dual-encoder classifier fine-tuning
requirements.txt
```

## Installation

Install the PyTorch build that matches your CUDA environment first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The MCMAE code is based on an older ConvMAE stack and requires `timm==0.3.2`. This version is pinned in `requirements.txt`. If your current environment already uses a newer `timm`, using a fresh virtual environment is recommended.

The training entry points include compatibility aliases for the legacy `torch._six` imports used by timm 0.3.2. CPU regression checks were run with Python 3.9, PyTorch 1.13.1, torchvision 0.14.1, NumPy 1.26.4, and Kornia 0.6.12.

## Data Layout

For SupCon pretraining, `--data_path` should point directly to class folders:

```text
supcon_dataset/
  0/*.tif
  1/*.tif
  2/*.tif
```

For MCMAE pretraining and fine-tuning, `--data_path` should point to a dataset root containing `train/` and `val/`. Pretraining reads `train/`; fine-tuning reads both `train/` and `val/`:

```text
mcmae_dataset/
  train/
    0/*.tif
    1/*.tif
    2/*.tif
  val/
    0/*.tif
    1/*.tif
    2/*.tif
```

Band indices are 0-based. For example, `--bands 0,1,2,3,4` or `--input_bands 0 1 2 3 4` reads the first five bands.

## Quick Start

SupCon pretraining:

```bash
python Pretrain/SupCon/10band_Supervised.py \
  --data_path "<supcon_dataset>" \
  --output_dir "./output_dir/supcon" \
  --bands 0,1,2,3,4 \
  --device cuda
```

ConvMAE pretraining:

```bash
python Pretrain/MCMAE-Multispectral/main_pretrain.py \
  --data_path "<mcmae_dataset>" \
  --output_dir "./output_dir/mcmae_pretrain" \
  --log_dir "./output_dir/mcmae_pretrain" \
  --input_bands 0 1 2 3 4 \
  --in_chans 5 \
  --device cuda
```

For small smoke tests without a GPU, replace `--device cuda` with `--device cpu`. GPU training is recommended for real experiments.

## Fine-Tuning

Single-encoder classifier fine-tuning:

```bash
python Finetune/finetune_classifier_head.py \
  --train-data-dir "<train_dir>" \
  --val-data-dir "<val_dir>" \
  --pretrained-backbone-path "<optional_supcon_checkpoint.pth>" \
  --bands 0,1,2,3,4 \
  --num-classes 3 \
  --output-dir "./output_dir/finetune_single"
```

Dual-encoder classifier fine-tuning expects paired high-temperature and control images. File stems should match across the paired folders:

```text
train_ht/0/sample_001.tif
train_ck/0/sample_001.tif
val_ht/0/sample_101.tif
val_ck/0/sample_101.tif
```

Pairs are matched by class folder and file stem. Unmatched or duplicate pairs stop training with an error. Validation uses the training class-to-index mapping, including when a validation split omits a class. Pretrained encoder and projection weights must be complete and have matching shapes; incompatible checkpoints are rejected. The dual-encoder report evaluates the saved best model.

```bash
python Finetune/train_dual_classifier.py \
  --train-ht "<train_high_temperature_dir>" \
  --train-ck "<train_control_dir>" \
  --val-ht "<val_high_temperature_dir>" \
  --val-ck "<val_control_dir>" \
  --convmae-ckpt "<convmae_checkpoint.pth>" \
  --supcon-ckpt "<supcon_checkpoint.pth>" \
  --bands 0,1,2,3,4 \
  --num-classes 3 \
  --output-dir "./output_dir/finetune_dual"
```

MCMAE fine-tuning and linear-probing commands are documented in [Pretrain/MCMAE-Multispectral/FINETUNE.md](Pretrain/MCMAE-Multispectral/FINETUNE.md).

Note: `Pretrain/MCMAE-Multispectral/main_linprobe.py` still follows the upstream RGB/ImageFolder linear-probing workflow. Use `main_finetune.py` for multispectral TIF fine-tuning, or extend `main_linprobe.py` before using it directly on multispectral TIF data.

ConvMAE Tiny pretraining uses stage widths `[64, 96, 192]`, depths `[2, 2, 8]`, and three attention heads. Use `--model convvit_tiny_patch16` in MCMAE fine-tuning to load these checkpoints. Older standalone ConvViT Tiny checkpoints with widths `[96, 192, 384]` require their original architecture.

## Regression Checks

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```

These checks cover checkpoint transfer, multi-band augmentation, paired TIFF data, validation labels, encoder freezing, contrastive loss, and gradient accumulation. Synthetic CPU training runs verify the main pretraining and classifier workflows; they do not reproduce the paper's reported accuracy.

## Outputs

Training outputs are written to the directory passed through the command line, such as `./output_dir/...`.

Do not commit:

- Raw TIF/TIFF remote-sensing images
- `.pth`, `.pt`, `.ckpt`, or other model checkpoint files
- `output_dir/`, `finetune_results/`, TensorBoard logs, or experiment artifacts
- Personal absolute paths, account information, or local-only configuration files
