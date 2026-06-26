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

## Outputs

Training outputs are written to the directory passed through the command line, such as `./output_dir/...`.

Do not commit:

- Raw TIF/TIFF remote-sensing images
- `.pth`, `.pt`, `.ckpt`, or other model checkpoint files
- `output_dir/`, `finetune_results/`, TensorBoard logs, or experiment artifacts
- Personal absolute paths, account information, or local-only configuration files

## Slurm / Submitit

`Pretrain/MCMAE-Multispectral/submitit_*.py` scripts are intended for Slurm clusters. For normal single-machine training, run `main_pretrain.py`, `main_finetune.py`, or `main_linprobe.py` directly.

The submitit scripts do not hard-code a cluster-specific shared directory, partition, or GPU constraint. When submitting to Slurm, pass `--job_dir`, `--partition`, and `--constraint` according to your own environment. If `--job_dir` is omitted, logs default to `<output_dir>/submitit/%j`.

## Repository Hygiene

Hard-coded personal paths and object detection examples have been removed. Before future commits, check the working tree and scan for local-only patterns:

```bash
git status
rg -n "<your local privacy patterns>" .
```

Keep dataset paths in local scripts or untracked configuration files rather than committing them to the repository.
