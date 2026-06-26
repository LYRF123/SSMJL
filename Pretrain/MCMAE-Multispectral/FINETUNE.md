# MCMAE-Multispectral

本目录包含 ConvMAE/ConvViT 在多光谱 TIF 图像上的预训练、分类微调和线性探测代码。目标检测相关示例已经从仓库移除。

## Environment

根目录 `requirements.txt` 已列出主要依赖。该部分代码要求：

```bash
pip install timm==0.3.2
```

普通单机训练请直接运行 `main_pretrain.py`、`main_finetune.py` 或 `main_linprobe.py`。`submitit_*.py` 只用于 Slurm 集群任务提交。

## Data Layout

预训练和分类微调入口使用多波段 TIF/TIFF 数据，`--data_path` 指向的数据根目录建议按如下方式组织：

```text
dataset_root/
  train/
    0/*.tif
    1/*.tif
    2/*.tif
  val/
    0/*.tif
    1/*.tif
    2/*.tif
```

预训练入口读取 `dataset_root/train`。分类微调读取 `dataset_root/train` 与 `dataset_root/val`。

线性探测入口 `main_linprobe.py` 仍保留上游 ConvMAE 的 `torchvision.datasets.ImageFolder`/RGB 图像流程，适合 3 通道图像或已经转换好的 ImageFolder 数据。如果要对多波段 TIF 做线性探测，建议先使用 `main_finetune.py`，或再扩展 `main_linprobe.py` 的 rasterio 数据加载逻辑。

## Pretrain

```bash
python main_pretrain.py \
  --data_path "<dataset_root>" \
  --output_dir "./output_dir/mcmae_pretrain" \
  --log_dir "./output_dir/mcmae_pretrain" \
  --input_bands 0 1 2 3 4 \
  --in_chans 5 \
  --batch_size 64 \
  --epochs 200 \
  --device cuda
```

## Fine-Tune

```bash
python main_finetune.py \
  --data_path "<dataset_root>" \
  --finetune "<pretrained_checkpoint.pth>" \
  --output_dir "./output_dir/mcmae_finetune" \
  --log_dir "./output_dir/mcmae_finetune" \
  --input_bands 0 1 2 3 4 \
  --in_chans 5 \
  --nb_classes 3 \
  --batch_size 64 \
  --epochs 100 \
  --device cuda
```

The evaluation code automatically skips Acc@5 when the number of classes is smaller than 5.

## Linear Probe

该入口当前是 RGB/ImageFolder 线性探测流程，不读取 `--input_bands`。

```bash
python main_linprobe.py \
  --data_path "<dataset_root>" \
  --finetune "<pretrained_checkpoint.pth>" \
  --output_dir "./output_dir/mcmae_linprobe" \
  --log_dir "./output_dir/mcmae_linprobe" \
  --nb_classes 3 \
  --batch_size 256 \
  --epochs 90 \
  --device cuda
```

## CPU Debugging

For a quick CPU smoke test, use a very small dataset and pass `--device cpu`, `--epochs 1`, and a small `--batch_size`. GPU is recommended for real training.

## Slurm / Submitit

`submitit_pretrain.py`、`submitit_finetune.py` 和 `submitit_linprobe.py` 用于 Slurm 集群。没有集群环境时，请直接运行上面的 `main_*.py`。

这些脚本不再内置特定集群的共享目录、队列名或 GPU 约束。需要提交到 Slurm 时，请按自己的环境显式传入 `--job_dir`、`--partition` 和 `--constraint`；如果不传 `--job_dir`，日志目录会默认放在 `<output_dir>/submitit/%j`。
