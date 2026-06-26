# Copyright (c) 2022 Alpha-VL
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
import rasterio

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_convmae

from engine_pretrain import train_one_epoch


class ToTensorMultiband:
    """Convert numpy array to tensor for multiband images"""
    def __call__(self, img):
        # img is numpy array (H, W, C)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # (C, H, W)
        return img


class SatelliteDataset(torch.utils.data.Dataset):
    """
    Abstract class for satellite datasets.
    """
    def __init__(self, in_c):
        self.in_c = in_c
        print(f"Dataset initialized with {in_c} channels")

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(ToTensorMultiband())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(ToTensorMultiband())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)


class MultibandDataset(SatelliteDataset):
    """Simple multiband TIFF dataset similar to EuroSat"""
    
    # Default mean and std for normalization (can be calculated from your data)
    mean = [0.5] *5  # Default to 0.5 for all channels
    std = [0.5] * 5   # Default to 0.5 for all channels
    
    def __init__(self, root_dir, transform=None, selected_bands=None, in_chans=5):
        """
        :param root_dir: Directory with subdirectories containing TIFF images
        :param transform: pytorch Transform for transforms and tensor conversion
        :param selected_bands: List of band indices to use (0-indexed)
        :param in_chans: Number of input channels
        """
        super().__init__(in_chans)
        if not os.path.isdir(root_dir):
            raise ValueError(f"Training directory not found: {root_dir}")

        self.root_dir = root_dir
        self.transform = transform
        self.selected_bands = selected_bands if selected_bands is not None else list(range(in_chans))
        
        # Update mean and std for the actual number of channels
        self.mean = self.mean[:in_chans]
        self.std = self.std[:in_chans]
        
        # Get all image paths and labels
        self.img_paths = []
        self.labels = []
        self.class_names = []
        
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_names.append(class_name)
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.tif', '.tiff')):
                        self.img_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)

        if not self.img_paths:
            raise ValueError(
                f"No .tif/.tiff images found under {root_dir}. "
                "Expected class subfolders such as train/0/*.tif."
            )
    
    def __len__(self):
        return len(self.img_paths)
    
    def open_image(self, img_path):
        """Load TIFF image with rasterio"""
        with rasterio.open(img_path) as src:
            # Check if requested bands exist
            if max(self.selected_bands) >= src.count:
                raise ValueError(f"Requested band {max(self.selected_bands)} but image only has {src.count} bands.")
            
            # Read selected bands (rasterio uses 1-based indexing)
            img = src.read([b + 1 for b in self.selected_bands])
            
            # Convert from (bands, height, width) to (height, width, bands)
            img = img.transpose(1, 2, 0).astype(np.float32)
            
        return img
    
    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = self.open_image(img_path)  # (h, w, c)
        
        # Apply transforms
        if self.transform:
            
            img = self.transform(img)
        
        return img, label


def get_args_parser():
    parser = argparse.ArgumentParser('ConvMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='convmae_convvit_tiny_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--in_chans', default=5, type=int,
                        help='Number of input image channels')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default=r'', type=str,
                        help='dataset path')

    parser.add_argument('--input_bands', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Which bands to use for TIFF images (e.g., --input_bands 4 2 1). Number of bands must match --in_chans.')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('='*80)
    print('ConvMAE Pre-training Script')
    print('='*80)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('\nTraining Arguments:')
    print('-'*50)
    for arg, value in vars(args).items():
        print(f'{arg:20}: {value}')
    print('='*80)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    print('\nDataset Configuration:')
    print('-'*50)
    if not args.data_path:
        raise ValueError("Please set --data_path before training.")

    # Handle band selection for TIFF images
    if args.input_bands is None:
        selected_bands = list(range(args.in_chans))
        print(f"📊 No input bands specified, defaulting to first {args.in_chans} bands: {selected_bands}")
    else:
        if len(args.input_bands) != args.in_chans:
            raise ValueError(f"Number of --input_bands ({len(args.input_bands)}) must match --in_chans ({args.in_chans}).")
        selected_bands = args.input_bands
        print(f"📊 Using selected bands: {selected_bands}")
    
    print(f"📁 Data path: {args.data_path}")
    print(f"🖼️  Input size: {args.input_size}x{args.input_size}")
    print(f"🎨 Input channels: {args.in_chans}")

    # Create dataset first to get mean and std
    dataset_train = MultibandDataset(
        root_dir=os.path.join(args.data_path, 'train'),
        selected_bands=selected_bands,
        in_chans=args.in_chans
    )
    
    # Build transform using the dataset's mean and std
    transform_train = SatelliteDataset.build_transform(
        is_train=True,
        input_size=args.input_size,
        mean=dataset_train.mean,
        std=dataset_train.std
    )
    
    # Set the transform to the dataset
    dataset_train.transform = transform_train
    
    print(f"\n📋 Dataset Info:")
    print(f"   - Total samples: {len(dataset_train)}")
    print(f"   - Classes: {len(dataset_train.class_names)}")
    print(f"   - Class names: {dataset_train.class_names}")
    print(f"   - Mean: {[f'{x:.3f}' for x in dataset_train.mean]}")
    print(f"   - Std:  {[f'{x:.3f}' for x in dataset_train.std]}")
    # 打印示例输入信息
    sample_img, sample_label = dataset_train[0]
    print(f"\n🔍 Sample input info: image shape: {sample_img.shape}, label: {sample_label}")

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=len(dataset_train) >= args.batch_size,
    )
    
    # define the model
    model = models_convmae.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        in_chans=args.in_chans
    )

    model.to(device)

    model_without_ddp = model
    
    print(f"\n🧠 Model Configuration:")
    print('-'*50)
    print(f"🏗️  Model: {args.model}")
    print(f"🎭 Mask ratio: {args.mask_ratio}")
    print(f"📐 Norm pixel loss: {args.norm_pix_loss}")
    
    # 计算并打印模型参数数量
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    total_params, trainable_params = count_parameters(model_without_ddp)
    print(f"🔢 Total parameters: {total_params:,}")
    print(f"🎯 Trainable parameters: {trainable_params:,}")
    print(f"💾 Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print(f"\n⚙️  Training Configuration:")
    print('-'*50)
    print(f"📦 Batch size (per GPU): {args.batch_size}")
    print(f"🔄 Accumulate iterations: {args.accum_iter}")
    print(f"📊 Effective batch size: {eff_batch_size}")
    print(f"📈 Base learning rate: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"🎯 Actual learning rate: {args.lr:.2e}")
    print(f"⚖️  Weight decay: {args.weight_decay}")
    print(f"🔥 Warmup epochs: {args.warmup_epochs}")
    print(f"🏃 Total epochs: {args.epochs}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"\n🚀 Starting Training:")
    print('='*80)
    print(f"📅 Training for {args.epochs} epochs")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"📊 Log directory: {args.log_dir}")
    print('='*80)
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 40 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print('\n🎉 Training Completed!')
    print('='*80)
    print(f'⏰ Total training time: {total_time_str}')
    print(f'💾 Model saved to: {args.output_dir}')
    print('='*80)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
