# Copyright (c) 2022 Alpha-VL
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE:  https://github.com/facebookresearch/mae
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
import rasterio

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_convvit

from engine_finetune import train_one_epoch, evaluate


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
    def build_transform(is_train, input_size, mean, std, use_augmentation=True):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :param use_augmentation: Whether to use data augmentation for training
        :return: Torch data transform for the input image before passing to model
        """
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(ToTensorMultiband())
            t.append(transforms.Normalize(mean, std))
            if use_augmentation:
                t.append(
                    transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),
                )
                t.append(transforms.RandomHorizontalFlip())
            else:
                # Simple resize and center crop for training without augmentation
                if input_size <= 224:
                    crop_pct = 224 / 256
                else:
                    crop_pct = 1.0
                size = int(input_size / crop_pct)
                t.append(transforms.Resize(size, interpolation=interpol_mode))
                t.append(transforms.CenterCrop(input_size))
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
    mean = [0.5] * 5  # Default to 0.5 for all channels
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
            raise ValueError(f"Dataset directory not found: {root_dir}")

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
    parser = argparse.ArgumentParser('ConvMAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='convvit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--in_chans', default=5, type=int,
                        help='Number of input image channels')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Use data augmentation (RandomResizedCrop, RandomHorizontalFlip)')
    parser.set_defaults(use_augmentation=True)
    parser.add_argument('--no_augmentation', action='store_false', dest='use_augmentation',
                        help='Disable data augmentation')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')

    parser.add_argument('--input_bands', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Which bands to use for TIFF images (e.g., --input_bands 4 2 1). Number of bands must match --in_chans.')

    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

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
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
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
    print('ConvMAE Fine-tuning Script (Multispectral)')
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
        raise ValueError("Please set --data_path before training or evaluation.")

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
    print(f"🏷️  Number of classes: {args.nb_classes}")

    # Create train dataset
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
        std=dataset_train.std,
        use_augmentation=args.use_augmentation
    )
    dataset_train.transform = transform_train
    
    # Create val dataset
    dataset_val = MultibandDataset(
        root_dir=os.path.join(args.data_path, 'val'),
        selected_bands=selected_bands,
        in_chans=args.in_chans
    )
    
    transform_val = SatelliteDataset.build_transform(
        is_train=False,
        input_size=args.input_size,
        mean=dataset_val.mean,
        std=dataset_val.std
    )
    dataset_val.transform = transform_val

    print(f"\n📋 Dataset Info:")
    print(f"   - Train samples: {len(dataset_train)}")
    print(f"   - Val samples: {len(dataset_val)}")
    print(f"   - Classes: {len(dataset_train.class_names)}")
    print(f"   - Class names: {dataset_train.class_names}")
    print(f"   - Mean: {[f'{x:.3f}' for x in dataset_train.mean]}")
    print(f"   - Std:  {[f'{x:.3f}' for x in dataset_train.std]}")
    
    # Print sample input info
    sample_img, sample_label = dataset_train[0]
    print(f"\n🔍 Sample input info: image shape: {sample_img.shape}, label: {sample_label}")

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None and not args.eval:
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

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    model = models_convvit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        in_chans=args.in_chans,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        
        # Remove head weights if shape doesn't match
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # Handle input channel mismatch for patch_embed1
        if 'patch_embed1.proj.weight' in checkpoint_model:
            pretrain_chans = checkpoint_model['patch_embed1.proj.weight'].shape[1]
            if pretrain_chans != args.in_chans:
                print(f"⚠️  Input channels mismatch: pretrained={pretrain_chans}, current={args.in_chans}")
                print(f"   Reinitializing patch_embed1.proj weights")
                del checkpoint_model['patch_embed1.proj.weight']
                if 'patch_embed1.proj.bias' in checkpoint_model:
                    del checkpoint_model['patch_embed1.proj.bias']

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            expected_missing = {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            # Add patch_embed1 weights if they were removed due to channel mismatch
            if 'patch_embed1.proj.weight' not in checkpoint_model:
                expected_missing.add('patch_embed1.proj.weight')
                expected_missing.add('patch_embed1.proj.bias')
            # Check missing keys
            actual_missing = set(msg.missing_keys)
            if not actual_missing.issubset(expected_missing):
                unexpected_missing = actual_missing - expected_missing
                print(f"⚠️  Unexpected missing keys: {unexpected_missing}")
        else:
            expected_missing = {'head.weight', 'head.bias'}
            if 'patch_embed1.proj.weight' not in checkpoint_model:
                expected_missing.add('patch_embed1.proj.weight')
                expected_missing.add('patch_embed1.proj.bias')

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n🧠 Model Configuration:")
    print('-'*50)
    print(f"🏗️  Model: {args.model}")
    print(f"🎨 Input channels: {args.in_chans}")
    print(f"🏷️  Number of classes: {args.nb_classes}")
    print(f"📉 Drop path rate: {args.drop_path}")
    print(f"🌐 Global pool: {args.global_pool}")
    print(f"🔢 Number of params (M): {n_parameters / 1.e6:.2f}")

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
    print(f"📉 Layer decay: {args.layer_decay}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print(f"📊 Criterion: {criterion}")

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"\n🚀 Starting Training:")
    print('='*80)
    print(f"📅 Training for {args.epochs} epochs")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"📊 Log directory: {args.log_dir}")
    print('='*80)
    
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        #max_accuracy = max(max_accuracy, test_stats["acc1"])
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            misc.save_best_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

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
    print(f'🏆 Best accuracy: {max_accuracy:.2f}%')
    print(f'💾 Model saved to: {args.output_dir}')
    print('='*80)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
