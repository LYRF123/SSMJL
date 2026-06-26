#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双编码器分类微调 — 同时加载 ConvMAE + SupCon 权重
=====================================================
流程（与 finetune_classifier_head.py 一致，只是编码器变两个）：

    高温TIF → ConvMAE编码器(冻结) → 100维 ─┐
    对照TIF → ConvMAE编码器(冻结) → 100维 ─┤ → 200维 ─┐
                                             │          │
    高温TIF → SupCon编码器(冻结) → 100维 ─┐  │          ├→ (f1+f2)/2 → 分类头 → 预测
    对照TIF → SupCon编码器(冻结) → 100维 ─┤ → 200维 ─┘
                                             │
    两个编码器都冻结，只训练分类头。
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from functools import partial
import rasterio
import random
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── 添加项目路径 ──────────────────────────────────────────────
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_current_dir)
for _path in (_current_dir, _repo_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from feature_extractor import ConvMAEFeatureExtractor
from models_vit import VisionTransformer


def parse_bands(value):
    return [int(item.strip()) for item in value.split(',') if item.strip()]


def get_args_parser():
    parser = argparse.ArgumentParser('Dual encoder classifier fine-tuning')
    parser.add_argument('--train-ht', required=True, help='训练集高温图像目录')
    parser.add_argument('--train-ck', required=True, help='训练集对照图像目录')
    parser.add_argument('--val-ht', required=True, help='验证集高温图像目录')
    parser.add_argument('--val-ck', required=True, help='验证集对照图像目录')
    parser.add_argument('--convmae-ckpt', required=True, help='ConvMAE 预训练权重 .pth 路径')
    parser.add_argument('--supcon-ckpt', required=True, help='SupCon 预训练权重 .pth 路径')
    parser.add_argument('--bands', default='0,1,2,3,4', help='使用的波段索引，如 0,1,2,3,4')
    parser.add_argument('--num-classes', type=int, default=3, help='分类类别数')
    parser.add_argument('--dropout-rate', type=float, default=0.2, help='Dropout 率')
    parser.add_argument('--img-size', type=int, default=224, help='输入图像边长')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='分类头学习率')
    parser.add_argument('--wd', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--output-dir', default='./finetune_dual_output', help='输出目录')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader worker 数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser


def config_from_args(args):
    return {
        'train_ht': args.train_ht,
        'train_ck': args.train_ck,
        'val_ht': args.val_ht,
        'val_ck': args.val_ck,
        'convmae_ckpt': args.convmae_ckpt,
        'supcon_ckpt': args.supcon_ckpt,
        'band_indices': parse_bands(args.bands),
        'num_classes': args.num_classes,
        'dropout_rate': args.dropout_rate,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'wd': args.wd,
        'output_dir': args.output_dir,
        'num_workers': args.num_workers,
        'seed': args.seed,
    }


# ================================================================
# 1. SupCon 模型 (ViT encoder + projection)
# ================================================================
class ViTFeatureExtractor(nn.Module):
    """ViT-Tiny encoder"""
    def __init__(self, in_chans=5):
        super().__init__()
        self.vit = VisionTransformer(
            embed_dim=192, depth=12, num_heads=3,
            mlp_ratio=4, qkv_bias=True, in_chans=in_chans,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool=False,
        )
        self.vit.head = nn.Identity()
        self.feature_dim = 192

    def forward(self, x):
        feat = self.vit(x)
        return feat[:, 0, :] if len(feat.shape) == 3 else feat


class SupConModel(nn.Module):
    """SupCon = ViT encoder + projector (192→256→100)"""
    def __init__(self, in_chans=5, hidden_dim=256, output_dim=100):
        super().__init__()
        self.encoder = ViTFeatureExtractor(in_chans=in_chans)
        self.projection = nn.Sequential(
            nn.Linear(192, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.projection(self.encoder(x))


# ================================================================
# 2. 双编码器分类模型
# ================================================================
class DualEncoderClassifier(nn.Module):
    """ConvMAE + SupCon → average fusion → MLP classifier"""
    def __init__(self, convmae_model, supcon_model, num_classes=3, dropout_rate=0.2):
        super().__init__()
        self.convmae = convmae_model   # frozen
        self.supcon = supcon_model     # frozen

        self.classifier = nn.Sequential(
            nn.Linear(200, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, ht_img_cm, ck_img_cm, ht_img_sc, ck_img_sc):
        with torch.no_grad():
            # ConvMAE branch
            h = self.convmae.projection(self.convmae.forward_encoder(ht_img_cm))  # (B,100)
            c = self.convmae.projection(self.convmae.forward_encoder(ck_img_cm))  # (B,100)
            feat_cm = torch.cat([h, c], dim=1)  # (B,200)

            # SupCon branch
            h = self.supcon(ht_img_sc)  # (B,100)
            c = self.supcon(ck_img_sc)  # (B,100)
            feat_sc = torch.cat([h, c], dim=1)  # (B,200)

        # simple average fusion
        fused = (feat_cm + feat_sc) / 2.0
        return self.classifier(fused)


# ================================================================
# 3. Dataset (matched HT + CK TIF pairs)
# ================================================================
class DualSourceDataset(Dataset):
    def __init__(self, ht_dir, ck_dir, band_indices, img_size=224):
        self.band_indices = band_indices
        self.img_size = img_size

        # collect labels from HT
        ht_files = {}
        unique_labels = set()
        for label_dir in sorted(os.listdir(ht_dir)):
            lp = os.path.join(ht_dir, label_dir)
            if not os.path.isdir(lp): continue
            unique_labels.add(label_dir)
            for fn in sorted(os.listdir(lp)):
                if fn.lower().endswith(('.tif', '.tiff')):
                    ht_files[os.path.splitext(fn)[0]] = (os.path.join(lp, fn), label_dir)

        self.label_to_idx = {l: i for i, l in enumerate(sorted(unique_labels))}
        self.num_classes = len(self.label_to_idx)

        # collect CK files
        ck_files = {}
        for label_dir in sorted(os.listdir(ck_dir)):
            lp = os.path.join(ck_dir, label_dir)
            if not os.path.isdir(lp): continue
            for fn in sorted(os.listdir(lp)):
                if fn.lower().endswith(('.tif', '.tiff')):
                    ck_files[os.path.splitext(fn)[0]] = os.path.join(lp, fn)

        self.pairs = []
        for fkey in sorted(set(ht_files) & set(ck_files)):
            ht_path, lbl = ht_files[fkey]
            self.pairs.append((ht_path, ck_files[fkey], self.label_to_idx[lbl], fkey + '.tif'))

        print(f"Dataset: {len(self.pairs)} pairs, {self.num_classes} classes → {self.label_to_idx}")

    def __len__(self):
        return len(self.pairs)

    def _read(self, path):
        with rasterio.open(path) as src:
            return src.read([i + 1 for i in self.band_indices])

    def _prep_convmae(self, raw):
        """bicubic resize, mean/std=[0.5], no /255"""
        img = raw.transpose(1, 2, 0).astype(np.float32)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        m = torch.tensor([0.5] * img.shape[0]).view(-1, 1, 1)
        s = torch.tensor([0.5] * img.shape[0]).view(-1, 1, 1)
        img = (img - m) / s
        if img.shape[1] != self.img_size or img.shape[2] != self.img_size:
            img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
                                mode='bicubic', align_corners=False).squeeze(0)
        return img

    def _prep_supcon(self, raw):
        """bilinear resize, /255, [-1,1]"""
        img = torch.from_numpy(raw.copy()).float()
        if img.max() > 1: img = img / 255.0
        if img.shape[1] != self.img_size or img.shape[2] != self.img_size:
            img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
                                mode='bilinear', align_corners=False).squeeze(0)
        m = torch.tensor([0.5] * img.shape[0]).view(-1, 1, 1)
        s = torch.tensor([0.5] * img.shape[0]).view(-1, 1, 1)
        return (img - m) / s

    def __getitem__(self, idx):
        ht_path, ck_path, label, fname = self.pairs[idx]
        ht_raw, ck_raw = self._read(ht_path), self._read(ck_path)
        return (self._prep_convmae(ht_raw), self._prep_convmae(ck_raw),
                self._prep_supcon(ht_raw), self._prep_supcon(ck_raw),
                label, fname)


# ================================================================
# 4. Load pretrained encoders
# ================================================================
def load_convmae(ckpt, device, in_chans=5):
    print(f"Loading ConvMAE: {ckpt}")
    model = ConvMAEFeatureExtractor(
        model_name='convmae_convvit_tiny_patch16', checkpoint_path=ckpt,
        in_chans=in_chans, input_size=224, selected_bands=list(range(in_chans)),
        device=str(device),
    )
    for p in model.parameters(): p.requires_grad = False
    model.eval()
    return model


def load_supcon(ckpt, device, in_chans=5, output_dim=100):
    print(f"Loading SupCon: {ckpt}")
    model = SupConModel(in_chans=in_chans, hidden_dim=256, output_dim=output_dim)
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd, strict=False)
    for p in model.parameters(): p.requires_grad = False
    model.eval()
    return model


# ================================================================
# 5. Trainer (same pattern as finetune_classifier_head.py)
# ================================================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model; self.train_loader = train_loader
        self.val_loader = val_loader; self.device = device; self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.classifier.parameters(),
                                           lr=config['lr'], weight_decay=config['wd'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        self.best_acc = 0.0
        self.train_hist = {'loss': [], 'acc': []}
        self.val_hist   = {'loss': [], 'acc': []}

    def _epoch(self, loader, training):
        self.model.train() if training else self.model.eval()
        ctx = torch.enable_grad() if training else torch.no_grad()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with ctx:
            for ht_cm, ck_cm, ht_sc, ck_sc, labels, _ in tqdm(loader, desc="Train" if training else "Val"):
                ht_cm, ck_cm = ht_cm.to(self.device), ck_cm.to(self.device)
                ht_sc, ck_sc = ht_sc.to(self.device), ck_sc.to(self.device)
                labels = labels.to(self.device)
                if training: self.optimizer.zero_grad()

                out = self.model(ht_cm, ck_cm, ht_sc, ck_sc)
                loss = self.criterion(out, labels)

                if training: loss.backward(); self.optimizer.step()

                total_loss += loss.item() * labels.size(0)
                _, pred = out.max(1)
                total += labels.size(0); correct += pred.eq(labels).sum().item()
                all_preds.extend(pred.cpu().numpy()); all_labels.extend(labels.cpu().numpy())

        return total_loss / total, 100. * correct / total, all_preds, all_labels

    def train(self):
        for epoch in range(1, self.config['epochs'] + 1):
            tr_loss, tr_acc, _, _ = self._epoch(self.train_loader, True)
            vl_loss, vl_acc, vl_p, vl_l = self._epoch(self.val_loader, False)
            self.scheduler.step()

            self.train_hist['loss'].append(tr_loss); self.train_hist['acc'].append(tr_acc)
            self.val_hist['loss'].append(vl_loss);   self.val_hist['acc'].append(vl_acc)

            print(f"Epoch {epoch:2d}/{self.config['epochs']}  "
                  f"train_loss={tr_loss:.4f} acc={tr_acc:.2f}%  "
                  f"val_loss={vl_loss:.4f} acc={vl_acc:.2f}%  "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e}")

            if vl_acc > self.best_acc:
                self.best_acc = vl_acc
                self._save('best_model.pth')
                print(f"  ✓ saved (acc={vl_acc:.2f}%)")

        print(f"\nDone! Best val acc: {self.best_acc:.2f}%")
        self._report(vl_p, vl_l)

    def _save(self, name):
        d = self.config['output_dir']; os.makedirs(d, exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict(),
                    'config': self.config, 'best_acc': self.best_acc}, os.path.join(d, name))

    def _report(self, preds, labels):
        d = self.config['output_dir']; os.makedirs(d, exist_ok=True)
        names = [str(i) for i in range(self.config['num_classes'])]

        report = classification_report(labels, preds, target_names=names)
        print("\n" + report)
        with open(os.path.join(d, 'classification_report.txt'), 'w') as f: f.write(report)

        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix'); plt.tight_layout()
        plt.savefig(os.path.join(d, 'confusion_matrix.png'), dpi=300); plt.close()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.train_hist['loss'], label='train'); ax1.plot(self.val_hist['loss'], label='val')
        ax1.set_title('Loss'); ax1.legend(); ax1.grid(True)
        ax2.plot(self.train_hist['acc'], label='train'); ax2.plot(self.val_hist['acc'], label='val')
        ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(d, 'training_curves.png'), dpi=300); plt.close()


# ================================================================
# 6. Main
# ================================================================
def main():
    args = get_args_parser().parse_args()
    config = config_from_args(args)

    torch.manual_seed(config['seed']); np.random.seed(config['seed']); random.seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    required_paths = ['train_ht', 'train_ck', 'val_ht', 'val_ck', 'convmae_ckpt', 'supcon_ckpt']
    missing_paths = [key for key in required_paths if not config[key]]
    if missing_paths:
        raise ValueError(f"请在 config 中设置路径: {', '.join(missing_paths)}")

    print("Loading dataset...")
    tr_ds = DualSourceDataset(config['train_ht'], config['train_ck'], config['band_indices'], config['img_size'])
    vl_ds = DualSourceDataset(config['val_ht'],   config['val_ck'],   config['band_indices'], config['img_size'])
    config['num_classes'] = tr_ds.num_classes

    tr_ld = DataLoader(tr_ds, batch_size=config['batch_size'], shuffle=True,
                       num_workers=config['num_workers'], pin_memory=True)
    vl_ld = DataLoader(vl_ds, batch_size=config['batch_size'], shuffle=False,
                       num_workers=config['num_workers'], pin_memory=True)

    cm_model = load_convmae(config['convmae_ckpt'], device, in_chans=len(config['band_indices']))
    sc_model = load_supcon(config['supcon_ckpt'],   device, in_chans=len(config['band_indices']))

    model = DualEncoderClassifier(cm_model, sc_model, config['num_classes'], config['dropout_rate'])
    model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_total:,} total | {n_train:,} trainable (classifier head only)")

    Trainer(model, tr_ld, vl_ld, device, config).train()


if __name__ == '__main__':
    main()
