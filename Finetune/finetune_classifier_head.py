import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from functools import partial
import rasterio
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# 导入原始的VisionTransformer模型
from models_vit import VisionTransformer


def parse_bands(value):
    return [int(item.strip()) for item in value.split(',') if item.strip()]


def get_args_parser():
    parser = argparse.ArgumentParser('ViT classifier fine-tuning')
    parser.add_argument('--train-data-dir', required=True, help='训练集目录，目录下按类别分文件夹存放 TIF')
    parser.add_argument('--val-data-dir', required=True, help='验证集目录，目录下按类别分文件夹存放 TIF')
    parser.add_argument('--pretrained-backbone-path', default='', help='可选：预训练编码器权重 .pth 路径')
    parser.add_argument('--bands', default='0,1,2,3,4', help='使用的波段索引，如 0,1,2,3,4')
    parser.add_argument('--num-classes', type=int, default=3, help='分类类别数')
    parser.add_argument('--dropout-rate', type=float, default=0.2, help='Dropout 率')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--num-epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--device', default='cuda', help='训练设备，如 cuda 或 cpu')
    parser.add_argument('--output-dir', default='./finetune_results', help='输出目录')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader worker 数')
    parser.add_argument('--target-size', type=int, default=224, help='输入图像边长')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser


def config_from_args(args):
    return {
        'train_data_dir': args.train_data_dir,
        'val_data_dir': args.val_data_dir,
        'pretrained_backbone_path': args.pretrained_backbone_path,
        'band_indices': parse_bands(args.bands),
        'num_classes': args.num_classes,
        'dropout_rate': args.dropout_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'device': args.device,
        'output_dir': args.output_dir,
        'num_workers': args.num_workers,
        'target_size': (args.target_size, args.target_size),
        'seed': args.seed
    }


class ViTFeatureExtractor(nn.Module):
    """ViT特征提取器 - 与训练代码保持一致"""
    def __init__(self, in_chans, pretrained=True):
        super(ViTFeatureExtractor, self).__init__()
        # ViT-Tiny 配置
        self.vit = VisionTransformer(
            embed_dim=192,       # ViT-Tiny 使用 192 维嵌入 
            depth=12,            # 12 个 transformer 层
            num_heads=3,         # 3 个注意力头
            mlp_ratio=4,
            qkv_bias=True,
            in_chans=in_chans,   # 动态设置输入通道数
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            global_pool=False
        )
        self.vit.head = nn.Identity()
        self.output_dim = 192    # ViT-Tiny的输出维度
        
    def forward(self, x):
        features = self.vit(x)
        if len(features.shape) == 2:
            cls_token_features = features
        else:
            cls_token_features = features[:, 0, :]  # 使用CLS token特征
        return cls_token_features


class ViTClassifier(nn.Module):
    """基于ViT的分类器，用于3分类任务"""
    def __init__(self, in_chans=4, num_classes=3, pretrained_backbone_path=None, dropout_rate=0.1):
        super(ViTClassifier, self).__init__()
        
        # 特征提取器
        self.feature_extractor = ViTFeatureExtractor(in_chans=in_chans, pretrained=True)
        
        # 分类头 - 可以根据需要调整结构
        self.classifier = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # 如果提供了预训练的特征提取器权重，加载它们
        if pretrained_backbone_path and os.path.exists(pretrained_backbone_path):
            self.load_pretrained_backbone(pretrained_backbone_path)
    
    def load_pretrained_backbone(self, pretrained_path):
        """从对比学习模型中加载特征提取器权重"""
        try:
            print(f"正在加载预训练特征提取器权重: {pretrained_path}")
            
            # 加载预训练权重
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # 获取当前模型的state_dict
            current_state_dict = self.state_dict()
            
            # 创建新的state_dict，只包含特征提取器的权重
            updated_state_dict = {}
            
            print("正在匹配权重...")
            for name, param in checkpoint.items():
                # 只加载特征提取器相关的权重，跳过投影头/分类头
                if name.startswith('feature_extractor.'):
                    if name in current_state_dict:
                        updated_state_dict[name] = param
                        print(f"成功匹配: {name}")
            
            # 加载权重（只加载特征提取器部分）
            missing_keys, unexpected_keys = self.load_state_dict(updated_state_dict, strict=False)
            
            print(f"成功加载 {len(updated_state_dict)} 个特征提取器权重")
            print(f"分类头使用随机初始化 ({len([k for k in missing_keys if 'classifier' in k])} 个参数)")
            
            if len(updated_state_dict) == 0:
                print("警告: 没有找到匹配的特征提取器权重")
                print("检查模型文件格式...")
                print("模型权重的键:", list(checkpoint.keys())[:10])
                
        except Exception as e:
            print(f"加载预训练权重时出错: {e}")
            print("将使用随机初始化的权重")
    
    def freeze_feature_extractor(self):
        """冻结特征提取器的参数"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        print("特征提取器参数已冻结")
    
    def unfreeze_feature_extractor(self):
        """解冻特征提取器的参数"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        print("特征提取器参数已解冻")
    
    def get_trainable_params(self):
        """获取可训练参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)
        
        # 分类
        logits = self.classifier(features)
        
        return logits


class ImageDataset(Dataset):
    """多波段遥感图像数据集"""
    def __init__(self, data_dir, band_indices, target_size=(224, 224), transform=None):
        self.data_dir = data_dir
        self.band_indices = band_indices
        self.target_size = target_size
        self.transform = transform
        
        # 获取所有类别文件夹
        self.class_folders = sorted([d for d in os.listdir(data_dir) 
                                   if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_folders)}
        
        # 收集所有图像路径和标签
        self.samples = []
        for class_name in self.class_folders:
            class_dir = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # 获取该类别下的所有.tif文件
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith('.tif'):
                    img_path = os.path.join(class_dir, img_file)
                    self.samples.append((img_path, class_idx))
        
        print(f"数据集加载完成: {len(self.samples)} 个样本")
        print(f"类别: {self.class_folders}")
        
        # 计算归一化参数
        self.mean = torch.tensor([0.5] * len(band_indices))
        self.std = torch.tensor([0.5] * len(band_indices))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # 使用rasterio读取多波段TIF图像
            with rasterio.open(img_path) as src:
                rasterio_indices = [i + 1 for i in self.band_indices]
                img = src.read(rasterio_indices)  # shape: (selected_bands, H, W)
            
            # 转换为torch tensor
            img = torch.from_numpy(img).float()
            
            # 归一化到0-1范围
            if img.max() > 1:
                img = img / 255.0
            
            # 调整大小
            if img.shape[1] != self.target_size[0] or img.shape[2] != self.target_size[1]:
                img = F.interpolate(
                    img.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False
                ).squeeze(0)
            
            # 应用归一化
            mean = self.mean.view(-1, 1, 1)
            std = self.std.view(-1, 1, 1)
            img = (img - mean) / std
            
            # 应用变换
            if self.transform:
                img = self.transform(img)
            
            return img, label
            
        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")
            # 返回一个默认的张量和标签
            return torch.zeros(len(self.band_indices), *self.target_size), label


class Trainer:
    """训练器类"""
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器 - 优化所有参数（端到端训练）
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['num_epochs']
        )
        
        # 记录训练历史
        self.train_history = {'loss': [], 'acc': []}
        self.val_history = {'loss': [], 'acc': []}
        self.best_val_acc = 0.0
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="训练中")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="验证中")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self):
        """完整训练过程"""
        print("开始训练...")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"验证样本: {len(self.val_loader.dataset)}")
        
        total_params, trainable_params = self.model.get_trainable_params()
        print(f"总参数数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            self.train_history['loss'].append(train_loss)
            self.train_history['acc'].append(train_acc)
            
            # 验证
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch()
            self.val_history['loss'].append(val_loss)
            self.val_history['acc'].append(val_acc)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"训练损失: {train_loss:.4f}, 训练精度: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证精度: {val_acc:.2f}%")
            print(f"学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(f"best_classifier_epoch_{epoch+1}.pth")
                print(f"保存最佳模型 (验证精度: {val_acc:.2f}%)")
        
        print(f"\n训练完成! 最佳验证精度: {self.best_val_acc:.2f}%")
        
        # 保存最终模型（最后一个epoch的模型）
        final_filename = f"final_model_epoch_{self.config['num_epochs']}.pth"
        os.makedirs(self.config['output_dir'], exist_ok=True)
        final_filepath = os.path.join(self.config['output_dir'], final_filename)
        
        torch.save({
            'model': self.model,  # 保存完整模型架构
            'model_state_dict': self.model.state_dict(),  # 保存权重
            'optimizer_state_dict': self.optimizer.state_dict(),  # 保存优化器状态
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_acc': self.best_val_acc,
            'final_val_acc': val_acc
        }, final_filepath)
        
        # 保存最终权重文件
        final_weights_filepath = final_filepath.replace('.pth', '_weights_only.pth')
        torch.save(self.model.state_dict(), final_weights_filepath)
        
        print(f"最终完整模型已保存: {final_filepath}")
        print(f"最终权重文件已保存: {final_weights_filepath}")
        
        # 生成最终验证报告
        self.generate_final_report(val_preds, val_labels)
    
    def save_model(self, filename):
        """保存模型"""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        filepath = os.path.join(self.config['output_dir'], filename)
        
        # 保存完整的模型架构和权重
        torch.save({
            'model': self.model,  # 保存完整模型架构
            'model_state_dict': self.model.state_dict(),  # 保存权重
            'optimizer_state_dict': self.optimizer.state_dict(),  # 保存优化器状态
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_acc': self.best_val_acc
        }, filepath)
        
        # 额外保存一个只包含权重的版本（兼容性）
        weights_filepath = filepath.replace('.pth', '_weights_only.pth')
        torch.save(self.model.state_dict(), weights_filepath)
        
        print(f"完整模型已保存: {filepath}")
        print(f"权重文件已保存: {weights_filepath}")
    
    def generate_final_report(self, val_preds, val_labels):
        """生成最终验证报告"""
        # 分类报告
        class_names = [str(i) for i in range(self.config['num_classes'])]
        report = classification_report(val_labels, val_preds, target_names=class_names)
        print("\n分类报告:")
        print(report)
        
        # 保存分类报告
        os.makedirs(self.config['output_dir'], exist_ok=True)
        with open(os.path.join(self.config['output_dir'], 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # 混淆矩阵
        cm = confusion_matrix(val_labels, val_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'confusion_matrix.png'), dpi=300)
        print(f"混淆矩阵已保存: {os.path.join(self.config['output_dir'], 'confusion_matrix.png')}")
        
        # 训练历史曲线
        self.plot_training_history()
    
    def plot_training_history(self):
        """绘制训练历史曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(self.train_history['loss'], label='训练损失')
        ax1.plot(self.val_history['loss'], label='验证损失')
        ax1.set_title('损失曲线')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_history['acc'], label='训练精度')
        ax2.plot(self.val_history['acc'], label='验证精度')
        ax2.set_title('准确率曲线')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'training_history.png'), dpi=300)
        print(f"训练历史曲线已保存: {os.path.join(self.config['output_dir'], 'training_history.png')}")


def main():
    """主函数"""
    print("="*60)
    print("ViT 端到端微调训练程序")
    print("="*60)
    args = get_args_parser().parse_args()
    config = config_from_args(args)
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    required_dirs = ['train_data_dir', 'val_data_dir']
    missing_dirs = [key for key in required_dirs if not config[key]]
    if missing_dirs:
        raise ValueError(f"请在 config 中设置路径: {', '.join(missing_dirs)}")
    
    # 设备设置
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建数据集
    print(f"\n正在加载数据集...")
    train_dataset = ImageDataset(
        config['train_data_dir'], 
        config['band_indices'], 
        config['target_size']
    )
    
    val_dataset = ImageDataset(
        config['val_data_dir'], 
        config['band_indices'], 
        config['target_size']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 创建模型
    print(f"\n正在初始化模型...")
    model = ViTClassifier(
        in_chans=len(config['band_indices']),
        num_classes=config['num_classes'],
        pretrained_backbone_path=config['pretrained_backbone_path'],
        dropout_rate=config['dropout_rate']
    )
    
    # 不冻结特征提取器，进行端到端训练
    model.unfreeze_feature_extractor()
    
    model.to(device)
    
    # 创建训练器并开始训练
    trainer = Trainer(model, train_loader, val_loader, device, config)
    trainer.train()
    
    print("\n训练完成!")


if __name__ == '__main__':
    main()
