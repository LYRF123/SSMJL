import argparse
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from functools import partial
import rasterio
from collections import Counter

import kornia.augmentation as K
import kornia   

import timm
import torchvision.models as models
from models_vit import VisionTransformer  # 保留自定义的 VisionTransformer 模型导入
from loss.supconloss import SupConLoss

#########################################
# 1. 参数解析
#########################################
def get_args_parser():
    parser = argparse.ArgumentParser('多波段监督对比学习', add_help=False)
    # 基本训练参数
    parser.add_argument('--batch_size', default=32, type=int,
                        help='批量大小')
    parser.add_argument('--epochs', default=20, type=int,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.00003,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='权重衰减')
    
    # 数据集参数
    parser.add_argument('--data_path', default='', type=str,
                        help='数据集路径')
    parser.add_argument('--output_dir', default='',
                        help='输出目录')
    parser.add_argument('--bands', type=str, default='0,1,2,3,4',
                        help='波段索引，格式如 0,1,2,3 表示使用第1-4个波段 (索引从0开始)')
    parser.add_argument('--train_val_split', type=float, default=0.8,
                        help='训练集比例，默认0.8，即80%训练，20%验证')
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='vit-tiny', 
                        choices=['resnet50', 'resnet18', 'vit', 'vit-tiny', 'vgg', 'vgg9'],
                        help='选择主干网络: resnet50, resnet18, vit, vit-tiny, vgg 或 vgg9')
    parser.add_argument('--mae_weights_path', type=str, default=r'',
                        help='MAE预训练模型权重路径，用于初始化ViT主干网络')
    parser.add_argument('--use_imagenet_pretrained', action='store_true', default=False,
                        help='是否使用ImageNet预训练权重（DeiT权重）初始化ViT-Tiny')
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='是否冻结编码器参数')
    parser.add_argument('--freeze_partial', action='store_true', default=False,
                        help='是否只冻结部分编码器 (而不是整个编码器)')
    parser.add_argument('--freeze_ratio', type=float, default=0,
                        help='冻结编码器的比例 (默认0.5表示冻结前半部分transformer块)')
    
    # 优化器和学习率调度相关参数
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='优化器类型: sgd 或 adam')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['cosine', 'reduce_on_plateau', 'none'],
                        help='学习率调度器: cosine(余弦退火), reduce_on_plateau(根据验证集性能), none(不使用)')
    parser.add_argument('--min_lr_ratio', type=float, default=0.001,
                        help='最小学习率与初始学习率的比例 (用于余弦退火调度器)')
    
    # 早停相关参数
    parser.add_argument('--patience', type=int, default=5,
                        help='早停耐心值: 验证集性能不再提升的轮数，超过此值则停止训练')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='最小提升值: 小于此值的性能提升视为无效提升')
    
    # 硬件和性能相关
    parser.add_argument('--device', default='cuda',
                        help='训练设备')
    parser.add_argument('--seed', default=0, type=int,
                        help='随机种子')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='数据加载线程数')
    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度训练加速')
    
    return parser

#########################################
# 2. 数据增强（使用 Kornia 进行多通道增强）
#########################################
# 定义可变通道数的归一化
def get_normalization_params(num_bands):
    return torch.tensor([0.5] * num_bands), torch.tensor([0.5] * num_bands)

class AugmentationPipeline(nn.Module):
    def __init__(self, num_bands):
        super().__init__()
        self.random_resized_crop = K.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))
        self.random_horizontal_flip = K.RandomHorizontalFlip(p=0.5)
        # 修改 ColorJitter：将 saturation 和 hue 设为 0，避免调用 rgb_to_hsv
        self.color_jitter = K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0)
        # 为当前波段数量获取归一化参数
        self.normalize_mean, self.normalize_std = get_normalization_params(num_bands)
    
    def forward(self, x):
        # x: Tensor, shape (C, H, W) —— 增加 batch 维度处理
        x = x.unsqueeze(0)  # (1, C, H, W)
        x = self.random_resized_crop(x)
        x = self.random_horizontal_flip(x)
        x = self.color_jitter(x)
        x = x.squeeze(0)
        # 手动归一化
        mean = self.normalize_mean.to(x.device).view(-1, 1, 1)
        std = self.normalize_std.to(x.device).view(-1, 1, 1)
        x = (x - mean) / std
        return x

class TwoCropTransform:
    """对同一图像生成两个增强视图"""
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        view1 = self.transform(x.clone())
        view2 = self.transform(x.clone())
        return [view1, view2]

#########################################
# 3. 数据集定义（使用选定波段读取TIF图像）
#########################################
class RasterioMultiChannelImageFolder(Dataset):
    def __init__(self, root, band_indices, transform=None):
        """
        Args:
          root: 数据集根目录路径（目录下包含"0"、"1"等类别文件夹）
          band_indices: 要使用的波段索引列表 [0, 1, 2, ...] (0-based)
          transform: 转换操作
        """
        self.root = root
        self.transform = transform
        self.band_indices = band_indices  # 0-based索引
        self.samples = []
        self.class_to_idx = {}
        self.class_counts = {}  # 存储每个类别的样本数量

        if not os.path.isdir(root):
            raise ValueError(f"数据集目录不存在: {root}")
        
        # 遍历类别文件夹
        for class_name in sorted(os.listdir(root)):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # 为类别分配索引
            idx = int(class_name) if class_name.isdigit() else len(self.class_to_idx)
            self.class_to_idx[class_name] = idx
            
            # 初始化类别计数
            self.class_counts[idx] = 0
            
            # 收集该类别下所有TIF文件
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.tif', '.tiff')):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, idx))
                    self.class_counts[idx] += 1

        if not self.samples:
            raise ValueError(
                f"数据集目录中没有找到 .tif/.tiff 图像: {root}。"
                "请按 类别名/*.tif 的结构组织数据。"
            )
    
    def __len__(self):
        return len(self.samples)
    
    def get_class_distribution(self):
        """返回类别分布情况"""
        total = len(self.samples)
        if total == 0:
            raise ValueError("数据集中没有样本，无法统计类别分布。")

        distribution = {}
        for class_name, idx in self.class_to_idx.items():
            count = self.class_counts[idx]
            percentage = (count / total) * 100
            distribution[class_name] = {
                'index': idx,
                'count': count,
                'percentage': percentage
            }
        return distribution
    
    def __getitem__(self, index):
        path, label = self.samples[index]
        
        # 读取选定波段
        with rasterio.open(path) as src:
            # 将0-based索引转换为1-based索引用于rasterio.read()
            rasterio_indices = [i + 1 for i in self.band_indices]
            img = src.read(rasterio_indices)  # shape: (selected_bands, H, W)
        
        img = torch.from_numpy(img).float()
        if img.max() > 1:
            img = img / 255.0
            
        # 确保图像大小为224x224
        if img.shape[1] != 224 or img.shape[2] != 224:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
            ).squeeze(0)
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

#########################################
# 4. 模型定义（使用自定义 VisionTransformer 及投影头）
#########################################
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, in_chans, pretrained=True, model_type='resnet50'):
        super(ResNetFeatureExtractor, self).__init__()
        # 根据model_type选择合适的ResNet模型
        if model_type == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
            self.output_dim = 2048  # ResNet50的输出维度
        elif model_type == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
            self.output_dim = 512   # ResNet18的输出维度
        
        # 修改第一层卷积以适应不同的输入通道数
        if in_chans != 3:
            self.resnet.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 移除分类头，仅保留特征提取器部分
        self.resnet.fc = nn.Identity()
        self.model_type = model_type
        
    def forward(self, x):
        return self.resnet(x)

class ViTFeatureExtractor(nn.Module):
    def __init__(self, in_chans, pretrained=True, mae_weights_path=None):
        super(ViTFeatureExtractor, self).__init__()
        # 替换为 ViT-Tiny 配置
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
        
        # 加载MAE预训练权重（如果提供）
        if mae_weights_path is not None and os.path.exists(mae_weights_path):
            print(f"正在从MAE预训练模型加载权重: {mae_weights_path}")
            
            try:
                # 加载MAE模型权重
                checkpoint = torch.load(mae_weights_path, map_location='cpu')
                
                # 打印顶层键以进行调试
                print("MAE checkpoint包含以下键:")
                if isinstance(checkpoint, dict):
                    for key in checkpoint.keys():
                        print(f"  - {key}")
                    
                    # 提取state_dict
                    if 'model' in checkpoint:
                        mae_state_dict = checkpoint['model']
                    elif 'model_state_dict' in checkpoint:
                        mae_state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        mae_state_dict = checkpoint['state_dict']
                    else:
                        # 尝试直接使用checkpoint作为state_dict
                        mae_state_dict = checkpoint
                else:
                    print("加载的checkpoint不是字典类型")
                    mae_state_dict = checkpoint
                
                # 打印MAE权重的所有键进行调试
                print("MAE模型权重包含以下键:")
                if isinstance(mae_state_dict, dict):
                    mae_keys = list(sorted(mae_state_dict.keys()))
                    for i, key in enumerate(mae_keys[:10]):  # 只打印前10个键以节省输出
                        print(f"  - {key}")
                    if len(mae_keys) > 10:
                        print(f"  - ... 以及更多 ({len(mae_keys)} 总数)")
                else:
                    print("MAE state_dict不是字典类型")
                    return
                
                # 获取VIT模型的现有键
                vit_state_dict = self.vit.state_dict()
                vit_keys = list(vit_state_dict.keys())
                print(f"VisionTransformer模型总共有 {len(vit_keys)} 个参数")
                
                # 打印VIT键作为参考
                print("VIT模型前10个参数键:")
                for key in vit_keys[:min(10, len(vit_keys))]:
                    print(f"  - {key}")
                
                # 创建映射字典
                updated_state_dict = {}
                
                # 专注于从MAE提取编码器部分的方法
                # 1. 直接映射常见参数
                encoder_param_mapping = {
                    'encoder.cls_token': 'cls_token',
                    'encoder.pos_embed': 'pos_embed',
                    'encoder.patch_embed.proj.weight': 'patch_embed.proj.weight',
                    'encoder.patch_embed.proj.bias': 'patch_embed.proj.bias'
                }
                
                # 直接映射块参数的模式
                block_pattern_mapping = {
                    'encoder.blocks.': 'blocks.'
                }
                
                # 应用直接映射
                for mae_key, vit_key in encoder_param_mapping.items():
                    if mae_key in mae_state_dict and vit_key in vit_keys:
                        updated_state_dict[vit_key] = mae_state_dict[mae_key]
                        print(f"直接映射: {mae_key} -> {vit_key}")
                
                # 应用模式替换搜索块参数
                for mae_pattern, vit_pattern in block_pattern_mapping.items():
                    for mae_key in mae_keys:
                        if mae_pattern in mae_key:
                            # 替换模式
                            vit_key = mae_key.replace(mae_pattern, vit_pattern)
                            
                            # 检查是否是VIT键
                            if vit_key in vit_keys:
                                updated_state_dict[vit_key] = mae_state_dict[mae_key]
                
                # 计算成功映射的参数数量
                mapped_keys = len(updated_state_dict)
                print(f"成功映射了 {mapped_keys}/{len(vit_keys)} 个参数")
                
                # 创建反向映射字典，记录每个VIT参数对应的MAE参数
                source_mapping = {}
                for mae_key, vit_key in encoder_param_mapping.items():
                    if vit_key in updated_state_dict:
                        source_mapping[vit_key] = mae_key
                
                # 记录block模式替换的映射
                for mae_pattern, vit_pattern in block_pattern_mapping.items():
                    for mae_key in mae_keys:
                        if mae_pattern in mae_key:
                            vit_key = mae_key.replace(mae_pattern, vit_pattern)
                            if vit_key in updated_state_dict:
                                source_mapping[vit_key] = mae_key
                
                # 打印成功映射的参数清单
                print("\n成功映射的参数 (全部列表):")
                for i, (vit_key, tensor) in enumerate(sorted(updated_state_dict.items())):
                    mae_key = source_mapping.get(vit_key, "未记录来源")
                    tensor_shape = tuple(tensor.shape)
                    print(f"  - VIT参数: {vit_key} ({tensor_shape}) <- MAE参数: {mae_key}")
                
                # 如果找不到足够的映射键，尝试更通用的方法
                if mapped_keys < len(vit_keys) // 2:
                    print("映射参数不足，尝试使用更通用的方法...")
                    
                    # 检查MAE键是否包含encoder
                    has_encoder_prefix = any('encoder.' in k for k in mae_keys)
                    
                    if has_encoder_prefix:
                        print("检测到encoder前缀，尝试直接提取encoder部分")
                        # 对于所有encoder开头的键
                        for mae_key in mae_keys:
                            if mae_key.startswith('encoder.'):
                                # 移除encoder前缀
                                vit_key = mae_key[8:]  # 'encoder.' 是8个字符
                                if vit_key in vit_keys and vit_key not in updated_state_dict:
                                    updated_state_dict[vit_key] = mae_state_dict[mae_key]
                                    source_mapping[vit_key] = mae_key  # 记录来源
                    else:
                        print("未检测到encoder前缀，尝试直接匹配键")
                        # 直接查找匹配的键
                        for vit_key in vit_keys:
                            if vit_key in mae_keys and vit_key not in updated_state_dict:
                                updated_state_dict[vit_key] = mae_state_dict[vit_key]
                                source_mapping[vit_key] = vit_key  # 记录来源
                
                # 如果还是映射不够，使用后缀匹配
                mapped_keys = len(updated_state_dict)
                if mapped_keys < len(vit_keys) // 2:
                    print("尝试使用后缀匹配...")
                    # 使用后缀匹配
                    for vit_key in vit_keys:
                        if vit_key not in updated_state_dict:  # 只处理尚未映射的键
                            parts = vit_key.split('.')
                            # 尝试匹配最后3个部分（通常足够唯一识别参数）
                            if len(parts) >= 3:
                                suffix = '.'.join(parts[-3:])
                                for mae_key in mae_keys:
                                    if mae_key.endswith(suffix):
                                        updated_state_dict[vit_key] = mae_state_dict[mae_key]
                                        source_mapping[vit_key] = mae_key  # 记录来源
                                        break
                
                # 最终映射结果
                mapped_keys = len(updated_state_dict)
                print(f"最终映射了 {mapped_keys}/{len(vit_keys)} 个参数")
                
                # 打印最终映射的参数清单
                print("\n最终成功映射的参数 (全部列表):")
                for i, (vit_key, tensor) in enumerate(sorted(updated_state_dict.items())):
                    mae_key = source_mapping.get(vit_key, "未记录来源")
                    tensor_shape = tuple(tensor.shape)
                    print(f"  - VIT参数: {vit_key} ({tensor_shape}) <- MAE参数: {mae_key}")
                
                # 尝试加载权重
                if mapped_keys > 0:
                    msg = self.vit.load_state_dict(updated_state_dict, strict=False)
                    print(f"加载结果: {msg}")
                    
                    # 打印所有未加载的参数详情
                    if len(msg.missing_keys) > 0:
                        missing_count = len(msg.missing_keys)
                        print(f"未加载的参数 (全部 {missing_count}个):")
                        for i, key in enumerate(sorted(msg.missing_keys)):
                            # 如果参数在VIT模型中存在，打印其形状
                            if key in vit_state_dict:
                                tensor_shape = tuple(vit_state_dict[key].shape)
                                print(f"  - {key} ({tensor_shape})")
                            else:
                                print(f"  - {key} (形状未知)")
                
                    if len(msg.unexpected_keys) > 0:
                        print(f"意外的参数 ({len(msg.unexpected_keys)}个):")
                        for key in sorted(msg.unexpected_keys):
                            print(f"  - {key}")
                else:
                    print("未能成功映射任何参数，检查MAE模型与VIT模型的兼容性")
                    
            except Exception as e:
                print(f"加载MAE权重时发生错误: {e}")
                import traceback
                traceback.print_exc()
        
    def forward(self, x):
        features = self.vit(x)
        if len(features.shape) == 2:
            cls_token_features = features
        else:
            cls_token_features = features[:, 0, :]
        return cls_token_features

class VGGFeatureExtractor(nn.Module):
    def __init__(self, in_chans, pretrained=True):
        super(VGGFeatureExtractor, self).__init__()
        # 使用VGG16作为特征提取器
        self.vgg = models.vgg16(pretrained=pretrained)
        self.output_dim = 4096  # VGG16的倒数第二层输出维度
        
        # 修改第一层卷积以适应不同的输入通道数
        if in_chans != 3:
            self.vgg.features[0] = nn.Conv2d(in_chans, 64, kernel_size=3, padding=1)
        
        # 移除分类头，仅保留特征提取器部分
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1])
        
    def forward(self, x):
        return self.vgg(x)

class VGG9FeatureExtractor(nn.Module):
    def __init__(self, in_chans, pretrained=True):
        super(VGG9FeatureExtractor, self).__init__()
        # 定义VGG9的架构
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(in_chans, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第四个卷积块
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第五个卷积块
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        
        self.output_dim = 4096  # VGG9的输出维度
        
        # 如果使用预训练权重，加载VGG16的权重并调整
        if pretrained:
            vgg16 = models.vgg16(pretrained=True)
            # 复制前9个卷积层的权重
            for i in range(9):
                if isinstance(self.features[i], nn.Conv2d):
                    self.features[i].weight.data = vgg16.features[i].weight.data.clone()
                    self.features[i].bias.data = vgg16.features[i].bias.data.clone()
            # 复制前两个全连接层的权重
            for i in range(4):
                if isinstance(self.classifier[i], nn.Linear):
                    self.classifier[i].weight.data = vgg16.classifier[i].weight.data.clone()
                    self.classifier[i].bias.data = vgg16.classifier[i].bias.data.clone()
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class FeatureExtractorWithProjection(nn.Module):
    def __init__(self, backbone='vit', in_chans=3, pretrained=True, hidden_dim=256, output_dim=100, mae_weights_path=None, freeze_encoder=False, freeze_partial=False, freeze_ratio=0.5, use_imagenet_pretrained=True):
        super(FeatureExtractorWithProjection, self).__init__()
        
        # 根据选择的主干网络初始化特征提取器
        if backbone == 'resnet50':
            self.feature_extractor = ResNetFeatureExtractor(in_chans=in_chans, pretrained=pretrained, model_type='resnet50')
            input_dim = 2048  # ResNet50的输出维度
        elif backbone == 'resnet18':
            self.feature_extractor = ResNetFeatureExtractor(in_chans=in_chans, pretrained=pretrained, model_type='resnet18')
            input_dim = 512   # ResNet18的输出维度
        elif backbone == 'vit' or backbone == 'vit-tiny':
            # 当不冻结参数且没有MAE权重且启用ImageNet预训练时，使用DeiT的imagenet预训练权重
            if not freeze_encoder and (mae_weights_path is None or not os.path.exists(mae_weights_path)) and use_imagenet_pretrained:
                print("使用DeiT的ViT-Tiny ImageNet预训练权重")
                
                # 创建自定义的ViT特征提取器
                self.feature_extractor = ViTFeatureExtractor(in_chans=in_chans, pretrained=False)
                
                # ViT-Tiny预训练权重URL
                vit_tiny_url = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
                
                try:
                    # 下载预训练权重
                    import urllib.request
                    import tempfile
                    
                    # 创建临时文件下载权重
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                        temp_path = tmp_file.name
                    
                    urllib.request.urlretrieve(vit_tiny_url, temp_path)
                    print(f"已下载DeiT ViT-Tiny预训练权重到: {temp_path}")
                    
                    # 加载预训练权重
                    pretrained_dict = torch.load(temp_path, map_location='cpu')
                    if 'model' in pretrained_dict:
                        pretrained_dict = pretrained_dict['model']
                    
                    # 获取当前模型权重
                    model_dict = self.feature_extractor.vit.state_dict()
                    
                    # 如果输入通道数不是3，需要修改第一层卷积
                    if in_chans != 3:
                        # 获取原始第一层卷积权重
                        orig_conv_key = 'patch_embed.proj.weight'
                        orig_conv_weight = pretrained_dict[orig_conv_key]
                        
                        # 创建新的卷积权重
                        if in_chans > 3:
                            # 复制前3个通道，对其余通道使用这3个通道的平均值
                            new_conv_weight = torch.zeros(
                                orig_conv_weight.shape[0],  # 输出通道数
                                in_chans,                   # 输入通道数
                                orig_conv_weight.shape[2],  # 核大小
                                orig_conv_weight.shape[3],  # 核大小
                                device=orig_conv_weight.device
                            )
                            new_conv_weight[:, :3, :, :] = orig_conv_weight
                            
                            # 对于额外的通道，使用RGB通道的平均值
                            channel_mean = orig_conv_weight.mean(dim=1, keepdim=True)
                            for c in range(3, in_chans):
                                new_conv_weight[:, c, :, :] = channel_mean.squeeze(1)
                        else:
                            # 如果通道数小于3，只保留前几个通道
                            new_conv_weight = orig_conv_weight[:, :in_chans, :, :]
                        
                        # 更新权重字典中的第一层卷积权重
                        pretrained_dict[orig_conv_key] = new_conv_weight
                    
                    # 过滤不匹配的键
                    # 1. 排除分类头
                    # 2. 排除与当前模型不匹配的键
                    filtered_dict = {k: v for k, v in pretrained_dict.items() 
                                    if k in model_dict and 'head' not in k}
                    
                    # 更新模型权重
                    model_dict.update(filtered_dict)
                    self.feature_extractor.vit.load_state_dict(model_dict, strict=False)
                    
                    # 清理临时文件
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    
                    print(f"成功加载DeiT ViT-Tiny预训练权重, 适应{in_chans}个输入通道")
                    
                except Exception as e:
                    print(f"加载预训练权重失败: {e}")
                    print("使用随机初始化权重")
            else:
                # 使用原来的MAE预训练权重或不使用预训练
                if not use_imagenet_pretrained:
                    print("禁用ImageNet预训练权重，使用随机初始化")
                self.feature_extractor = ViTFeatureExtractor(in_chans=in_chans, pretrained=pretrained, mae_weights_path=mae_weights_path)
            
            input_dim = 192  # ViT-Tiny的输出维度
        elif backbone == 'vgg':
            self.feature_extractor = VGGFeatureExtractor(in_chans=in_chans, pretrained=pretrained)
            input_dim = 4096  # VGG16的倒数第二层输出维度
        elif backbone == 'vgg9':
            self.feature_extractor = VGG9FeatureExtractor(in_chans=in_chans, pretrained=pretrained)
            input_dim = 4096  # VGG9的输出维度
        else:
            raise ValueError(f"不支持的主干网络: {backbone}")
            
        self.projection_head = ProjectionHead(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.backbone = backbone
        
        # 冻结参数
        if freeze_encoder:
            if freeze_partial and backbone == 'vit':
                self._freeze_partial_encoder(freeze_ratio)
                print(f"已冻结{backbone}特征提取器的前{int(freeze_ratio*100)}%参数")
            else:
                self._freeze_feature_extractor()
                print(f"已冻结{backbone}特征提取器的所有参数，只训练投影头")
    
    def _freeze_feature_extractor(self):
        """冻结特征提取器的所有参数"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def _freeze_partial_encoder(self, ratio=0.5):
        """只冻结部分编码器 (针对ViT设计)
        
        Args:
            ratio: 冻结的比例 (0.5表示冻结前一半的transformer块)
        """
        if hasattr(self.feature_extractor, 'vit') and hasattr(self.feature_extractor.vit, 'blocks'):
            # 冻结embedding layer和cls token
            if hasattr(self.feature_extractor.vit, 'patch_embed'):
                for param in self.feature_extractor.vit.patch_embed.parameters():
                    param.requires_grad = False
                print("已冻结patch embedding层")
            
            if hasattr(self.feature_extractor.vit, 'cls_token'):
                self.feature_extractor.vit.cls_token.requires_grad = False
                print("已冻结cls_token")
            
            if hasattr(self.feature_extractor.vit, 'pos_embed'):
                self.feature_extractor.vit.pos_embed.requires_grad = False
                print("已冻结pos_embed")
            
            # 冻结前一半transformer块
            blocks = self.feature_extractor.vit.blocks
            num_blocks = len(blocks)
            freeze_blocks = int(num_blocks * ratio)
            
            for i in range(freeze_blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = False
            
            print(f"已冻结前{freeze_blocks}/{num_blocks}个transformer块")
        else:
            print("模型结构不支持部分冻结，将冻结全部特征提取器")
            self._freeze_feature_extractor()
    
    def unfreeze_feature_extractor(self):
        """解冻特征提取器的所有参数"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        print(f"已解冻特征提取器的所有参数")
    
    def print_trainable_parameters(self):
        """打印可训练参数的数量和比例"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params:.2%})")
        print(f"冻结参数: {frozen_params:,} ({frozen_params/total_params:.2%})")
        
        # 打印特征提取器和投影头的参数数量
        extractor_params = sum(p.numel() for p in self.feature_extractor.parameters())
        extractor_trainable = sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)
        projection_params = sum(p.numel() for p in self.projection_head.parameters())
        
        print(f"特征提取器参数: {extractor_params:,} ({extractor_params/total_params:.2%})")
        print(f"特征提取器可训练参数: {extractor_trainable:,} ({extractor_trainable/extractor_params:.2%})")
        print(f"投影头参数: {projection_params:,} ({projection_params/total_params:.2%})")
        
        # 如果是ViT，打印每一层的冻结情况
        if hasattr(self.feature_extractor, 'vit') and hasattr(self.feature_extractor.vit, 'blocks'):
            blocks = self.feature_extractor.vit.blocks
            print("\nViT各transformer块训练状态:")
            for i, block in enumerate(blocks):
                block_params = sum(p.numel() for p in block.parameters())
                block_trainable = sum(p.numel() for p in block.parameters() if p.requires_grad)
                status = "可训练" if block_trainable == block_params else "已冻结"
                print(f"  - Block {i}: {status} ({block_trainable}/{block_params} 参数可训练)")
        
    def forward(self, x):
        features = self.feature_extractor(x)
        projected_features = self.projection_head(features)
        return projected_features
    
    
def contrastive_loss(z1, z2, temperature=0.05):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(sim.size(0)).to(sim.device)
    loss = F.cross_entropy(sim, labels)
    return loss

#########################################
# 辅助函数：计算模型参数数量和打印模型结构
#########################################
def count_parameters(model):
    """计算模型的参数数量（百万）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def print_model_summary(model, input_size=(1, 4, 224, 224), device='cuda'):
    """打印模型结构摘要"""
    try:
        from torchsummary import summary
        if device == 'cuda':
            model = model.cuda()
        summary(model, input_size=input_size[1:])
    except ImportError:
        print("请安装 torchsummary 包以显示详细模型结构。")
        print(f"模型结构: {model}")

def print_class_distribution(dataset):
    """打印类别分布情况"""
    distribution = dataset.get_class_distribution()
    total = len(dataset)
    
    print("\n" + "="*50)
    print(" "*15 + "数据集类别分布")
    print("="*50)
    print(f"总样本数量: {total}")
    print("-"*50)
    print(f"{'类别名称':<10}{'索引':<8}{'样本数量':<12}{'比例 (%)':<10}")
    print("-"*50)
    
    # 按类别索引排序
    for class_name, info in sorted(distribution.items(), key=lambda x: x[1]['index']):
        print(f"{class_name:<10}{info['index']:<8}{info['count']:<12}{info['percentage']:.2f}%")
    
    print("="*50)
    
    return distribution

def visualize_class_distribution(distribution, save_path=None):
    """可视化类别分布情况"""
    class_names = []
    counts = []
    
    for class_name, info in sorted(distribution.items(), key=lambda x: x[1]['index']):
        class_names.append(class_name)
        counts.append(info['count'])
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, counts, color='skyblue')
    
    # 在条形上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{int(height)}', ha='center', va='bottom')
    
    # 使用英文标签避免中文字体问题
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Sample Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"创建输出目录: {save_dir}")
            
        plt.savefig(save_path, dpi=300)
        print(f"类别分布图已保存至: {save_path}")
    
    return plt.gcf()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if not args.data_path:
        raise ValueError("请通过 --data_path 指定数据集路径")
    if not args.output_dir:
        raise ValueError("请通过 --output_dir 指定输出目录")
    if not 0 < args.train_val_split < 1:
        raise ValueError("--train_val_split 必须在 0 和 1 之间")
    
    # 解析波段选择参数
    band_indices = [int(b) for b in args.bands.split(',')]
    num_bands = len(band_indices)
    
    # 检查波段索引是否有效
    if any(i < 0 or i > 9 for i in band_indices):
        raise ValueError("波段索引必须在0-9之间")
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"创建输出目录: {args.output_dir}")
    
    # 数据集根目录（使用命令行参数中的路径）
    dataset_root = args.data_path
    
    # 创建适应当前波段数的增强
    transform = TwoCropTransform(AugmentationPipeline(num_bands=num_bands))
    
    # 使用动态波段选择的数据集
    full_dataset = RasterioMultiChannelImageFolder(
        root=dataset_root, 
        band_indices=band_indices, 
        transform=transform
    )
    
    # 打印并可视化类别分布
    bands_str = '_'.join(map(str, band_indices))
    distribution = print_class_distribution(full_dataset)
    distribution_fig = visualize_class_distribution(
        distribution, save_path=os.path.join(args.output_dir, f'class_distribution_bands_{bands_str}.png')
    )
    
    # 按8:2划分训练集和验证集
    train_size = int(len(full_dataset) * args.train_val_split)
    val_size = len(full_dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError(
            f"数据集划分后训练集/验证集为空: train={train_size}, val={val_size}。"
            "请增加样本数量或调整 --train_val_split。"
        )

    dataset_train, dataset_val = random_split(full_dataset, [train_size, val_size])
    
    print(f"\n训练集样本数: {len(dataset_train)}")
    print(f"验证集样本数: {len(dataset_val)}")
    
    # 创建数据加载器
    data_loader_train = DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        drop_last=len(dataset_train) >= args.batch_size,
        pin_memory=True
    )
    
    data_loader_val = DataLoader(
        dataset_val, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        drop_last=False, 
        pin_memory=True
    )
    
    # 设备选择
    requested_device = torch.device(args.device)
    if requested_device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError("请求使用 CUDA，但当前环境不可用。请改用 --device cpu。")
    device = str(requested_device)
    
    # 初始化模型时传入通道数和主干网络类型
    if args.backbone == 'vit-tiny':
        model = FeatureExtractorWithProjection(
            backbone='vit',  # 仍使用vit，但ViTFeatureExtractor内部已修改为tiny配置
            in_chans=num_bands, 
            pretrained=True,
            mae_weights_path=args.mae_weights_path,  # 传递MAE预训练权重路径
            freeze_encoder=args.freeze_encoder,  # 从命令行参数获取是否冻结编码器
            freeze_partial=args.freeze_partial,  # 从命令行参数获取是否冻结编码器
            freeze_ratio=args.freeze_ratio,  # 从命令行参数获取冻结比例
            use_imagenet_pretrained=args.use_imagenet_pretrained  # 从命令行参数获取是否启用ImageNet预训练
        )
    else:
        model = FeatureExtractorWithProjection(
            backbone=args.backbone,
            in_chans=num_bands, 
            pretrained=False,
            freeze_encoder=args.freeze_encoder,  # 从命令行参数获取是否冻结编码器
            freeze_partial=args.freeze_partial,  # 从命令行参数获取是否冻结编码器
            freeze_ratio=args.freeze_ratio,  # 从命令行参数获取冻结比例
            use_imagenet_pretrained=args.use_imagenet_pretrained  # 从命令行参数获取是否启用ImageNet预训练
        )
    model.to(device)
    
    # 计算参数数量
    model_params = count_parameters(model)
    
    # 打印可训练参数情况
    print("\n" + "="*50)
    print(" "*15 + "参数统计")
    print("="*50)
    model.print_trainable_parameters()
    print("="*50)
    
    # 打印训练配置信息
    print("\n" + "="*50)
    print(" "*15 + "训练配置信息")
    print("="*50)
    print(f"设备: {device}")
    print(f"主干网络: {args.backbone}")
    print(f"选择的波段索引: {band_indices}")
    print(f"波段数量: {num_bands}")
    print(f"MAE预训练权重: {args.mae_weights_path if args.mae_weights_path else '未使用'}")
    print(f"使用ImageNet预训练权重: {'是' if args.use_imagenet_pretrained else '否'}")
    
    # 更新编码器冻结状态的显示信息
    if args.freeze_encoder:
        if args.freeze_partial:
            freeze_status = f"部分冻结 (前{int(args.freeze_ratio*100)}%的transformer块)"
        else:
            freeze_status = "完全冻结，只训练投影头"
    else:
        freeze_status = "未冻结，整体训练"
    print(f"编码器状态: {freeze_status}")
    
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"总样本数: {len(full_dataset)}")
    print(f"训练集样本数: {len(dataset_train)}")
    print(f"验证集样本数: {len(dataset_val)}")
    print(f"每轮训练迭代次数: {len(data_loader_train)}")
    print(f"每轮验证迭代次数: {len(data_loader_val)}")
    print(f"类别数量: {len(full_dataset.class_to_idx)}")
    print(f"模型参数量: {model_params:.2f}M")
    print(f"早停耐心值: {args.patience}")
    print(f"最小提升值: {args.min_delta}")
    print("="*50)
    
    # 打印模型结构
    print("\n" + "="*50)
    print(" "*15 + "模型架构")
    print("="*50)
    
    # 使用详细的模型结构打印
    input_size = (1, num_bands, 224, 224)
    print_model_summary(model, input_size=input_size, device=device)
    
    print("="*50 + "\n")
    
    # 设置训练参数
    criterion = SupConLoss(temperature=0.07)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    # 根据参数选择学习率调度器
    if args.scheduler == 'cosine':
        min_lr = args.lr * args.min_lr_ratio
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=min_lr
        )
        print(f"使用余弦退火学习率调度器，最小学习率: {min_lr:.6f}")
    elif args.scheduler == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        print("使用按验证集性能调整的学习率调度器")
    else:
        scheduler = None
        print("不使用学习率调度器")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建包含波段和主干网络信息的模型保存路径
    best_model_path = os.path.join(args.output_dir, f'best_model_{args.backbone}_bands_{bands_str}.pth')
    final_model_path = os.path.join(args.output_dir, f'final_model_{args.backbone}_bands_{bands_str}.pth')
    
    # 创建日志文件名
    log_file_name = os.path.join(args.output_dir, f'training_losses_{args.backbone}_bands_{bands_str}.txt')
    
    # 初始化最佳验证损失和对应的epoch
    best_val_loss = float('inf')
    best_epoch = -1
    
    # 早停相关变量
    patience_counter = 0  # 当前无改进的轮数
    
    # 记录训练和验证损失
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 添加训练起始时间记录
    import time
    import datetime
    
    # 获取当前时间作为训练起始时间
    start_time = time.time()
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"训练开始时间: {start_time_str}")
    
    with open(log_file_name, "w") as loss_file:
        # 记录配置信息到日志
        loss_file.write(f"===== 训练配置 =====\n")
        loss_file.write(f"训练开始时间: {start_time_str}\n")
        loss_file.write(f"主干网络: {args.backbone}\n")
        loss_file.write(f"训练使用波段: {band_indices}\n")
        loss_file.write(f"波段数量: {num_bands}\n")
        loss_file.write(f"MAE预训练权重: {args.mae_weights_path if args.mae_weights_path else '未使用'}\n")
        loss_file.write(f"使用ImageNet预训练权重: {'是' if args.use_imagenet_pretrained else '否'}\n")
        
        # 更新编码器冻结状态的记录
        if args.freeze_encoder:
            if args.freeze_partial:
                freeze_status = f"部分冻结 (前{int(args.freeze_ratio*100)}%的transformer块)"
            else:
                freeze_status = "完全冻结，只训练投影头"
        else:
            freeze_status = "未冻结，整体训练"
        loss_file.write(f"编码器状态: {freeze_status}\n")
        
        loss_file.write(f"模型参数量: {model_params:.2f}M\n")
        loss_file.write(f"批量大小: {args.batch_size}\n")
        loss_file.write(f"学习率: {args.lr}\n")
        loss_file.write(f"总样本数: {len(full_dataset)}\n")
        loss_file.write(f"训练集样本数: {len(dataset_train)}\n")
        loss_file.write(f"验证集样本数: {len(dataset_val)}\n")
        loss_file.write(f"早停耐心值: {args.patience}\n")
        loss_file.write(f"最小提升值: {args.min_delta}\n\n")
        
        # 记录类别分布
        loss_file.write(f"===== 类别分布 =====\n")
        loss_file.write(f"{'类别名称':<10}{'索引':<8}{'样本数量':<12}{'比例 (%)':<10}\n")
        for class_name, info in sorted(distribution.items(), key=lambda x: x[1]['index']):
            loss_file.write(f"{class_name:<10}{info['index']:<8}{info['count']:<12}{info['percentage']:.2f}%\n")
        loss_file.write("\n")
        
        # 记录训练过程
        loss_file.write(f"===== 训练过程 =====\n")
        
        num_epochs = args.epochs
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_total_loss = 0.0
            train_total_samples = 0
            batch_losses = []  # 记录每个batch的损失
            
            for idx, (images, labels) in tqdm(enumerate(data_loader_train), total=len(data_loader_train),
                                             desc=f'Train Epoch {epoch+1}/{num_epochs}'):
                # images 为列表，包含两个视图
                images = torch.cat([images[0], images[1]], dim=0)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                bsz = labels.shape[0]
                features = model(images)
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                
                loss = criterion(features, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item() * bsz
                train_total_loss += batch_loss
                train_total_samples += bsz
                batch_losses.append(batch_loss / bsz)  # 记录当前batch的平均损失
                
                # 每20个batch打印一次学习率和损失信息
                if (idx + 1) % 20 == 0:
                    # 获取当前学习率
                    current_lr = optimizer.param_groups[0]['lr']
                    # 计算最近20个batch的平均损失
                    recent_batch_loss = sum(batch_losses[-20:]) / min(20, len(batch_losses[-20:]))
                    # 计算到目前为止的总平均损失
                    avg_loss_so_far = train_total_loss / train_total_samples
                    
                    batch_info = f"Epoch [{epoch+1}/{num_epochs}], Batch [{idx+1}/{len(data_loader_train)}], " \
                                f"LR: {current_lr:.6f}, Recent 20-Batch Loss: {recent_batch_loss:.4f}, " \
                                f"Avg Loss So Far: {avg_loss_so_far:.4f}"
                    print(batch_info)
                    loss_file.write(batch_info + "\n")
            
            train_avg_loss = train_total_loss / train_total_samples
            train_losses.append(train_avg_loss)
            
            # 打印当前epoch结束后的学习率
            current_lr = optimizer.param_groups[0]['lr']
            lr_info = f"Epoch [{epoch+1}/{num_epochs}] 结束, 当前学习率: {current_lr:.6f}"
            print(lr_info)
            loss_file.write(lr_info + "\n")
            
            # 验证阶段
            model.eval()
            val_total_loss = 0.0
            val_total_samples = 0
            
            with torch.no_grad():
                for idx, (images, labels) in tqdm(enumerate(data_loader_val), total=len(data_loader_val),
                                                 desc=f'Val Epoch {epoch+1}/{num_epochs}'):
                    # images 为列表，包含两个视图
                    images = torch.cat([images[0], images[1]], dim=0)
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    bsz = labels.shape[0]
                    features = model(images)
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    
                    loss = criterion(features, labels)
                    
                    val_total_loss += loss.item() * bsz
                    val_total_samples += bsz
            
            val_avg_loss = val_total_loss / val_total_samples
            val_losses.append(val_avg_loss)
            
            # 记录当前epoch的训练和验证损失
            epoch_info = f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_avg_loss:.4f}, Val Loss: {val_avg_loss:.4f}"
            loss_file.write(epoch_info + "\n")
            print(epoch_info)
            
            # 如果使用学习率调度器，更新学习率
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_avg_loss)
                else:
                    scheduler.step()
            
            # 如果当前验证损失是最好的，保存模型并重置耐心计数器
            if val_avg_loss < best_val_loss - args.min_delta:
                best_val_loss = val_avg_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)
                print(f"找到新的最佳模型! Epoch {best_epoch}, 验证损失: {best_val_loss:.4f}")
                loss_file.write(f"新的最佳模型! Epoch {best_epoch}, 验证损失: {best_val_loss:.4f}\n")
                patience_counter = 0  # 重置耐心计数器
            else:
                # 验证损失没有足够改善，增加耐心计数器
                patience_counter += 1
                print(f"验证损失未改善, 当前耐心计数: {patience_counter}/{args.patience}")
                loss_file.write(f"验证损失未改善, 当前耐心计数: {patience_counter}/{args.patience}\n")
                
                # 如果超过耐心值，触发早停
                if patience_counter >= args.patience:
                    early_stop_msg = f"早停触发! {args.patience}轮内验证损失无显著改善。"
                    print(early_stop_msg)
                    loss_file.write(f"\n{early_stop_msg}\n")
                    break
        
        # 记录训练结束信息
        end_time = time.time()
        training_time = end_time - start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        time_str = f"{hours}小时 {minutes}分钟 {seconds}秒"
        end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if patience_counter >= args.patience:
            end_reason = f"由于早停机制触发，训练在第{epoch+1}轮提前结束。"
        else:
            end_reason = f"训练完成所有{num_epochs}轮。"
            
        # 记录最佳模型信息和训练时间
        best_model_info = f"\n===== 训练结束 =====\n"
        best_model_info += f"训练结束时间: {end_time_str}\n"
        best_model_info += f"总训练时间: {time_str}\n"
        best_model_info += f"{end_reason}\n"
        best_model_info += f"最佳模型: Epoch {best_epoch}\n"
        best_model_info += f"验证损失: {best_val_loss:.4f}\n"
        best_model_info += f"保存路径: {best_model_path}"
        
        loss_file.write(best_model_info)
        print(best_model_info)
    
    # 保存最终模型
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存至: {final_model_path}")
    
    # 创建更美观的损失曲线图
    plt.figure(figsize=(12, 8))
    
    # 绘制训练和验证损失曲线
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', markersize=6, 
             linestyle='-', linewidth=2, color='#3498db', label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, marker='s', markersize=6, 
             linestyle='-', linewidth=2, color='#e74c3c', label='Val Loss')
    
    # 标记最佳模型点
    plt.axvline(x=best_epoch, color='#2ecc71', linestyle='--', linewidth=2, 
                label=f'Best Model (Epoch {best_epoch})')
    plt.scatter([best_epoch], [val_losses[best_epoch-1]], s=150, color='#2ecc71', 
                zorder=5, edgecolor='white', linewidth=2)
    
    # 设置图表标题和标签
    plt.title(f'Training and Validation Loss Curves\n({args.backbone}, Bands: {bands_str})', 
              fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=14, labelpad=10)
    plt.ylabel('Loss', fontsize=14, labelpad=10)
    
    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray', 
               shadow=True, loc='upper right')
    
    # 设置轴范围和刻度
    plt.xlim(0.5, len(train_losses) + 0.5)
    y_min = min(min(train_losses), min(val_losses)) * 0.9
    y_max = max(max(train_losses[:5]), max(val_losses[:5])) * 1.1  # 使用前5个epoch的最大值来设置上限
    plt.ylim(y_min, y_max)
    
    # 美化图表
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(args.output_dir, f'loss_curve_{args.backbone}_bands_{bands_str}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练完成! 最佳模型在第 {best_epoch} 轮, 验证损失: {best_val_loss:.4f}")
    print(f"训练开始时间: {start_time_str}")
    print(f"总训练时间: {time_str}")
    print(f"最佳模型已保存至: {best_model_path}")
