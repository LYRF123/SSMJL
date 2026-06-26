import torch
import torch.nn as nn
import numpy as np
import rasterio
from pathlib import Path
import os
import sys

# 添加项目路径以便导入模型
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import models_convmae
from util.pos_embed import get_2d_sincos_pos_embed


class ToTensorMultiband:
    """Convert numpy array to tensor for multiband images"""
    def __call__(self, img):
        # img is numpy array (H, W, C)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # (C, H, W)
        return img


class ConvMAEFeatureExtractor(nn.Module):
    """
    基于ConvMAE的双图像特征提取器
    输入：高温tif图 + 对照tif图
    输出：拼接后的特征向量
    """
    
    def __init__(self, 
                 model_name='convmae_convvit_tiny_patch16',
                 checkpoint_path=None,
                 in_chans=5,
                 input_size=224,
                 selected_bands=[0, 1, 2, 3, 4],
                 mean=None,
                 std=None,
                 device='cuda'):
        """
        Args:
            model_name: ConvMAE模型名称
            checkpoint_path: 预训练权重路径
            in_chans: 输入通道数
            input_size: 输入图像尺寸
            selected_bands: 选择的波段
            mean: 标准化均值
            std: 标准化标准差
            device: 设备
        """
        super().__init__()
        
        self.in_chans = in_chans
        self.input_size = input_size
        self.selected_bands = selected_bands
        self.device = device
        
        # 默认标准化参数（与训练时一致）
        self.mean = mean if mean is not None else [0.5] * in_chans
        self.std = std if std is not None else [0.5] * in_chans
        
        # 创建ConvMAE模型
        print(f"🏗️ 创建模型: {model_name}")
        self.convmae = models_convmae.__dict__[model_name](
            norm_pix_loss=True,
            in_chans=in_chans
        )
        
        # 加载预训练权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_pretrained_weights(checkpoint_path)
        else:
            print("⚠️ 未提供checkpoint路径或文件不存在，使用随机初始化权重")
        
        # 提取encoder部分（移除decoder）
        self._setup_feature_extractor()
        
        # 移动到指定设备
        self.to(device)
        self.eval()  # 设置为评估模式
        
        print(f"✅ 特征提取器初始化完成")
        print(f"   - 输入通道数: {in_chans}")
        print(f"   - 输入尺寸: {input_size}x{input_size}")
        print(f"   - 选择波段: {selected_bands}")
        print(f"   - 设备: {device}")
    
    def _load_pretrained_weights(self, checkpoint_path):
        """加载预训练权重"""
        print(f"📥 加载预训练权重: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理不同的checkpoint格式
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 加载权重（忽略decoder部分）
        model_dict = self.convmae.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            # 只保留encoder相关的权重
            if k in model_dict and not k.startswith('decoder'):
                pretrained_dict[k] = v
                
        model_dict.update(pretrained_dict)
        self.convmae.load_state_dict(model_dict, strict=False)
        
        print(f"✅ 成功加载 {len(pretrained_dict)} 个参数")
    
    def _setup_feature_extractor(self):
        """设置特征提取器"""
        # 提取encoder组件
        self.patch_embed1 = self.convmae.patch_embed1
        self.patch_embed2 = self.convmae.patch_embed2
        self.patch_embed3 = self.convmae.patch_embed3
        self.patch_embed4 = self.convmae.patch_embed4
        
        self.blocks1 = self.convmae.blocks1
        self.blocks2 = self.convmae.blocks2
        self.blocks3 = self.convmae.blocks3
        
        self.pos_embed = self.convmae.pos_embed
        self.norm = self.convmae.norm
        
        # 提取192->100的投影层，仅保留第一层线性映射
        if hasattr(self.convmae, 'decoder_embed'):
            # decoder_embed[0] 为 nn.Linear(embed_dim[-1], 100)
            self.projection = self.convmae.decoder_embed[0]
            self.feature_dim = self.projection.out_features
        else:
            # 如果没有decoder_embed，则手动构建相同的线性层
            embed_dim = self.convmae.embed_dim[-1] if hasattr(self.convmae, 'embed_dim') else 768
            self.projection = nn.Linear(embed_dim, 100, bias=True)
            self.feature_dim = 100
        
        print(f"🎯 特征维度: {self.feature_dim}")
        print(f"🔗 最终特征维度: {self.feature_dim * 2} (双图像拼接)")
    
    def forward_encoder(self, x):
        """ConvMAE编码器前向传播"""
        # Stage 1: patch_embed1 + blocks1
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x)
        
        # Stage 2: patch_embed2 + blocks2  
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        
        # Stage 3: patch_embed3 + patch_embed4 + blocks3
        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        x = self.patch_embed4(x)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer blocks
        for blk in self.blocks3:
            x = blk(x)
        
        # 标准化
        x = self.norm(x)
        
        # 全局平均池化
        x = x.mean(dim=1)  # (B, embed_dim)
        
        return x
    
    def preprocess_image(self, img_path):
        """预处理TIFF图像"""
        # 读取TIFF图像
        with rasterio.open(img_path) as src:
            # 检查波段数
            if max(self.selected_bands) >= src.count:
                raise ValueError(f"请求的波段 {max(self.selected_bands)} 超出图像波段数 {src.count}")
            
            # 读取选定波段（rasterio使用1-based索引）
            img = src.read([b + 1 for b in self.selected_bands])
            
            # 转换为 (H, W, C) 格式
            img = img.transpose(1, 2, 0).astype(np.float32)
        
        # 转换为tensor
        img = ToTensorMultiband()(img)  # (C, H, W)
        
        # 标准化
        img = self._normalize(img)
        
        # 调整尺寸
        if img.shape[-1] != self.input_size or img.shape[-2] != self.input_size:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), 
                size=(self.input_size, self.input_size), 
                mode='bicubic', 
                align_corners=False
            ).squeeze(0)
        
        return img
    
    def _normalize(self, img):
        """标准化图像"""
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        return (img - mean) / std
    
    def extract_features(self, high_temp_path, control_path):
        """
        提取双图像特征
        
        Args:
            high_temp_path: 高温图像路径
            control_path: 对照图像路径
            
        Returns:
            features: 拼接后的特征向量 (feature_dim * 2,)
        """
        with torch.no_grad():
            # 预处理两张图像
            high_temp_img = self.preprocess_image(high_temp_path).to(self.device)
            control_img = self.preprocess_image(control_path).to(self.device)
            
            # 添加batch维度
            high_temp_img = high_temp_img.unsqueeze(0)  # (1, C, H, W)
            control_img = control_img.unsqueeze(0)      # (1, C, H, W)
            
            # 提取encoder特征
            high_temp_features = self.forward_encoder(high_temp_img)  # (1, embed_dim)
            control_features = self.forward_encoder(control_img)      # (1, embed_dim)
            
            # 通过投影层
            high_temp_features = self.projection(high_temp_features)  # (1, feature_dim)
            control_features = self.projection(control_features)      # (1, feature_dim)
            
            # 拼接特征
            combined_features = torch.cat([high_temp_features, control_features], dim=1)  # (1, feature_dim*2)
            
            return combined_features.squeeze(0).cpu().numpy()  # (feature_dim*2,)
    
    def extract_batch_features(self, high_temp_paths, control_paths):
        """
        批量提取特征
        
        Args:
            high_temp_paths: 高温图像路径列表
            control_paths: 对照图像路径列表
            
        Returns:
            features: 特征矩阵 (N, feature_dim*2)
        """
        assert len(high_temp_paths) == len(control_paths), "两类图像数量必须相同"
        
        all_features = []
        
        for ht_path, ctrl_path in zip(high_temp_paths, control_paths):
            features = self.extract_features(ht_path, ctrl_path)
            all_features.append(features)
        
        return np.array(all_features)
    
    def get_feature_dim(self):
        """获取最终特征维度"""
        return self.feature_dim * 2


def create_feature_extractor(checkpoint_path, **kwargs):
    """
    创建特征提取器的便捷函数
    
    Args:
        checkpoint_path: 预训练模型路径
        **kwargs: 其他参数
        
    Returns:
        feature_extractor: ConvMAEFeatureExtractor实例
    """
    return ConvMAEFeatureExtractor(checkpoint_path=checkpoint_path, **kwargs)


# 使用示例
if __name__ == "__main__":
    # 创建特征提取器
    extractor = ConvMAEFeatureExtractor(
        model_name='convmae_convvit_tiny_patch16',  # 根据你的训练选择
        checkpoint_path='',  # pass your checkpoint path explicitly
        in_chans=5,
        input_size=224,
        selected_bands=[0, 1, 2, 3, 4],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 提取单对图像特征
    high_temp_img_path = "path/to/high_temp.tif"
    control_img_path = "path/to/control.tif"
    
    # features = extractor.extract_features(high_temp_img_path, control_img_path)
    # print(f"特征维度: {features.shape}")
    # print(f"特征范围: [{features.min():.3f}, {features.max():.3f}]")
    
    # 批量提取特征示例
    # high_temp_paths = ["ht1.tif", "ht2.tif", "ht3.tif"]
    # control_paths = ["ctrl1.tif", "ctrl2.tif", "ctrl3.tif"]
    # batch_features = extractor.extract_batch_features(high_temp_paths, control_paths)
    # print(f"批量特征形状: {batch_features.shape}")
    
    print("✅ 特征提取器创建完成!")
