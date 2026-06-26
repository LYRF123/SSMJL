import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRLoss(nn.Module):
    """
    SimCLR无监督对比损失函数
    
    基于论文: "A Simple Framework for Contrastive Learning of Visual Representations"
    https://arxiv.org/abs/2002.05709
    """
    
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Args:
            temperature: 温度参数，用于缩放相似度
            base_temperature: 基础温度参数（通常与temperature相同）
        """
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, z1, z2):
        """
        计算SimCLR损失
        
        Args:
            z1: 第一个视图的投影特征 [batch_size, feature_dim]
            z2: 第二个视图的投影特征 [batch_size, feature_dim]
            
        Returns:
            loss: SimCLR对比损失
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # 归一化特征向量
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        # 拼接两个视图的特征 [2*batch_size, feature_dim]
        features = torch.cat([z1_norm, z2_norm], dim=0)
        
        # 计算相似度矩阵 [2*batch_size, 2*batch_size]
        similarity_matrix = torch.mm(features, features.T) / self.temperature
        
        # 创建标签：每个样本的正样本对是另一个视图
        # 对于前batch_size个样本，正样本在后batch_size位置
        # 对于后batch_size个样本，正样本在前batch_size位置
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # 创建掩码，排除自相似度（对角线元素）
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # 计算交叉熵损失
        loss = self.criterion(similarity_matrix, labels)
        
        return loss
    
    def info_nce_loss(self, z1, z2):
        """
        另一种实现方式：InfoNCE损失
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # 归一化
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        # 计算正样本对的相似度
        pos_sim = torch.sum(z1_norm * z2_norm, dim=1) / self.temperature  # [batch_size]
        
        # 计算所有可能的负样本对相似度
        # z1与z2中所有其他样本的相似度
        neg_sim = torch.mm(z1_norm, z2_norm.T) / self.temperature  # [batch_size, batch_size]
        
        # 移除正样本对（对角线元素）
        mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        neg_sim = neg_sim.masked_fill(mask, float('-inf'))
        
        # 计算InfoNCE损失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, batch_size]
        labels = torch.zeros(batch_size, device=device, dtype=torch.long)
        
        loss = self.criterion(logits, labels)
        
        return loss

class SupConLoss(nn.Module):
    """
    监督对比损失（用于比较）
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: 特征向量 [bsz, n_views, feature_dim] 或 [bsz, feature_dim]
            labels: 类别标签 [bsz]
            mask: 对比掩码 [bsz, bsz]
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 计算logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss 