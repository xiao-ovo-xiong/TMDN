"""
特征融合模块（FusionModule）

该模块负责融合来自三个不同模态的特征向量。
支持多种融合策略：加权求和、拼接、注意力机制。
"""

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """
    特征融合模块
    
    该模块接收三个特征向量并将它们融合为单一特征向量。
    支持不同的融合策略，默认使用可学习的加权求和。
    
    Args:
        feature_dim (int): 特征向量的维度
        fusion_type (str): 融合类型，可选 'weighted_sum', 'concat', 'attention'
        learnable (bool): 权重是否可学习
    """
    
    def __init__(self, 
                 feature_dim: int,
                 fusion_type: str = 'weighted_sum',
                 learnable: bool = True):
        super(FusionModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        self.learnable = learnable
        
        # 验证融合类型
        valid_fusion_types = ['weighted_sum', 'concat', 'attention']
        if fusion_type not in valid_fusion_types:
            raise ValueError(
                f"Invalid fusion_type: {fusion_type}. "
                f"Must be one of {valid_fusion_types}"
            )
        
        # 初始化融合权重参数
        # 使用均匀分布 [1/3, 1/3, 1/3] 作为初始值
        if fusion_type == 'weighted_sum':
            # 创建原始权重参数（未归一化）
            # 使用 log(1/3) 作为初始值，这样经过 softmax 后会得到 [1/3, 1/3, 1/3]
            init_value = torch.log(torch.tensor(1.0 / 3.0))
            self.raw_weights = nn.Parameter(
                torch.full((3,), init_value.item()),
                requires_grad=learnable
            )
        
        elif fusion_type == 'concat':
            # 拼接模式：将三个特征向量拼接后通过线性层降维
            self.projection = nn.Linear(feature_dim * 3, feature_dim)
        
        elif fusion_type == 'attention':
            # 注意力模式：使用自注意力机制动态计算融合权重
            self.query = nn.Linear(feature_dim, feature_dim)
            self.key = nn.Linear(feature_dim, feature_dim)
            self.value = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, 
                feat1: torch.Tensor,
                feat2: torch.Tensor,
                feat3: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            feat1: 第一个特征向量 [batch_size, feature_dim]
            feat2: 第二个特征向量 [batch_size, feature_dim]
            feat3: 第三个特征向量 [batch_size, feature_dim]
        
        Returns:
            融合后的特征向量 [batch_size, feature_dim]
        """
        # 验证输入形状
        if feat1.shape != feat2.shape or feat1.shape != feat3.shape:
            raise ValueError(
                f"Feature shape mismatch: feat1={feat1.shape}, "
                f"feat2={feat2.shape}, feat3={feat3.shape}"
            )
        
        if feat1.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, "
                f"got {feat1.shape[-1]}"
            )
        
        if self.fusion_type == 'weighted_sum':
            return self._weighted_sum_fusion(feat1, feat2, feat3)
        elif self.fusion_type == 'concat':
            return self._concat_fusion(feat1, feat2, feat3)
        elif self.fusion_type == 'attention':
            return self._attention_fusion(feat1, feat2, feat3)
    
    def _weighted_sum_fusion(self,
                            feat1: torch.Tensor,
                            feat2: torch.Tensor,
                            feat3: torch.Tensor) -> torch.Tensor:
        """
        加权求和融合
        
        使用 softmax 归一化权重确保和为 1，然后执行加权求和。
        """
        # 通过 softmax 归一化权重，确保和为 1
        weights = torch.softmax(self.raw_weights, dim=0)
        
        # 执行加权求和
        fused = weights[0] * feat1 + weights[1] * feat2 + weights[2] * feat3
        
        return fused
    
    def _concat_fusion(self,
                      feat1: torch.Tensor,
                      feat2: torch.Tensor,
                      feat3: torch.Tensor) -> torch.Tensor:
        """
        拼接融合
        
        将三个特征向量拼接后通过线性层降维到原始维度。
        """
        # 拼接三个特征向量
        concatenated = torch.cat([feat1, feat2, feat3], dim=-1)
        
        # 通过线性层降维
        fused = self.projection(concatenated)
        
        return fused
    
    def _attention_fusion(self,
                         feat1: torch.Tensor,
                         feat2: torch.Tensor,
                         feat3: torch.Tensor) -> torch.Tensor:
        """
        注意力融合
        
        使用自注意力机制动态计算融合权重。
        """
        # 堆叠特征向量 [batch_size, 3, feature_dim]
        features = torch.stack([feat1, feat2, feat3], dim=1)
        
        # 计算 query, key, value
        q = self.query(features)  # [batch_size, 3, feature_dim]
        k = self.key(features)    # [batch_size, 3, feature_dim]
        v = self.value(features)  # [batch_size, 3, feature_dim]
        
        # 计算注意力分数
        # [batch_size, 3, 3]
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 应用注意力权重
        # [batch_size, 3, feature_dim]
        attended = torch.matmul(attention_weights, v)
        
        # 对三个模态的输出求平均
        fused = attended.mean(dim=1)  # [batch_size, feature_dim]
        
        return fused
