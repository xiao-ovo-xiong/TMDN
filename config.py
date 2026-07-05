"""
配置数据类模块

提供多模态ConvNeXt模型的配置管理功能，支持从JSON/YAML文件加载配置。
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import yaml
import os


@dataclass
class MultiModalConfig:
    """
    多模态ConvNeXt模型配置
    
    包含模型架构、融合策略、预处理参数等所有可配置选项。
    支持从JSON/YAML文件加载和保存配置。
    
    Attributes:
        num_classes: 分类类别数
        depths: ConvNeXt各阶段的深度列表，默认[3, 3, 27, 3]
        dims: ConvNeXt各阶段的通道数列表，默认[96, 192, 384, 768]
        fusion_type: 特征融合类型，可选'weighted_sum', 'concat', 'attention'
        shared_weights: 三个特征提取器是否共享权重
        drop_path_rate: DropPath正则化率
        wavelet_type: 小波变换类型，如'db1', 'haar', 'sym2'等
        fourier_mode: 傅里叶变换模式，'magnitude'或'phase'
        learnable_fusion: 融合权重是否可学习（仅对weighted_sum有效）
    """
    
    num_classes: int
    depths: List[int] = field(default_factory=lambda: [3, 3, 27, 3])
    dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    fusion_type: str = 'weighted_sum'
    shared_weights: bool = False
    drop_path_rate: float = 0.0
    wavelet_type: str = 'db1'
    fourier_mode: str = 'magnitude'
    learnable_fusion: bool = True
    
    def __post_init__(self):
        """验证配置参数的有效性"""
        self._validate()
    
    def _validate(self):
        """
        验证配置参数
        
        Raises:
            ValueError: 当配置参数无效时
        """
        # 验证num_classes
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        
        # 验证depths和dims列表长度
        if len(self.depths) != 4:
            raise ValueError(f"depths must have length 4, got {len(self.depths)}")
        if len(self.dims) != 4:
            raise ValueError(f"dims must have length 4, got {len(self.dims)}")
        
        # 验证depths和dims的值
        if any(d <= 0 for d in self.depths):
            raise ValueError(f"All depths must be positive, got {self.depths}")
        if any(d <= 0 for d in self.dims):
            raise ValueError(f"All dims must be positive, got {self.dims}")
        
        # 验证fusion_type
        valid_fusion_types = ['weighted_sum', 'concat', 'attention']
        if self.fusion_type not in valid_fusion_types:
            raise ValueError(
                f"Invalid fusion_type: {self.fusion_type}. "
                f"Must be one of {valid_fusion_types}"
            )
        
        # 验证drop_path_rate
        if not 0.0 <= self.drop_path_rate < 1.0:
            raise ValueError(
                f"drop_path_rate must be in [0.0, 1.0), got {self.drop_path_rate}"
            )
        
        # 验证wavelet_type
        valid_wavelet_types = ['db1', 'db2', 'db3', 'db4', 'haar', 'sym2', 'sym3', 'coif1']
        if self.wavelet_type not in valid_wavelet_types:
            raise ValueError(
                f"Invalid wavelet_type: {self.wavelet_type}. "
                f"Must be one of {valid_wavelet_types}"
            )
        
        # 验证fourier_mode
        valid_fourier_modes = ['magnitude', 'phase']
        if self.fourier_mode not in valid_fourier_modes:
            raise ValueError(
                f"Invalid fourier_mode: {self.fourier_mode}. "
                f"Must be one of {valid_fourier_modes}"
            )
    
    def to_dict(self) -> dict:
        """
        将配置转换为字典
        
        Returns:
            包含所有配置参数的字典
        """
        return asdict(self)
    
    def to_json(self, filepath: str, indent: int = 2) -> None:
        """
        将配置保存为JSON文件
        
        Args:
            filepath: JSON文件路径
            indent: JSON缩进空格数
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)
    
    def to_yaml(self, filepath: str) -> None:
        """
        将配置保存为YAML文件
        
        Args:
            filepath: YAML文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MultiModalConfig':
        """
        从字典创建配置对象
        
        Args:
            config_dict: 包含配置参数的字典
            
        Returns:
            MultiModalConfig实例
        """
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'MultiModalConfig':
        """
        从JSON文件加载配置
        
        Args:
            filepath: JSON文件路径
            
        Returns:
            MultiModalConfig实例
            
        Raises:
            FileNotFoundError: 当文件不存在时
            json.JSONDecodeError: 当JSON格式无效时
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'MultiModalConfig':
        """
        从YAML文件加载配置
        
        Args:
            filepath: YAML文件路径
            
        Returns:
            MultiModalConfig实例
            
        Raises:
            FileNotFoundError: 当文件不存在时
            yaml.YAMLError: 当YAML格式无效时
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'MultiModalConfig':
        """
        从文件加载配置（自动检测JSON或YAML格式）
        
        Args:
            filepath: 配置文件路径
            
        Returns:
            MultiModalConfig实例
            
        Raises:
            FileNotFoundError: 当文件不存在时
            ValueError: 当文件格式不支持时
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.json':
            return cls.from_json(filepath)
        elif ext in ['.yaml', '.yml']:
            return cls.from_yaml(filepath)
        else:
            raise ValueError(
                f"Unsupported config file format: {ext}. "
                "Must be .json, .yaml, or .yml"
            )
    
    def __repr__(self) -> str:
        """返回配置的字符串表示"""
        return (
            f"MultiModalConfig(\n"
            f"  num_classes={self.num_classes},\n"
            f"  depths={self.depths},\n"
            f"  dims={self.dims},\n"
            f"  fusion_type='{self.fusion_type}',\n"
            f"  shared_weights={self.shared_weights},\n"
            f"  drop_path_rate={self.drop_path_rate},\n"
            f"  wavelet_type='{self.wavelet_type}',\n"
            f"  fourier_mode='{self.fourier_mode}',\n"
            f"  learnable_fusion={self.learnable_fusion}\n"
            f")"
        )
