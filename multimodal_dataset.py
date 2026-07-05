from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Callable, Dict
import os


class MultiModalDataset(Dataset):
    """多模态数据集，同时加载原始图像、小波变换图像和傅里叶变换图像
    
    该数据集用于多模态ConvNeXt网络训练，同时提供三种不同的图像表示。
    
    Args:
        original_paths: 原始图像路径列表
        wavelet_paths: 小波变换图像路径列表
        fourier_paths: 傅里叶变换图像路径列表
        labels: 标签列表
        transform: 数据增强变换（可选），将应用于所有三种图像
        
    Raises:
        ValueError: 当三个路径列表长度不一致时
        
    Example:
        >>> dataset = MultiModalDataset(
        ...     original_paths=['data/original/train/cat/1.jpg'],
        ...     wavelet_paths=['data/wavelet/train/cat/1.jpg'],
        ...     fourier_paths=['data/fourier/train/cat/1.jpg'],
        ...     labels=[0],
        ...     transform=transforms.Compose([...])
        ... )
        >>> original, wavelet, fourier, label = dataset[0]
    """

    def __init__(
        self,
        original_paths: List[str],
        wavelet_paths: List[str],
        fourier_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None
    ):
        # 验证三个路径列表长度一致
        if not (len(original_paths) == len(wavelet_paths) == len(fourier_paths) == len(labels)):
            raise ValueError(
                f"路径列表长度不一致: "
                f"original={len(original_paths)}, "
                f"wavelet={len(wavelet_paths)}, "
                f"fourier={len(fourier_paths)}, "
                f"labels={len(labels)}"
            )
        
        self.original_paths = original_paths
        self.wavelet_paths = wavelet_paths
        self.fourier_paths = fourier_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.original_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """获取指定索引的数据
        
        Args:
            idx: 数据索引
            
        Returns:
            (original_img, wavelet_img, fourier_img, label) 元组
            
        Raises:
            ValueError: 当图像不是RGB模式时
        """
        # 加载三种模态的图像
        original_img = Image.open(self.original_paths[idx])
        wavelet_img = Image.open(self.wavelet_paths[idx])
        fourier_img = Image.open(self.fourier_paths[idx])
        
        # 验证图像模式为RGB
        if original_img.mode != 'RGB':
            raise ValueError(
                f"原始图像不是RGB模式: {self.original_paths[idx]}, mode={original_img.mode}"
            )
        if wavelet_img.mode != 'RGB':
            raise ValueError(
                f"小波图像不是RGB模式: {self.wavelet_paths[idx]}, mode={wavelet_img.mode}"
            )
        if fourier_img.mode != 'RGB':
            raise ValueError(
                f"傅里叶图像不是RGB模式: {self.fourier_paths[idx]}, mode={fourier_img.mode}"
            )
        
        # 获取标签
        label = self.labels[idx]
        
        # 对三种图像应用相同的transform
        if self.transform is not None:
            original_img = self.transform(original_img)
            wavelet_img = self.transform(wavelet_img)
            fourier_img = self.transform(fourier_img)
        
        return original_img, wavelet_img, fourier_img, label

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """批次整理函数
        
        将多个样本整理成批次张量。
        
        Args:
            batch: 样本列表，每个样本是 (original, wavelet, fourier, label) 元组
            
        Returns:
            (original_batch, wavelet_batch, fourier_batch, labels_batch) 元组
        """
        original_imgs, wavelet_imgs, fourier_imgs, labels = tuple(zip(*batch))
        
        original_batch = torch.stack(original_imgs, dim=0)
        wavelet_batch = torch.stack(wavelet_imgs, dim=0)
        fourier_batch = torch.stack(fourier_imgs, dim=0)
        labels_batch = torch.as_tensor(labels)
        
        return original_batch, wavelet_batch, fourier_batch, labels_batch

    @staticmethod
    def from_directories(
        original_root: str,
        wavelet_root: str,
        fourier_root: str,
        transform: Optional[Callable] = None
    ) -> 'MultiModalDataset':
        """从目录结构自动构建MultiModalDataset
        
        该方法自动扫描三个根目录，提取图像路径和标签，并验证目录结构的一致性。
        
        目录结构应该如下：
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
        
        Args:
            original_root: 原始图像根目录
            wavelet_root: 小波图像根目录
            fourier_root: 傅里叶图像根目录
            transform: 数据增强变换（可选）
            
        Returns:
            MultiModalDataset实例
            
        Raises:
            ValueError: 当目录结构不一致时
            FileNotFoundError: 当目录不存在时
            
        Example:
            >>> dataset = MultiModalDataset.from_directories(
            ...     original_root='data/original/train',
            ...     wavelet_root='data/wavelet/train',
            ...     fourier_root='data/fourier/train',
            ...     transform=transforms.Compose([...])
            ... )
        """
        # 验证目录存在
        for root_dir, name in [(original_root, 'original'), 
                                (wavelet_root, 'wavelet'), 
                                (fourier_root, 'fourier')]:
            if not os.path.exists(root_dir):
                raise FileNotFoundError(f"{name}目录不存在: {root_dir}")
            if not os.path.isdir(root_dir):
                raise ValueError(f"{name}路径不是目录: {root_dir}")
        
        # 构建路径列表和标签
        original_paths, original_labels, class_to_idx = MultiModalDataset._build_path_list(original_root)
        wavelet_paths, wavelet_labels, _ = MultiModalDataset._build_path_list(wavelet_root)
        fourier_paths, fourier_labels, _ = MultiModalDataset._build_path_list(fourier_root)
        
        # 验证三个目录结构的一致性
        MultiModalDataset._verify_consistency(
            original_paths, wavelet_paths, fourier_paths,
            original_labels, wavelet_labels, fourier_labels,
            original_root, wavelet_root, fourier_root
        )
        
        return MultiModalDataset(
            original_paths=original_paths,
            wavelet_paths=wavelet_paths,
            fourier_paths=fourier_paths,
            labels=original_labels,
            transform=transform
        )

    @staticmethod
    def _build_path_list(root_dir: str) -> Tuple[List[str], List[int], Dict[str, int]]:
        """从目录结构构建路径列表和标签
        
        扫描根目录下的所有类别文件夹，提取图像路径并分配标签。
        
        Args:
            root_dir: 根目录路径
            
        Returns:
            (paths, labels, class_to_idx) 元组
            - paths: 图像路径列表（按字母顺序排序）
            - labels: 对应的标签列表
            - class_to_idx: 类别名到索引的映射字典
            
        Raises:
            ValueError: 当根目录下没有找到类别文件夹时
        """
        # 获取所有类别文件夹（按字母顺序排序以确保一致性）
        classes = sorted([d for d in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir, d))])
        
        if len(classes) == 0:
            raise ValueError(f"在{root_dir}中没有找到类别文件夹")
        
        # 创建类别到索引的映射
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        # 收集所有图像路径和标签
        paths = []
        labels = []
        
        # 支持的图像扩展名
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        for cls_name in classes:
            cls_dir = os.path.join(root_dir, cls_name)
            cls_idx = class_to_idx[cls_name]
            
            # 获取该类别下的所有图像文件（按字母顺序排序）
            image_files = sorted([f for f in os.listdir(cls_dir)
                                 if os.path.isfile(os.path.join(cls_dir, f)) and
                                 os.path.splitext(f)[1].lower() in valid_extensions])
            
            for img_file in image_files:
                img_path = os.path.join(cls_dir, img_file)
                paths.append(img_path)
                labels.append(cls_idx)
        
        return paths, labels, class_to_idx

    @staticmethod
    def _verify_consistency(
        original_paths: List[str],
        wavelet_paths: List[str],
        fourier_paths: List[str],
        original_labels: List[int],
        wavelet_labels: List[int],
        fourier_labels: List[int],
        original_root: str,
        wavelet_root: str,
        fourier_root: str
    ) -> None:
        """验证三个目录结构的一致性
        
        检查三个目录是否包含相同数量的图像，以及对应的图像是否具有相同的相对路径和标签。
        
        Args:
            original_paths: 原始图像路径列表
            wavelet_paths: 小波图像路径列表
            fourier_paths: 傅里叶图像路径列表
            original_labels: 原始图像标签列表
            wavelet_labels: 小波图像标签列表
            fourier_labels: 傅里叶图像标签列表
            original_root: 原始图像根目录
            wavelet_root: 小波图像根目录
            fourier_root: 傅里叶图像根目录
            
        Raises:
            ValueError: 当目录结构不一致时
        """
        # 检查数量是否一致
        if not (len(original_paths) == len(wavelet_paths) == len(fourier_paths)):
            raise ValueError(
                f"三个目录中的图像数量不一致: "
                f"original={len(original_paths)}, "
                f"wavelet={len(wavelet_paths)}, "
                f"fourier={len(fourier_paths)}"
            )
        
        # 检查标签是否一致
        if not (original_labels == wavelet_labels == fourier_labels):
            raise ValueError(
                "三个目录中的标签顺序不一致，请确保目录结构完全相同"
            )
        
        # 检查相对路径是否一致
        for i in range(len(original_paths)):
            original_rel = os.path.relpath(original_paths[i], original_root)
            wavelet_rel = os.path.relpath(wavelet_paths[i], wavelet_root)
            fourier_rel = os.path.relpath(fourier_paths[i], fourier_root)
            
            if not (original_rel == wavelet_rel == fourier_rel):
                raise ValueError(
                    f"索引{i}处的相对路径不一致:\n"
                    f"  original: {original_rel}\n"
                    f"  wavelet: {wavelet_rel}\n"
                    f"  fourier: {fourier_rel}"
                )
