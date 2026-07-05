"""
图像预处理器模块 - 修复版

该模块提供ImagePreprocessor类，用于离线生成小波变换和傅里叶变换图像。
修复了奇数尺寸图像的维度不匹配问题。
支持多进程并行处理以提升速度。
"""

import os
import numpy as np
import pywt
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial


class ImagePreprocessor:
    """
    图像预处理器，负责生成小波变换和傅里叶变换图像并保存到磁盘。
    
    该类支持批量处理图像数据集，自动创建目录结构，并保持与原始数据集
    相同的文件组织方式。
    
    Attributes:
        wavelet_type (str): 小波类型，如'db1', 'haar', 'sym2'等
        fourier_mode (str): 傅里叶变换模式，'magnitude'或'phase'
    """
    
    def __init__(self, wavelet_type: str = 'db1', fourier_mode: str = 'magnitude'):
        """
        初始化ImagePreprocessor。
        
        Args:
            wavelet_type (str): 小波类型，默认为'db1' (Daubechies 1)
                支持的类型包括: 'db1', 'db2', 'haar', 'sym2'等
            fourier_mode (str): 傅里叶变换模式，默认为'magnitude'
                - 'magnitude': 提取幅度谱
                - 'phase': 提取相位谱
        """
        self.wavelet_type = wavelet_type
        self.fourier_mode = fourier_mode
    
    def wavelet_transform(self, image: np.ndarray) -> np.ndarray:
        """
        执行小波变换。
        
        对RGB三通道图像的每个通道独立进行小波变换，保持输出图像的空间维度
        与输入一致，并将变换结果归一化到[0, 255]范围。
        
        小波系数（cA, cH, cV, cD）被组合成一个与原始图像相同大小的图像，
        用于特征提取。
        
        Args:
            image (np.ndarray): 输入图像，形状为 [H, W, C]，数据类型为uint8
        
        Returns:
            np.ndarray: 变换后的图像，形状为 [H, W, C]，数据类型为uint8
        
        Raises:
            ValueError: 如果输入图像不是3通道RGB图像
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image with shape [H, W, 3], got {image.shape}"
            )
        
        height, width, channels = image.shape
        transformed_image = np.zeros((height, width, channels), dtype=np.float64)
        
        # 对每个通道独立进行小波变换
        for c in range(channels):
            channel = image[:, :, c].astype(np.float64)
            
            # 执行2D小波分解
            coeffs = pywt.dwt2(channel, self.wavelet_type)
            cA, (cH, cV, cD) = coeffs
            
            # 获取系数尺寸
            cA_h, cA_w = cA.shape
            cH_h, cH_w = cH.shape
            cV_h, cV_w = cV.shape
            cD_h, cD_w = cD.shape
            
            # 创建与原始图像相同大小的数组
            reconstructed = np.zeros((height, width), dtype=np.float64)
            
            # 将四个系数组合成一个图像（2x2布局）
            # 左上：近似系数 (cA)
            reconstructed[:cA_h, :cA_w] = cA
            
            # 右上：水平细节系数 (cH)
            h_end = min(cH_h, height)
            w_start = cA_w
            w_end = min(cA_w + cH_w, width)
            reconstructed[:h_end, w_start:w_end] = cH[:h_end, :(w_end - w_start)]
            
            # 左下：垂直细节系数 (cV)
            h_start = cA_h
            h_end = min(cA_h + cV_h, height)
            w_end = min(cV_w, width)
            reconstructed[h_start:h_end, :w_end] = cV[:(h_end - h_start), :w_end]
            
            # 右下：对角细节系数 (cD)
            h_start = cA_h
            h_end = min(cA_h + cD_h, height)
            w_start = cA_w
            w_end = min(cA_w + cD_w, width)
            reconstructed[h_start:h_end, w_start:w_end] = cD[:(h_end - h_start), :(w_end - w_start)]
            
            transformed_image[:, :, c] = reconstructed
        
        # 归一化到[0, 255]范围
        min_val = transformed_image.min()
        max_val = transformed_image.max()
        
        # 避免除零
        if max_val - min_val > 1e-10:
            normalized = (transformed_image - min_val) / (max_val - min_val) * 255.0
        else:
            normalized = np.zeros_like(transformed_image)
        
        # 转换为uint8
        return normalized.astype(np.uint8)
    
    def fourier_transform(self, image: np.ndarray) -> np.ndarray:
        """
        执行傅里叶变换。

        对RGB三通道图像的每个通道独立进行傅里叶变换，提取幅度谱并进行对数缩放，
        保持输出图像的空间维度与输入一致，并将变换结果归一化到[0, 255]范围。

        Args:
            image (np.ndarray): 输入图像，形状为 [H, W, C]，数据类型为uint8

        Returns:
            np.ndarray: 变换后的图像，形状为 [H, W, C]，数据类型为uint8

        Raises:
            ValueError: 如果输入图像不是3通道RGB图像
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image with shape [H, W, 3], got {image.shape}"
            )

        height, width, channels = image.shape
        transformed_image = np.zeros_like(image, dtype=np.float64)

        # 对每个通道独立进行傅里叶变换
        for c in range(channels):
            channel = image[:, :, c].astype(np.float64)

            # 执行2D傅里叶变换
            fft = np.fft.fft2(channel)

            # 将零频率分量移到频谱中心
            fft_shifted = np.fft.fftshift(fft)

            # 提取幅度谱
            magnitude = np.abs(fft_shifted)

            # 对数缩放以增强可视化效果
            # 添加小的epsilon避免log(0)
            epsilon = 1e-10
            log_magnitude = np.log(magnitude + epsilon)

            transformed_image[:, :, c] = log_magnitude

        # 归一化到[0, 255]范围
        min_val = transformed_image.min()
        max_val = transformed_image.max()

        # 避免除零
        if max_val - min_val > 1e-10:
            normalized = (transformed_image - min_val) / (max_val - min_val) * 255.0
        else:
            normalized = np.zeros_like(transformed_image)

        # 转换为uint8
        return normalized.astype(np.uint8)
    
    def process_dataset(self, original_root: str, output_root: str, num_workers: int = None) -> None:
        """
        处理整个数据集（支持多进程并行）。
        
        遍历原始图像目录，对每张图像执行小波变换和傅里叶变换，并保存到
        对应的输出目录。自动创建wavelet和fourier目录结构，保持原始目录
        层次结构（train/test和类别文件夹）。
        
        Args:
            original_root (str): 原始图像根目录，包含'original'子目录
            output_root (str): 输出根目录，将创建'wavelet'和'fourier'子目录
            num_workers (int): 并行处理的进程数，默认为CPU核心数-1
        
        Raises:
            ValueError: 如果original_root不包含'original'目录
            OSError: 如果无法创建输出目录或保存文件
        """
        # 验证输入路径
        if not os.path.exists(original_root):
            raise ValueError(f"Original root directory does not exist: {original_root}")
        
        # 收集所有图像文件路径
        image_files = []
        for root, dirs, files in os.walk(original_root):
            for filename in files:
                if self._is_image_file(filename):
                    original_path = os.path.join(root, filename)
                    image_files.append(original_path)
        
        total_files = len(image_files)
        print(f"Found {total_files} images to process")
        
        if total_files == 0:
            print("No images found!")
            return
        
        # 确定工作进程数
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        
        print(f"Using {num_workers} worker processes")
        
        # 创建处理函数的部分应用
        process_func = partial(
            self._process_single_image,
            original_root=original_root,
            output_root=output_root,
            wavelet_type=self.wavelet_type,
            fourier_mode=self.fourier_mode
        )
        
        # 使用多进程池处理
        success_count = 0
        failed_files = []
        
        with Pool(processes=num_workers) as pool:
            # 使用imap_unordered以便实时显示进度
            results = pool.imap_unordered(process_func, image_files, chunksize=10)
            
            for i, (success, original_path, error_msg) in enumerate(results, 1):
                if success:
                    success_count += 1
                else:
                    failed_files.append((original_path, error_msg))
                
                # 每处理100个文件显示一次进度
                if i % 100 == 0 or i == total_files:
                    print(f"Progress: {i}/{total_files} ({i*100//total_files}%) - Success: {success_count}, Failed: {len(failed_files)}")
        
        # 打印处理结果摘要
        print(f"\nDataset processing completed:")
        print(f"  Total files: {total_files}")
        print(f"  Successfully processed: {success_count}")
        print(f"  Failed: {len(failed_files)}")
        
        if failed_files:
            print("\nFailed files:")
            for path, error in failed_files[:10]:
                print(f"  {path}: {error}")
            if len(failed_files) > 10:
                print(f"  ... and {len(failed_files) - 10} more")
    
    @staticmethod
    def _process_single_image(original_path, original_root, output_root, wavelet_type, fourier_mode):
        """
        处理单个图像文件（静态方法，用于多进程）。
        
        Args:
            original_path (str): 原始图像路径
            original_root (str): 原始图像根目录
            output_root (str): 输出根目录
            wavelet_type (str): 小波类型
            fourier_mode (str): 傅里叶模式
        
        Returns:
            tuple: (success, original_path, error_msg)
        """
        try:
            # 创建临时的预处理器实例
            preprocessor = ImagePreprocessor(wavelet_type, fourier_mode)
            
            # 读取原始图像
            image = preprocessor._load_image(original_path)
            
            # 执行小波变换
            wavelet_image = preprocessor.wavelet_transform(image)
            
            # 执行傅里叶变换
            fourier_image = preprocessor.fourier_transform(image)
            
            # 生成输出路径
            wavelet_path = preprocessor._get_output_path(
                original_path, original_root, output_root, 'wavelet'
            )
            fourier_path = preprocessor._get_output_path(
                original_path, original_root, output_root, 'fourier'
            )
            
            # 保存变换后的图像
            preprocessor._save_image(wavelet_image, wavelet_path)
            preprocessor._save_image(fourier_image, fourier_path)
            
            return (True, original_path, None)
            
        except Exception as e:
            return (False, original_path, str(e))
    
    def _is_image_file(self, filename: str) -> bool:
        """
        检查文件是否为图像文件。
        
        Args:
            filename (str): 文件名
        
        Returns:
            bool: 如果是图像文件返回True，否则返回False
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        ext = os.path.splitext(filename)[1].lower()
        return ext in image_extensions
    
    def _load_image(self, path: str) -> np.ndarray:
        """
        加载图像文件。
        
        Args:
            path (str): 图像文件路径
        
        Returns:
            np.ndarray: 图像数组，形状为 [H, W, C]
        
        Raises:
            ValueError: 如果图像无法加载或不是RGB格式
        """
        try:
            img = Image.open(path)
            
            # 转换为RGB格式（如果是RGBA或灰度图）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 转换为numpy数组
            image_array = np.array(img)
            
            return image_array
            
        except Exception as e:
            raise ValueError(f"Failed to load image from {path}: {e}")
    
    def _get_output_path(self, original_path: str, original_root: str, 
                        output_root: str, transform_type: str) -> str:
        """
        生成输出路径。
        
        将原始路径中的'original'替换为指定的变换类型（'wavelet'或'fourier'），
        并将根目录替换为输出根目录。保持文件名不变，但扩展名改为.jpg。
        
        Args:
            original_path (str): 原始图像路径
            original_root (str): 原始图像根目录
            output_root (str): 输出根目录
            transform_type (str): 变换类型，'wavelet'或'fourier'
        
        Returns:
            str: 输出路径
        """
        # 获取相对于原始根目录的相对路径
        rel_path = os.path.relpath(original_path, original_root)
        
        # 将路径中的'original'替换为变换类型
        # 处理路径分隔符
        path_parts = rel_path.split(os.sep)
        new_parts = [transform_type if part == 'original' else part for part in path_parts]
        new_rel_path = os.path.join(*new_parts)
        
        # 构建完整的输出路径
        output_path = os.path.join(output_root, new_rel_path)
        
        # 将扩展名改为.jpg
        base, _ = os.path.splitext(output_path)
        output_path = base + '.jpg'
        
        return output_path
    
    def _save_image(self, image: np.ndarray, path: str) -> None:
        """
        保存图像到磁盘。
        
        自动创建所需的目录结构。
        
        Args:
            image (np.ndarray): 图像数组，形状为 [H, W, C]，数据类型为uint8
            path (str): 保存路径
        
        Raises:
            OSError: 如果无法创建目录或保存文件
        """
        # 创建目录（如果不存在）
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # 转换为PIL图像并保存
        img = Image.fromarray(image)
        img.save(path, format='JPEG', quality=95)
