#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集预处理脚本

该脚本使用ImagePreprocessor对整个数据集进行批量预处理，
生成小波变换和傅里叶变换图像并保存到磁盘。

使用示例:
    python preprocess_dataset.py --original_root ../_DATA_3_40_加入的false蒸馏样本 --output_root ../_DATA_3_40_加入的false蒸馏样本/processed
    python preprocess_dataset.py --original_root data/original --output_root data/processed --wavelet_type db2
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from image_preprocessor import ImagePreprocessor


def setup_logging(log_level: str = 'INFO'):
    """
    配置日志系统
    
    Args:
        log_level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stderr)]
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='预处理图像数据集，生成小波变换和傅里叶变换图像',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--original_root',
        type=str,
        required=True,
        help='原始图像数据集根目录路径（包含original子目录）'
    )
    
    parser.add_argument(
        '--output_root',
        type=str,
        required=True,
        help='输出根目录路径（将创建wavelet和fourier子目录）'
    )
    
    parser.add_argument(
        '--wavelet_type',
        type=str,
        default='db1',
        choices=['db1', 'db2', 'db4', 'haar', 'sym2', 'sym4', 'coif1'],
        help='小波变换类型'
    )
    
    parser.add_argument(
        '--fourier_mode',
        type=str,
        default='magnitude',
        choices=['magnitude', 'phase'],
        help='傅里叶变换模式（幅度谱或相位谱）'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='并行处理的进程数（默认为CPU核心数-1）'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='日志级别'
    )
    
    return parser.parse_args()


def validate_paths(original_root: str, output_root: str):
    """
    验证输入输出路径
    
    Args:
        original_root: 原始数据集根目录
        output_root: 输出根目录
        
    Raises:
        ValueError: 如果路径无效
    """
    original_path = Path(original_root)
    
    if not original_path.exists():
        raise ValueError(f"原始数据集路径不存在: {original_root}")
    
    if not original_path.is_dir():
        raise ValueError(f"原始数据集路径不是目录: {original_root}")
    
    # 检查是否包含original子目录
    original_subdir = original_path / 'original'
    if not original_subdir.exists():
        logging.warning(
            f"未找到'original'子目录，将直接处理 {original_root} 下的所有图像"
        )
    
    # 检查输出目录是否可写
    output_path = Path(output_root)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"输出路径存在但不是目录: {output_root}")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(log_level=args.log_level)
    
    logging.info("验证输入和输出路径...")
    
    try:
        # 验证路径
        validate_paths(args.original_root, args.output_root)
        logging.info("路径验证通过")
        
        # 统计图像文件数量
        logging.info("统计图像文件数量...")
        original_path = Path(args.original_root)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in original_path.rglob('*') 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        num_images = len(image_files)
        logging.info(f"找到 {num_images} 个图像文件")
        
        if num_images == 0:
            logging.warning("未找到任何图像文件，退出")
            return 0
        
        # 创建预处理器
        logging.info(f"创建ImagePreprocessor实例 (wavelet_type={args.wavelet_type}, fourier_mode={args.fourier_mode})")
        preprocessor = ImagePreprocessor(
            wavelet_type=args.wavelet_type,
            fourier_mode=args.fourier_mode
        )
        
        # 开始处理
        start_time = time.time()
        logging.info("开始处理数据集...")
        logging.info(f"  原始图像目录: {args.original_root}")
        logging.info(f"  输出目录: {args.output_root}")
        if args.num_workers:
            logging.info(f"  并行进程数: {args.num_workers}")
        
        preprocessor.process_dataset(
            original_root=args.original_root,
            output_root=args.output_root,
            num_workers=args.num_workers
        )
        
        elapsed_time = time.time() - start_time
        
        # 处理完成
        logging.info("数据集处理完成！")
        logging.info(f"总耗时: {elapsed_time:.2f} 秒")
        
        if num_images > 0:
            avg_speed = num_images / elapsed_time
            logging.info(f"平均处理速度: {avg_speed:.2f} 图像/秒")
        
        wavelet_path = Path(args.output_root) / 'wavelet'
        fourier_path = Path(args.output_root) / 'fourier'
        logging.info(f"小波变换图像保存在: {wavelet_path}")
        logging.info(f"傅里叶变换图像保存在: {fourier_path}")
        
        return 0
        
    except Exception as e:
        logging.error(f"预处理失败: {str(e)}")
        if args.log_level == 'DEBUG':
            logging.exception("详细错误信息:")
        return 1


if __name__ == '__main__':
    sys.exit(main())
