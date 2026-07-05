import os
import json
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from multimodal_convnext import MultiModalConvNeXt
from twin_network import TwinNetwork
from model import convnext_small as create_model


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """
    绘制并保存混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def predict_single_modal(args):
    """单模态预测"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    print(f"Mode: Single-modal ({args.modality} images)")

    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载类别映射
    assert os.path.exists(args.class_json), f"{args.class_json} not found."
    with open(args.class_json, "r") as f:
        class_indict = json.load(f)

    class_name_to_idx = {v: k for k, v in class_indict.items()}
    class_names = [class_indict[str(i)] for i in range(len(class_indict))]

    # 创建单模态模型
    model = create_model(num_classes=args.num_classes).to(device)
    
    # 加载权重
    state_dict = torch.load(args.weights, map_location=device)
    
    # 🔍 检查是否为多模态权重
    if any('fusion_module' in key for key in state_dict.keys()):
        print("\n" + "="*60)
        print("❌ ERROR: You are trying to load MULTI-MODAL weights")
        print("   with SINGLE-MODAL prediction mode!")
        print("="*60)
        print(f"Weights file: {args.weights}")
        print("This file contains multi-modal model weights (fusion_module detected).")
        print("\nPlease use one of the following:")
        print("  1. Use --mode multi (or --mode auto) for multi-modal prediction")
        print("  2. Use a single-modal weights file for single-modal prediction")
        print("="*60)
        raise ValueError("Mode mismatch: Cannot load multi-modal weights in single-modal mode")
    
    # 处理嵌套的 'model' 键（原始 ConvNeXt 预训练权重格式）
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    
    # 🔍 严格加载权重，确保完全匹配
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ Successfully loaded single-modal weights")
    except RuntimeError as e:
        print("\n" + "="*60)
        print("❌ ERROR: Weight loading failed!")
        print("="*60)
        print(f"Error details: {str(e)}")
        print("\nPossible reasons:")
        print("  1. The weights file is not compatible with single-modal model")
        print("  2. The weights file has different architecture (num_classes, depths, dims)")
        print("  3. The weights file is corrupted")
        print("="*60)
        raise
    
    model.eval()

    # 测试目录 - 根据 modality 参数选择
    test_dir = os.path.join(args.data_root, f"{args.modality}/test")
    
    # 用于混淆矩阵
    all_true_labels = []
    all_pred_labels = []
    
    total_correct = 0
    total_images = 0

    # 遍历测试目录
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)

        if not os.path.isdir(class_dir):
            continue

        true_class_idx = class_name_to_idx.get(class_name)
        if true_class_idx is None:
            print(f"Warning: Class {class_name} not found in class_indices.json, skipping...")
            continue

        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in tqdm(image_files, desc=f"Processing {class_name}"):
            img_path = os.path.join(class_dir, img_file)

            try:
                img = Image.open(img_path).convert('RGB')
                tensor = data_transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = torch.squeeze(model(tensor))
                    predict = torch.softmax(output, dim=0)
                    pred_class_idx = torch.argmax(predict).item()

                total_images += 1
                all_true_labels.append(int(true_class_idx))
                all_pred_labels.append(pred_class_idx)
                
                if str(pred_class_idx) == true_class_idx:
                    total_correct += 1

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

    return total_images, total_correct, all_true_labels, all_pred_labels, class_names


def predict_multi_modal(args):
    """多模态预测"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    print("Mode: Multi-modal (Original + Wavelet + Fourier)")

    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载类别映射
    assert os.path.exists(args.class_json), f"{args.class_json} not found."
    with open(args.class_json, "r") as f:
        class_indict = json.load(f)

    class_name_to_idx = {v: k for k, v in class_indict.items()}
    class_names = [class_indict[str(i)] for i in range(len(class_indict))]

    # 创建多模态模型
    model = MultiModalConvNeXt(
        num_classes=args.num_classes,
        depths=args.depths,
        dims=args.dims,
        fusion_type=args.fusion_type,
        shared_weights=args.shared_weights,
        drop_path_rate=0.0
    ).to(device)
    
    # 加载权重
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 显示融合权重
    if hasattr(model.fusion_module, 'raw_weights') and args.fusion_type == 'weighted_sum':
        weights = torch.softmax(model.fusion_module.raw_weights, dim=0)
        print(f"Fusion weights: Original={weights[0]:.4f}, Wavelet={weights[1]:.4f}, Fourier={weights[2]:.4f}")

    # 多模态数据路径
    original_test_dir = os.path.join(args.data_root, "original/test")
    wavelet_test_dir = os.path.join(args.data_root, "wavelet/test")
    fourier_test_dir = os.path.join(args.data_root, "fourier/test")

    # 用于混淆矩阵
    all_true_labels = []
    all_pred_labels = []
    
    total_correct = 0
    total_images = 0

    # 遍历测试目录
    for class_name in os.listdir(original_test_dir):
        class_dir_original = os.path.join(original_test_dir, class_name)
        class_dir_wavelet = os.path.join(wavelet_test_dir, class_name)
        class_dir_fourier = os.path.join(fourier_test_dir, class_name)

        if not os.path.isdir(class_dir_original):
            continue

        true_class_idx = class_name_to_idx.get(class_name)
        if true_class_idx is None:
            print(f"Warning: Class {class_name} not found in class_indices.json, skipping...")
            continue

        image_files = [f for f in os.listdir(class_dir_original) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in tqdm(image_files, desc=f"Processing {class_name}"):
            img_path_original = os.path.join(class_dir_original, img_file)
            img_path_wavelet = os.path.join(class_dir_wavelet, img_file)
            img_path_fourier = os.path.join(class_dir_fourier, img_file)

            try:
                # 加载三种模态
                img_original = Image.open(img_path_original).convert('RGB')
                img_wavelet = Image.open(img_path_wavelet).convert('RGB')
                img_fourier = Image.open(img_path_fourier).convert('RGB')
                
                tensor_original = data_transform(img_original).unsqueeze(0).to(device)
                tensor_wavelet = data_transform(img_wavelet).unsqueeze(0).to(device)
                tensor_fourier = data_transform(img_fourier).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = torch.squeeze(model(tensor_original, tensor_wavelet, tensor_fourier))
                    predict = torch.softmax(output, dim=0)
                    pred_class_idx = torch.argmax(predict).item()

                total_images += 1
                all_true_labels.append(int(true_class_idx))
                all_pred_labels.append(pred_class_idx)
                
                if str(pred_class_idx) == true_class_idx:
                    total_correct += 1

            except Exception as e:
                print(f"Error processing {img_path_original}: {str(e)}")
                continue

    return total_images, total_correct, all_true_labels, all_pred_labels, class_names


def predict_twin(args):
    """双胞胎网络预测"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    print(f"Mode: Twin network ({args.twin_fusion_type})")

    data_transform = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    assert os.path.exists(args.class_json), f"{args.class_json} not found."
    with open(args.class_json, "r") as f:
        class_indict = json.load(f)

    class_name_to_idx = {v: k for k, v in class_indict.items()}
    class_names = [class_indict[str(i)] for i in range(len(class_indict))]

    model = TwinNetwork(
        num_classes=args.num_classes,
        convnext_depths=args.depths,
        convnext_dims=args.dims,
        twin_fusion_type=args.twin_fusion_type,
        swin_embed_dim=args.swin_embed_dim,
        swin_depths=tuple(args.swin_depths),
        swin_num_heads=tuple(args.swin_num_heads),
        swin_window_size=args.swin_window_size,
    ).to(device)

    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    original_test_dir = os.path.join(args.data_root, "original/test")
    wavelet_test_dir = os.path.join(args.data_root, "wavelet/test")
    fourier_test_dir = os.path.join(args.data_root, "fourier/test")

    all_true_labels, all_pred_labels = [], []
    total_correct = total_images = 0

    for class_name in os.listdir(original_test_dir):
        class_dir_o = os.path.join(original_test_dir, class_name)
        class_dir_w = os.path.join(wavelet_test_dir, class_name)
        class_dir_f = os.path.join(fourier_test_dir, class_name)

        if not os.path.isdir(class_dir_o):
            continue

        true_class_idx = class_name_to_idx.get(class_name)
        if true_class_idx is None:
            print(f"Warning: Class {class_name} not found in class_indices.json, skipping...")
            continue

        image_files = [f for f in os.listdir(class_dir_o)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in tqdm(image_files, desc=f"Processing {class_name}"):
            try:
                t_o = data_transform(Image.open(os.path.join(class_dir_o, img_file)).convert('RGB')).unsqueeze(0).to(device)
                t_w = data_transform(Image.open(os.path.join(class_dir_w, img_file)).convert('RGB')).unsqueeze(0).to(device)
                t_f = data_transform(Image.open(os.path.join(class_dir_f, img_file)).convert('RGB')).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = torch.squeeze(model(t_o, t_w, t_f, t_o, t_w, t_f))
                    pred_class_idx = torch.argmax(torch.softmax(output, dim=0)).item()

                total_images += 1
                all_true_labels.append(int(true_class_idx))
                all_pred_labels.append(pred_class_idx)
                if str(pred_class_idx) == true_class_idx:
                    total_correct += 1
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue

    return total_images, total_correct, all_true_labels, all_pred_labels, class_names


def _detect_twin_fusion_type(state_dict: dict) -> str:
    """
    从 state_dict 的 key/shape 推断 twin_fusion_type。
    - feature_gating: 有 twin_fusion.shared_head.weight
    - prob_gating / logits_gating: 通过 gate.weight 形状区分
      gate.weight shape [2, 2*num_classes] → logits_gating 或 prob_gating
      两者结构相同，无法从权重区分，默认返回 prob_gating
      （实际上两者可互换加载，因为结构完全一致）
    """
    keys = set(state_dict.keys())
    if 'twin_fusion.shared_head.weight' in keys:
        return 'feature_gating'
    if 'twin_fusion.gate.weight' in keys:
        gate_shape = state_dict['twin_fusion.gate.weight'].shape
        # gate: [2, 2*num_classes] — logits_gating 和 prob_gating 结构相同
        # 无法区分，返回 prob_gating（两者权重可互换）
        return 'prob_gating'
    return None


def main():
    parser = argparse.ArgumentParser(description='Predict with single-modal or multi-modal ConvNeXt')
    
    # 基本参数
    parser.add_argument('--mode', type=str, default='auto', choices=['single', 'multi', 'twin', 'auto'],
                       help='Prediction mode: single, multi, twin, or auto (detect from weights)')
    parser.add_argument('--modality', type=str, default='original', choices=['original', 'wavelet', 'fourier'],
                       help='Modality to use for single-modal prediction (original, wavelet, or fourier)')
    parser.add_argument('--weights', type=str, default='./weights/best_model.pth',
                       help='Path to model weights')
    parser.add_argument('--data-root', type=str, default='../_DATA_2',
                       help='Root directory of test data')
    parser.add_argument('--class-json', type=str, default='./class_indices.json',
                       help='Path to class indices JSON file')
    parser.add_argument('--num-classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (cuda:0 or cpu)')
    
    # 多模态参数
    parser.add_argument('--depths', type=int, nargs='+', default=[3, 3, 27, 3],
                       help='ConvNeXt depths for each stage')
    parser.add_argument('--dims', type=int, nargs='+', default=[96, 192, 384, 768],
                       help='ConvNeXt dimensions for each stage')
    parser.add_argument('--fusion-type', type=str, default='weighted_sum',
                       choices=['weighted_sum', 'concat', 'attention'],
                       help='Feature fusion strategy')
    parser.add_argument('--shared-weights', action='store_true',
                       help='Share weights across the three feature extractors')
    
    # Twin network parameters
    parser.add_argument('--twin-fusion-type', type=str, default='prob_gating',
                       choices=['logits_gating', 'prob_gating', 'feature_gating'],
                       help='Twin fusion strategy')
    parser.add_argument('--swin-embed-dim', type=int, default=96)
    parser.add_argument('--swin-depths', type=int, nargs='+', default=[2, 2, 6, 2])
    parser.add_argument('--swin-num-heads', type=int, nargs='+', default=[3, 6, 12, 24])
    parser.add_argument('--swin-window-size', type=int, default=7)
    
    # 混淆矩阵参数
    parser.add_argument('--save-confusion-matrix', action='store_true',
                       help='Save confusion matrix plot')
    parser.add_argument('--confusion-matrix-path', type=str, default='confusion_matrix.png',
                       help='Path to save confusion matrix')
    
    args = parser.parse_args()

    # 自动检测模式 + twin fusion type 推断
    if args.mode == 'auto':
        try:
            state_dict = torch.load(args.weights, map_location='cpu')
            if any('twin_fusion' in key for key in state_dict.keys()):
                args.mode = 'twin'
                # 从权重 key 自动推断 fusion type
                detected_type = _detect_twin_fusion_type(state_dict)
                if detected_type and detected_type != args.twin_fusion_type:
                    print(f"Auto-detected: Twin network model ({detected_type})")
                    args.twin_fusion_type = detected_type
                else:
                    print("Auto-detected: Twin network model")
            elif any('fusion_module' in key for key in state_dict.keys()):
                args.mode = 'multi'
                print("Auto-detected: Multi-modal model")
            else:
                args.mode = 'single'
                print("Auto-detected: Single-modal model")
        except Exception as e:
            print(f"Error detecting model type: {e}")
            print("Defaulting to multi-modal mode")
            args.mode = 'multi'
    elif args.mode == 'twin':
        # 手动指定 twin 模式时，也校验 fusion type 是否与权重匹配
        try:
            state_dict = torch.load(args.weights, map_location='cpu')
            detected_type = _detect_twin_fusion_type(state_dict)
            if detected_type and detected_type != args.twin_fusion_type:
                print(f"⚠️  Warning: checkpoint was saved with fusion_type='{detected_type}', "
                      f"but you specified '--twin-fusion-type {args.twin_fusion_type}'.")
                print(f"   Auto-correcting to '{detected_type}' to match the checkpoint.")
                args.twin_fusion_type = detected_type
        except Exception:
            pass

    # 🔍 参数验证：检查用户是否在命令行中明确指定了 modality 参数
    # 通过检查 sys.argv 来判断用户是否显式提供了 --modality
    import sys
    modality_explicitly_set = '--modality' in sys.argv
    
    if args.mode == 'multi' and modality_explicitly_set:
        print("\n" + "="*60)
        print("⚠️  WARNING: --modality parameter is ignored in multi-modal mode")
        print("="*60)
        print(f"You specified: --modality {args.modality}")
        print("But multi-modal prediction ALWAYS uses all three modalities:")
        print("  - Original images")
        print("  - Wavelet images")
        print("  - Fourier images")
        print("\nThe --modality parameter is only used in single-modal mode.")
        print("If you want to test only one modality, use:")
        print(f"  python predict.py --mode single --modality {args.modality} --weights <single_modal_weights>")
        print("="*60)
        print()

    # 执行预测
    if args.mode == 'single':
        total_images, total_correct, all_true_labels, all_pred_labels, class_names = predict_single_modal(args)
    elif args.mode == 'twin':
        total_images, total_correct, all_true_labels, all_pred_labels, class_names = predict_twin(args)
    else:
        total_images, total_correct, all_true_labels, all_pred_labels, class_names = predict_multi_modal(args)

    # 计算准确率
    if total_images == 0:
        print("No valid images found.")
        return

    accuracy = total_correct / total_images
    
    print("\n" + "="*70)
    print(" "*25 + "TEST RESULTS")
    print("="*70)
    print(f"Total Images:          {total_images}")
    print(f"Correct Predictions:   {total_correct}")
    print(f"Wrong Predictions:     {total_images - total_correct}")
    print(f"Overall Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*70)

    # 生成混淆矩阵
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    
    print("\n" + "="*70)
    print(" "*23 + "CONFUSION MATRIX")
    print("="*70)
    print(f"{'':>15} {'Predicted False':>18} {'Predicted True':>18}")
    print(f"{'Actual False':<15} {cm[0][0]:>18} {cm[0][1]:>18}")
    print(f"{'Actual True':<15} {cm[1][0]:>18} {cm[1][1]:>18}")
    print("="*70)
    
    # 计算每个类别的指标
    print("\n" + "="*70)
    print(" "*20 + "DETAILED METRICS BY CLASS")
    print("="*70)
    
    # 使用 classification_report 获取详细指标
    report = classification_report(all_true_labels, all_pred_labels, 
                                   target_names=class_names, 
                                   digits=4, 
                                   output_dict=True)
    
    # 打印每个类别的指标
    print(f"{'Class':<12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print("-"*70)
    for class_name in class_names:
        metrics = report[class_name]
        print(f"{class_name:<12} "
              f"{metrics['precision']:>12.4f} "
              f"{metrics['recall']:>12.4f} "
              f"{metrics['f1-score']:>12.4f} "
              f"{int(metrics['support']):>12}")
    
    print("-"*70)
    # 打印平均指标
    print(f"{'Macro Avg':<12} "
          f"{report['macro avg']['precision']:>12.4f} "
          f"{report['macro avg']['recall']:>12.4f} "
          f"{report['macro avg']['f1-score']:>12.4f} "
          f"{int(report['macro avg']['support']):>12}")
    print(f"{'Weighted Avg':<12} "
          f"{report['weighted avg']['precision']:>12.4f} "
          f"{report['weighted avg']['recall']:>12.4f} "
          f"{report['weighted avg']['f1-score']:>12.4f} "
          f"{int(report['weighted avg']['support']):>12}")
    print("="*70)
    
    # 额外的分析
    print("\n" + "="*70)
    print(" "*25 + "ERROR ANALYSIS")
    print("="*70)
    false_positives = cm[0][1]  # 实际是False，预测为True
    false_negatives = cm[1][0]  # 实际是True，预测为False
    print(f"False Positives (False → True):  {false_positives} ({false_positives/total_images*100:.2f}%)")
    print(f"False Negatives (True → False):  {false_negatives} ({false_negatives/total_images*100:.2f}%)")
    print("="*70)
    
    # 保存混淆矩阵图
    if args.save_confusion_matrix:
        plot_confusion_matrix(cm, class_names, args.confusion_matrix_path)
        print(f"\n✓ Confusion matrix plot saved to: {args.confusion_matrix_path}")


if __name__ == '__main__':
    main()
