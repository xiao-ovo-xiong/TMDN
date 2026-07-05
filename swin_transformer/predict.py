import os
import sys
import json
import argparse

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# 将父目录加入 sys.path
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from swin_transformer.model import swin_tiny_patch4_window7_224 as create_model
from swin_transformer.multimodal_swin import MultiModalSwinTransformer


def _load_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def _load_class_map(class_json):
    assert os.path.exists(class_json), f"{class_json} not found."
    with open(class_json) as f:
        class_indict = json.load(f)
    class_name_to_idx = {v: k for k, v in class_indict.items()}
    class_names = [class_indict[str(i)] for i in range(len(class_indict))]
    return class_name_to_idx, class_names


def _print_results(total_images, total_correct, all_true, all_pred, class_names):
    if total_images == 0:
        print("No valid images found.")
        return
    accuracy = total_correct / total_images
    print("\n" + "=" * 70)
    print(" " * 25 + "TEST RESULTS")
    print("=" * 70)
    print(f"Total Images:          {total_images}")
    print(f"Correct Predictions:   {total_correct}")
    print(f"Wrong Predictions:     {total_images - total_correct}")
    print(f"Overall Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 70)

    cm = confusion_matrix(all_true, all_pred)
    print("\n" + "=" * 70)
    print(" " * 23 + "CONFUSION MATRIX")
    print("=" * 70)
    print(f"{'':>15} {'Predicted False':>18} {'Predicted True':>18}")
    print(f"{'Actual False':<15} {cm[0][0]:>18} {cm[0][1]:>18}")
    print(f"{'Actual True':<15} {cm[1][0]:>18} {cm[1][1]:>18}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print(" " * 20 + "DETAILED METRICS BY CLASS")
    print("=" * 70)
    print(f"{'Class':<12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print("-" * 70)
    report = classification_report(all_true, all_pred, target_names=class_names,
                                   digits=4, output_dict=True)
    for class_name in class_names:
        m = report[class_name]
        print(f"{class_name:<12} {m['precision']:>12.4f} {m['recall']:>12.4f} "
              f"{m['f1-score']:>12.4f} {int(m['support']):>12}")
    print("-" * 70)
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
    print("=" * 70)

    fp = cm[0][1]
    fn = cm[1][0]
    print("\n" + "=" * 70)
    print(" " * 25 + "ERROR ANALYSIS")
    print("=" * 70)
    print(f"False Positives (False → True):  {fp} ({fp/total_images*100:.2f}%)")
    print(f"False Negatives (True → False):  {fn} ({fn/total_images*100:.2f}%)")
    print("=" * 70)


def predict_single(args):
    """单模态预测，支持 --modality 指定使用哪个模态的测试集"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    print(f"Mode: Single-modal ({args.modality} images)")

    transform = _load_transform(args.img_size)
    class_name_to_idx, class_names = _load_class_map(args.class_json)

    model = create_model(num_classes=args.num_classes).to(device)
    state_dict = torch.load(args.weights, map_location=device)

    # 检查是否误用了多模态权重
    if any('fusion_module' in k for k in state_dict.keys()):
        print("\n" + "=" * 60)
        print("❌ ERROR: 检测到多模态权重，请使用 --mode multi")
        print("=" * 60)
        raise ValueError("Mode mismatch: multi-modal weights in single-modal mode")

    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("✓ Successfully loaded single-modal weights")

    test_dir = os.path.join(args.data_root, f"{args.modality}/test")
    all_true, all_pred = [], []
    total_correct = total_images = 0

    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        true_idx = class_name_to_idx.get(class_name)
        if true_idx is None:
            print(f"Warning: Class {class_name} not found in class_indices.json, skipping...")
            continue
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in tqdm(image_files, desc=f"Processing {class_name}"):
            try:
                t = transform(Image.open(os.path.join(class_dir, img_file)).convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_idx = torch.argmax(torch.softmax(torch.squeeze(model(t)), dim=0)).item()
                total_images += 1
                all_true.append(int(true_idx))
                all_pred.append(pred_idx)
                if str(pred_idx) == true_idx:
                    total_correct += 1
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    return total_images, total_correct, all_true, all_pred, class_names


def predict_multi(args):
    """多模态预测（original + wavelet + fourier）"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    print(f"Mode: Multi-modal (Original + Wavelet + Fourier, fusion={args.fusion_type})")

    # 如果用户显式指定了 --modality，给出提示
    if '--modality' in sys.argv:
        print("\n⚠️  WARNING: --modality is ignored in multi-modal mode.")
        print("   Multi-modal prediction always uses all three modalities.\n")

    transform = _load_transform(args.img_size)
    class_name_to_idx, class_names = _load_class_map(args.class_json)

    model = MultiModalSwinTransformer(
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
        window_size=args.window_size,
        fusion_type=args.fusion_type,
    ).to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 显示融合权重
    if hasattr(model.fusion_module, 'raw_weights') and args.fusion_type == 'weighted_sum':
        w = torch.softmax(model.fusion_module.raw_weights, dim=0)
        print(f"Fusion weights: Original={w[0]:.4f}, Wavelet={w[1]:.4f}, Fourier={w[2]:.4f}")

    dirs = {
        'original': os.path.join(args.data_root, "original/test"),
        'wavelet':  os.path.join(args.data_root, "wavelet/test"),
        'fourier':  os.path.join(args.data_root, "fourier/test"),
    }
    all_true, all_pred = [], []
    total_correct = total_images = 0

    for class_name in os.listdir(dirs['original']):
        class_dirs = {k: os.path.join(v, class_name) for k, v in dirs.items()}
        if not os.path.isdir(class_dirs['original']):
            continue
        true_idx = class_name_to_idx.get(class_name)
        if true_idx is None:
            print(f"Warning: Class {class_name} not found in class_indices.json, skipping...")
            continue
        image_files = [f for f in os.listdir(class_dirs['original'])
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in tqdm(image_files, desc=f"Processing {class_name}"):
            try:
                t_o = transform(Image.open(os.path.join(class_dirs['original'], img_file)).convert('RGB')).unsqueeze(0).to(device)
                t_w = transform(Image.open(os.path.join(class_dirs['wavelet'],  img_file)).convert('RGB')).unsqueeze(0).to(device)
                t_f = transform(Image.open(os.path.join(class_dirs['fourier'],  img_file)).convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_idx = torch.argmax(torch.softmax(torch.squeeze(model(t_o, t_w, t_f)), dim=0)).item()
                total_images += 1
                all_true.append(int(true_idx))
                all_pred.append(pred_idx)
                if str(pred_idx) == true_idx:
                    total_correct += 1
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    return total_images, total_correct, all_true, all_pred, class_names


def main():
    parser = argparse.ArgumentParser(description='Swin Transformer Predict')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['single', 'multi', 'auto'],
                        help='Prediction mode: single, multi, or auto (detect from weights)')
    parser.add_argument('--modality', type=str, default='original',
                        choices=['original', 'wavelet', 'fourier'],
                        help='Modality for single-modal prediction (default: original)')
    parser.add_argument('--weights', type=str, default='./weights/best_model.pth',
                        help='Path to model weights')
    parser.add_argument('--data-root', type=str, default='../_DATA_2',
                        help='Root directory of test data')
    parser.add_argument('--class-json', type=str, default='./class_indices.json',
                        help='Path to class indices JSON file')
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda:0')
    # 多模态参数
    parser.add_argument('--fusion-type', type=str, default='weighted_sum',
                        choices=['weighted_sum', 'concat', 'attention'])
    parser.add_argument('--embed-dim', type=int, default=96)
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 6, 2])
    parser.add_argument('--num-heads', type=int, nargs='+', default=[3, 6, 12, 24])
    parser.add_argument('--window-size', type=int, default=7)
    args = parser.parse_args()

    # auto 检测模式
    if args.mode == 'auto':
        try:
            sd = torch.load(args.weights, map_location='cpu')
            if any('fusion_module' in k for k in sd.keys()):
                args.mode = 'multi'
                print("Auto-detected: Multi-modal model")
            else:
                args.mode = 'single'
                print("Auto-detected: Single-modal model")
        except Exception as e:
            print(f"Detection error: {e}, defaulting to single")
            args.mode = 'single'

    if args.mode == 'multi':
        total_images, total_correct, all_true, all_pred, class_names = predict_multi(args)
    else:
        total_images, total_correct, all_true, all_pred, class_names = predict_single(args)

    _print_results(total_images, total_correct, all_true, all_pred, class_names)


if __name__ == '__main__':
    main()
