"""
Segmentation Validation Script — Optimized for 4GB GPU
Hardware: Intel Core 5 210H, 16 GB RAM, 4 GB VRAM

Key optimizations vs original:
  - Mixed-precision (AMP FP16) → faster inference, lower VRAM
  - num_workers + pin_memory → faster data loading
  - Batch-level metric accumulation (avoids per-image Python loops)
  - Optional: load best checkpoint (segmentation_head_best.pth) for higher IoU
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm

plt.switch_backend('Agg')


# ============================================================================
# Mask Conversion
# ============================================================================

value_map = {
    0: 0,       # background
    100: 1,     # Trees
    200: 2,     # Lush Bushes
    300: 3,     # Dry Grass
    500: 4,     # Dry Bushes
    550: 5,     # Ground Clutter
    700: 6,     # Logs
    800: 7,     # Rocks
    7100: 8,    # Landscape
    10000: 9    # Sky
}

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

n_classes = len(value_map)

color_palette = np.array([
    [0, 0, 0],        # Background
    [34, 139, 34],    # Trees
    [0, 255, 0],      # Lush Bushes
    [210, 180, 140],  # Dry Grass
    [139, 90, 43],    # Dry Bushes
    [128, 128, 0],    # Ground Clutter
    [139, 69, 19],    # Logs
    [128, 128, 128],  # Rocks
    [160, 82, 45],    # Landscape
    [135, 206, 235],  # Sky
], dtype=np.uint8)


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_classes):
        color_mask[mask == c] = color_palette[c]
    return color_mask


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, img_size):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids  = sorted(os.listdir(self.image_dir))
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_resize = transforms.Resize(
            img_size, interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask  = Image.open(os.path.join(self.masks_dir, data_id))
        mask  = convert_mask(mask)
        image = self.img_transform(image)
        mask  = self.mask_resize(mask)
        mask  = torch.from_numpy(np.array(mask)).long()
        return image, mask, data_id


# ============================================================================
# Segmentation Head — must match the deeper head used in training
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH, hidden=256):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden),
            nn.GELU()
        )

        def make_block(ch):
            return nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=7, padding=3, groups=ch),
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.Conv2d(ch, ch * 2, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(ch * 2, ch, kernel_size=1),
            )

        self.block1 = make_block(hidden)
        self.block2 = make_block(hidden)
        self.block3 = make_block(hidden)
        self.classifier = nn.Conv2d(hidden, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou_batch(pred_classes, target, num_classes=10):
    """Vectorized per-class IoU over a flattened batch."""
    p = pred_classes.view(-1)
    t = target.view(-1)
    iou_list = []
    for c in range(num_classes):
        inter = ((p == c) & (t == c)).sum().float()
        union = ((p == c) | (t == c)).sum().float()
        if union > 0:
            iou_list.append((inter / union).item())
    return float(np.nanmean(iou_list)) if iou_list else 0.0, \
           [((( (p == c) & (t == c)).sum() / ((p == c) | (t == c)).sum()).item()
             if ((p == c) | (t == c)).sum() > 0 else float('nan'))
            for c in range(num_classes)]


def compute_pixel_accuracy(pred_classes, target):
    return (pred_classes == target).float().mean().item()


# ============================================================================
# Visualization
# ============================================================================

def denorm(img_tensor):
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = np.moveaxis(img, 0, -1) * std + mean
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def save_comparison(img_t, gt_mask, pred_mask, path, title=""):
    img      = denorm(img_t)
    gt_color = mask_to_color(gt_mask.cpu().numpy())
    pr_color = mask_to_color(pred_mask.cpu().numpy())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);      axes[0].set_title('Input');        axes[0].axis('off')
    axes[1].imshow(gt_color); axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(pr_color); axes[2].set_title('Prediction');   axes[2].axis('off')
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()


def save_metrics_summary(results, output_dir):
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write(f"Mean Pixel Acc:    {results['mean_acc']:.4f}\n\n")
        f.write("Per-Class IoU:\n")
        for name, iou in zip(class_names, results['class_iou']):
            s = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<20}: {s}\n")

    # Bar chart
    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(n_classes), valid_iou,
           color=[color_palette[i] / 255 for i in range(n_classes)], edgecolor='black')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f'Per-Class IoU (Mean: {results["mean_iou"]:.4f})')
    ax.set_ylim(0, 1)
    ax.axhline(results['mean_iou'], color='red', linestyle='--', label='Mean')
    ax.axhline(0.5, color='orange', linestyle=':', label='Target 0.5')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics to {output_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=os.path.join(script_dir, 'segmentation_head_best.pth'),
                        help='Path to model weights. Use segmentation_head_best.pth for best IoU.')
    parser.add_argument('--data_dir',   default=os.path.join(script_dir, 'Offroad_Segmentation_testImages'))
    parser.add_argument('--output_dir', default='./predictions')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of side-by-side comparison images to save')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Must match the hidden_dim used during training (default 256)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Resolution (must match training)
    w = int(((960 / 2) // 14) * 14)   # 476
    h = int(((540 / 2) // 14) * 14)   # 266

    # Dataset
    print(f"Loading dataset from {args.data_dir}...")
    valset     = MaskDataset(args.data_dir, img_size=(h, w))
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"Loaded {len(valset)} samples")

    # Backbone
    print("Loading DINOv2-small backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)

    # Probe dims
    dummy = torch.zeros(1, 3, h, w, device=device)
    with torch.no_grad():
        n_embedding = backbone.forward_features(dummy)["x_norm_patchtokens"].shape[2]
    tokenH, tokenW = h // 14, w // 14
    print(f"Embedding dim: {n_embedding} | Token grid: {tokenH}×{tokenW}")

    # Classifier
    print(f"Loading model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        # Fallback to non-best checkpoint
        fallback = os.path.join(script_dir, 'segmentation_head.pth')
        print(f"  '{args.model_path}' not found, trying '{fallback}'")
        args.model_path = fallback
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding, out_channels=n_classes,
        tokenW=tokenW, tokenH=tokenH, hidden=args.hidden_dim
    )
    classifier.load_state_dict(torch.load(args.model_path, map_location=device))
    classifier.eval().to(device)
    print("Model loaded!")

    # Output dirs
    masks_dir      = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    for d in [masks_dir, masks_color_dir, comparisons_dir]:
        os.makedirs(d, exist_ok=True)

    # Inference
    iou_scores, pixel_accs, all_class_iou = [], [], []
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating", unit="batch")
        for imgs, labels, data_ids in pbar:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                feats   = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits  = classifier(feats)
                outputs = F.interpolate(logits, size=imgs.shape[2:],
                                        mode="bilinear", align_corners=False)

            preds = torch.argmax(outputs, dim=1)
            iou, class_iou = compute_iou_batch(preds, labels)
            acc             = compute_pixel_accuracy(preds, labels)

            iou_scores.append(iou)
            pixel_accs.append(acc)
            all_class_iou.append(class_iou)
            pbar.set_postfix(iou=f"{iou:.3f}")

            # Save outputs
            for i in range(imgs.shape[0]):
                base = os.path.splitext(data_ids[i])[0]
                pred_np = preds[i].cpu().numpy().astype(np.uint8)

                Image.fromarray(pred_np).save(os.path.join(masks_dir, f'{base}_pred.png'))
                cv2.imwrite(os.path.join(masks_color_dir, f'{base}_pred_color.png'),
                            cv2.cvtColor(mask_to_color(pred_np), cv2.COLOR_RGB2BGR))

                if sample_count < args.num_samples:
                    save_comparison(
                        imgs[i], labels[i], preds[i],
                        os.path.join(comparisons_dir, f'sample_{sample_count}.png'),
                        title=data_ids[i]
                    )
                sample_count += 1

    mean_iou = float(np.nanmean(iou_scores))
    mean_acc = float(np.mean(pixel_accs))
    avg_class_iou = np.nanmean(all_class_iou, axis=0)

    results = {'mean_iou': mean_iou, 'mean_acc': mean_acc, 'class_iou': avg_class_iou}

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean IoU       : {mean_iou:.4f}  {'✓ ABOVE 0.5' if mean_iou > 0.5 else '✗ below 0.5'}")
    print(f"Pixel Accuracy : {mean_acc:.4f}")
    print("=" * 50)
    for name, iou in zip(class_names, avg_class_iou):
        s = f"{iou:.4f}" if not np.isnan(iou) else "N/A "
        print(f"  {name:<20}: {s}")

    save_metrics_summary(results, args.output_dir)
    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()