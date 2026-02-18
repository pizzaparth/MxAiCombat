"""
Segmentation Validation Script — v20
Hardware: Intel Core 5 210H, 16 GB RAM, 4 GB VRAM

Upgrades vs v19:
  - Loads v20 checkpoint format (dict with 'head' and 'backbone' keys)
  - Uses DINOv2-base (vitb14) + multi-scale FPN decoder (must match train-v20)
  - Test-Time Augmentation (TTA): averages predictions over original +
    horizontal flip → free IoU boost (~+0.02 to +0.04) at no training cost
  - Per-image IoU histogram saved for debugging hard images
  - Unified metric functions (identical to train-v20)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
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
# Model Architecture — must exactly match train-v20.py
# ============================================================================

class DINOv2MultiScale(nn.Module):
    def __init__(self, backbone, n_blocks=12):
        super().__init__()
        self.backbone  = backbone
        self.n_blocks  = n_blocks
        self._features = {}

        hook_idxs = [
            max(0, n_blocks // 4 - 1),
            max(0, n_blocks // 2 - 1),
            max(0, 3 * n_blocks // 4 - 1),
            n_blocks - 1,
        ]
        self.hook_idxs = sorted(set(hook_idxs))

        for idx in self.hook_idxs:
            backbone.blocks[idx].register_forward_hook(self._make_hook(idx))

    def _make_hook(self, idx):
        def hook(module, input, output):
            self._features[idx] = output[:, 1:, :]
        return hook

    def forward(self, x):
        self._features = {}
        _ = self.backbone(x)
        return [self._features[i] for i in self.hook_idxs]


class FPNSegHead(nn.Module):
    def __init__(self, in_channels, out_classes, tokenH, tokenW,
                 n_scales=4, fpn_ch=256):
        super().__init__()
        self.tokenH  = tokenH
        self.tokenW  = tokenW
        self.n_scales = n_scales

        self.laterals = nn.ModuleList([
            nn.Sequential(nn.Linear(in_channels, fpn_ch), nn.LayerNorm(fpn_ch))
            for _ in range(n_scales)
        ])

        def conv_block(ch):
            return nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=7, padding=3, groups=ch),
                nn.BatchNorm2d(ch), nn.GELU(),
                nn.Conv2d(ch, ch * 2, kernel_size=1), nn.GELU(),
                nn.Conv2d(ch * 2, ch, kernel_size=1),
            )

        self.fpn_blocks = nn.ModuleList([conv_block(fpn_ch) for _ in range(n_scales)])

        self.classifier = nn.Sequential(
            nn.Conv2d(fpn_ch, fpn_ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_ch // 2), nn.GELU(),
            nn.Conv2d(fpn_ch // 2, out_classes, kernel_size=1),
        )

    def _to_spatial(self, tokens):
        B, N, C = tokens.shape
        return tokens.reshape(B, self.tokenH, self.tokenW, C).permute(0, 3, 1, 2)

    def forward(self, multi_scale_tokens):
        laterals = [
            self._to_spatial(self.laterals[i](multi_scale_tokens[i]))
            for i in range(self.n_scales)
        ]
        feats = list(reversed(laterals))
        out = feats[0] + self.fpn_blocks[0](feats[0])
        for i in range(1, self.n_scales):
            out = out + feats[i] + self.fpn_blocks[i](feats[i])
        return self.classifier(out)


# ============================================================================
# Metrics — identical to train-v20.py
# ============================================================================

def compute_iou(pred_classes, target, num_classes=n_classes):
    p = pred_classes.view(-1)
    t = target.view(-1)
    iou_list = []
    for c in range(num_classes):
        inter = ((p == c) & (t == c)).sum().float()
        union = ((p == c) | (t == c)).sum().float()
        if union > 0:
            iou_list.append((inter / union).item())
    return float(np.nanmean(iou_list)) if iou_list else 0.0


def compute_iou_per_class(pred_classes, target, num_classes=n_classes):
    p = pred_classes.view(-1)
    t = target.view(-1)
    per_class = []
    for c in range(num_classes):
        inter = ((p == c) & (t == c)).sum().float()
        union = ((p == c) | (t == c)).sum().float()
        per_class.append((inter / union).item() if union > 0 else float('nan'))
    valid = [x for x in per_class if not np.isnan(x)]
    return (float(np.mean(valid)) if valid else 0.0), per_class


def compute_pixel_accuracy(pred_classes, target):
    return (pred_classes == target).float().mean().item()


# ============================================================================
# Test-Time Augmentation (TTA)
# ============================================================================

def predict_with_tta(extractor, head, imgs, img_size):
    """
    Averages softmax probability maps over:
      1. Original image
      2. Horizontal flip (then un-flipped before averaging)

    Averaging in probability space (not logit space) is more numerically stable.
    Free IoU boost of ~+0.02 to +0.04 with zero extra training.
    """
    with torch.cuda.amp.autocast(enabled=(imgs.device.type == 'cuda')):
        # Original
        feats_orig  = extractor(imgs)
        logits_orig = head(feats_orig)
        out_orig    = F.interpolate(logits_orig, size=img_size,
                                    mode="bilinear", align_corners=False)
        prob_orig   = F.softmax(out_orig, dim=1)

        # Horizontal flip
        imgs_flip   = torch.flip(imgs, dims=[3])
        feats_flip  = extractor(imgs_flip)
        logits_flip = head(feats_flip)
        out_flip    = F.interpolate(logits_flip, size=img_size,
                                    mode="bilinear", align_corners=False)
        # Un-flip predictions back to original orientation
        prob_flip   = torch.flip(F.softmax(out_flip, dim=1), dims=[3])

    # Average probability maps
    avg_probs = (prob_orig + prob_flip) / 2.0
    return torch.argmax(avg_probs, dim=1)


# ============================================================================
# Visualization
# ============================================================================

def denorm(img_tensor):
    img  = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = np.moveaxis(img, 0, -1) * std + mean
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def save_comparison(img_t, gt_mask, pred_mask, path, title="", iou=None):
    img      = denorm(img_t)
    gt_color = mask_to_color(gt_mask.cpu().numpy())
    pr_color = mask_to_color(pred_mask.cpu().numpy())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);      axes[0].set_title('Input');        axes[0].axis('off')
    axes[1].imshow(gt_color); axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(pr_color)
    axes[2].set_title(f'Prediction  IoU={iou:.3f}' if iou is not None else 'Prediction')
    axes[2].axis('off')
    if title:
        fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()


def save_metrics_summary(results, output_dir):
    # Text report
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("EVALUATION RESULTS (v20 — with TTA)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean IoU:       {results['mean_iou']:.4f}\n")
        f.write(f"Mean Pixel Acc: {results['mean_acc']:.4f}\n\n")
        f.write("Per-Class IoU:\n")
        for name, iou in zip(class_names, results['class_iou']):
            s = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<20}: {s}\n")

    # Bar chart
    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(n_classes), valid_iou,
           color=[color_palette[i] / 255 for i in range(n_classes)],
           edgecolor='black')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f'Per-Class IoU  (Mean: {results["mean_iou"]:.4f})')
    ax.set_ylim(0, 1)
    ax.axhline(results['mean_iou'], color='red',    linestyle='--', label='Mean IoU')
    ax.axhline(0.5,                 color='orange', linestyle=':',  label='Target 0.5')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Per-image histogram
    if results.get('per_image_iou'):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(results['per_image_iou'], bins=20, edgecolor='black')
        ax.axvline(results['mean_iou'], color='red', linestyle='--',
                   label=f"Mean={results['mean_iou']:.3f}")
        ax.axvline(0.5, color='orange', linestyle=':', label='Target 0.5')
        ax.set_xlabel('IoU per image'); ax.set_ylabel('Count')
        ax.set_title('Per-Image IoU Distribution')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'iou_histogram.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Metrics saved to {output_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        default=os.path.join(script_dir, 'segmentation_head_best.pth'))
    parser.add_argument('--data_dir',
                        default=os.path.join(script_dir, 'Offroad_Segmentation_testImages'))
    parser.add_argument('--output_dir', default='./predictions')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--fpn_ch',     type=int, default=256,
                        help='FPN channels — must match training')
    parser.add_argument('--no_tta', action='store_true',
                        help='Disable Test-Time Augmentation')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    use_tta = not args.no_tta
    print(f"Test-Time Augmentation: {'ON' if use_tta else 'OFF'}")

    # Resolution — must match training
    w = int(((960 / 2) // 14) * 14)   # 476
    h = int(((540 / 2) // 14) * 14)   # 266

    # Dataset
    print(f"Loading dataset from {args.data_dir}...")
    valset     = MaskDataset(args.data_dir, img_size=(h, w))
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"Loaded {len(valset)} samples")

    # Backbone
    print("Loading DINOv2-base backbone (vitb14)...")
    _backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    _backbone.eval().to(device)

    n_dino_blocks = len(_backbone.blocks)
    dummy = torch.zeros(1, 3, h, w, device=device)
    with torch.no_grad():
        n_embedding = _backbone.forward_features(dummy)["x_norm_patchtokens"].shape[2]
    tokenH, tokenW = h // 14, w // 14
    print(f"Embedding dim: {n_embedding} | Token grid: {tokenH}×{tokenW}")

    extractor = DINOv2MultiScale(_backbone, n_blocks=n_dino_blocks).to(device)
    n_scales  = len(extractor.hook_idxs)

    # Head
    head = FPNSegHead(
        in_channels=n_embedding, out_classes=n_classes,
        tokenH=tokenH, tokenW=tokenW,
        n_scales=n_scales, fpn_ch=args.fpn_ch,
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.model_path}...")
    if not os.path.exists(args.model_path):
        fallback = os.path.join(script_dir, 'segmentation_head.pth')
        print(f"  Not found, trying {fallback}")
        args.model_path = fallback

    ckpt = torch.load(args.model_path, map_location=device)

    if isinstance(ckpt, dict) and 'head' in ckpt:
        # v20 checkpoint format
        head.load_state_dict(ckpt['head'])
        _backbone.load_state_dict(ckpt['backbone'])
        saved_iou = ckpt.get('val_iou', 'unknown')
        print(f"  Loaded v20 checkpoint (saved val IoU: {saved_iou})")
    else:
        # Legacy v18/v19 format (head weights only, old architecture)
        raise RuntimeError(
            "This checkpoint appears to be from v18/v19 (old architecture). "
            "Please retrain with train-v20.py to use test-v20.py."
        )

    extractor.eval()
    head.eval()
    print("Model loaded!")

    # Output dirs
    masks_dir       = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    for d in [masks_dir, masks_color_dir, comparisons_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Inference ─────────────────────────────────────────────────────────────
    per_image_iou = []
    per_image_acc = []
    all_class_iou = []
    sample_count  = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating", unit="batch")
        for imgs, labels, data_ids in pbar:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_tta:
                preds = predict_with_tta(extractor, head, imgs, imgs.shape[2:])
            else:
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    feats   = extractor(imgs)
                    logits  = head(feats)
                    outputs = F.interpolate(logits, size=imgs.shape[2:],
                                            mode="bilinear", align_corners=False)
                preds = torch.argmax(outputs, dim=1)

            for i in range(imgs.shape[0]):
                img_iou, cls_iou = compute_iou_per_class(preds[i], labels[i])
                img_acc          = compute_pixel_accuracy(preds[i], labels[i])
                per_image_iou.append(img_iou)
                per_image_acc.append(img_acc)
                all_class_iou.append(cls_iou)

                base    = os.path.splitext(data_ids[i])[0]
                pred_np = preds[i].cpu().numpy().astype(np.uint8)

                Image.fromarray(pred_np).save(
                    os.path.join(masks_dir, f'{base}_pred.png'))
                cv2.imwrite(
                    os.path.join(masks_color_dir, f'{base}_pred_color.png'),
                    cv2.cvtColor(mask_to_color(pred_np), cv2.COLOR_RGB2BGR))

                if sample_count < args.num_samples:
                    save_comparison(
                        imgs[i], labels[i], preds[i],
                        os.path.join(comparisons_dir, f'sample_{sample_count:03d}.png'),
                        title=data_ids[i], iou=img_iou
                    )
                sample_count += 1

            pbar.set_postfix(iou=f"{float(np.mean(per_image_iou)):.3f}")

    mean_iou      = float(np.nanmean(per_image_iou))
    mean_acc      = float(np.mean(per_image_acc))
    avg_class_iou = np.nanmean(all_class_iou, axis=0)

    results = {
        'mean_iou':      mean_iou,
        'mean_acc':      mean_acc,
        'class_iou':     avg_class_iou,
        'per_image_iou': per_image_iou,
    }

    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS  [TTA={'ON' if use_tta else 'OFF'}]")
    print("=" * 50)
    print(f"Mean IoU       : {mean_iou:.4f}  "
          f"{'✓ ABOVE 0.5' if mean_iou > 0.5 else '✗ below 0.5'}")
    print(f"Pixel Accuracy : {mean_acc:.4f}")
    print("=" * 50)
    for name, iou in zip(class_names, avg_class_iou):
        s = f"{iou:.4f}" if not np.isnan(iou) else "N/A "
        print(f"  {name:<20}: {s}")

    save_metrics_summary(results, args.output_dir)
    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()