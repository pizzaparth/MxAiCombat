"""
Segmentation Training Script — Optimized for 4GB GPU / IoU > 0.5
Hardware: Intel Core 5 210H, 16 GB RAM, 4 GB VRAM

Key optimizations:
  - Mixed-precision (AMP FP16) → halves VRAM, ~2x faster
  - Backbone feature caching per epoch → avoids repeated DINOv2 forward passes
  - AdamW + CosineAnnealingLR → much better convergence than SGD
  - Deeper ConvNeXt head → higher capacity segmentation head
  - Class-weighted loss + label smoothing → handles class imbalance
  - Data augmentation (flips, color jitter) → reduces overfitting
  - Best-checkpoint saving → always keep the best IoU model
  - num_workers + pin_memory → faster data loading
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import os
import random
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
n_classes = len(value_map)


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset with Augmentation
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, img_size, augment=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids = sorted(os.listdir(self.image_dir))
        self.img_size = img_size  # (H, W)
        self.augment = augment

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
        mask = Image.open(os.path.join(self.masks_dir, data_id))
        mask = convert_mask(mask)

        # Augmentation (joint, so spatial ops are applied to both)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # Random vertical flip
            if random.random() > 0.3:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            # Color jitter on image only
            if random.random() > 0.5:
                image = transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
                )(image)

        image = self.img_transform(image)
        mask = self.mask_resize(mask)
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask


# ============================================================================
# Deeper Segmentation Head (ConvNeXt-style, 3 blocks)
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
        x = x + self.block1(x)   # residual connections
        x = x + self.block2(x)
        x = x + self.block3(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    iou_list = []
    for c in range(num_classes):
        p = pred == c
        t = target == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        if union > 0:
            iou_list.append((inter / union).item())
    return float(np.nanmean(iou_list)) if iou_list else 0.0


def compute_pixel_accuracy(pred, target):
    return (torch.argmax(pred, dim=1) == target).float().mean().item()


@torch.no_grad()
def evaluate(classifier, cached_features, labels_list, device, img_sizes):
    """Evaluate using pre-cached backbone features."""
    classifier.eval()
    ious, accs = [], []
    for feats, labels, img_sz in zip(cached_features, labels_list, img_sizes):
        feats = feats.to(device)
        labels = labels.to(device)
        logits = classifier(feats)
        outputs = F.interpolate(logits, size=img_sz, mode="bilinear", align_corners=False)
        ious.append(compute_iou(outputs, labels))
        accs.append(compute_pixel_accuracy(outputs, labels))
    classifier.train()
    return float(np.mean(ious)), float(np.mean(accs))


# ============================================================================
# Feature Caching
# ============================================================================

@torch.no_grad()
def cache_features(backbone, loader, device):
    """Pre-extract all backbone features. Massive speedup since backbone is frozen."""
    backbone.eval()
    all_feats, all_labels, all_sizes = [], [], []
    for imgs, labels in tqdm(loader, desc="Caching features", leave=False):
        imgs = imgs.to(device)
        with torch.cuda.amp.autocast():
            feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
        all_feats.append(feats.cpu())
        all_labels.append(labels.cpu())
        all_sizes.append(imgs.shape[2:])
    return all_feats, all_labels, all_sizes


# ============================================================================
# Plots
# ============================================================================

def save_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'], label='val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history['train_iou'], label='train')
    axes[1].plot(history['val_iou'], label='val')
    axes[1].axhline(0.5, color='red', linestyle='--', label='IoU=0.5 target')
    axes[1].set_title('Mean IoU'); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(history['train_acc'], label='train')
    axes[2].plot(history['val_acc'], label='val')
    axes[2].set_title('Pixel Accuracy'); axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {output_dir}/training_curves.png")


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Hyperparameters ──────────────────────────────────────────────────────
    # Reduced resolution fits 4GB VRAM while keeping patch alignment (mult of 14)
    w = int(((960 / 2) // 14) * 14)   # 476
    h = int(((540 / 2) // 14) * 14)   # 266 (was 336, ~20% fewer tokens)
    batch_size = 4          # AMP allows larger batch on 4GB
    lr = 3e-4
    n_epochs = 30
    patience = 7            # early stopping
    hidden_dim = 256        # segmentation head width

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    data_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir  = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')

    trainset = MaskDataset(data_dir, img_size=(h, w), augment=True)
    valset   = MaskDataset(val_dir,  img_size=(h, w), augment=False)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(valset,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"Train: {len(trainset)} | Val: {len(valset)}")

    # ── Backbone (frozen) ────────────────────────────────────────────────────
    print("Loading DINOv2-small backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    print("Backbone loaded and frozen.")

    # Probe embedding dim
    dummy = torch.zeros(1, 3, h, w, device=device)
    with torch.no_grad():
        n_embedding = backbone.forward_features(dummy)["x_norm_patchtokens"].shape[2]
    tokenH, tokenW = h // 14, w // 14
    print(f"Embedding dim: {n_embedding} | Token grid: {tokenH}×{tokenW}")

    # ── Cache backbone features (HUGE speedup for frozen backbone) ───────────
    print("\nCaching train backbone features (one-time cost)...")
    train_feats, train_labels, train_sizes = cache_features(backbone, train_loader, device)
    print("Caching val backbone features...")
    val_feats,   val_labels,   val_sizes   = cache_features(backbone, val_loader,   device)
    print("Feature caching complete!")

    # ── Segmentation head ────────────────────────────────────────────────────
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding, out_channels=n_classes,
        tokenW=tokenW, tokenH=tokenH, hidden=hidden_dim
    ).to(device)

    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"Segmentation head params: {total_params:,}")

    # ── Loss: weighted CrossEntropy to handle class imbalance ────────────────
    # Compute class frequencies from cached labels for weighting
    all_label_flat = torch.cat([l.view(-1) for l in train_labels])
    class_counts = torch.bincount(all_label_flat, minlength=n_classes).float()
    class_weights = 1.0 / (class_counts + 1)  # inverse freq
    class_weights = (class_weights / class_weights.sum() * n_classes).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    loss_fct = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {k: [] for k in ['train_loss', 'val_loss', 'train_iou', 'val_iou',
                                'train_acc',  'val_acc']}
    best_val_iou = 0.0
    no_improve   = 0
    best_path    = os.path.join(script_dir, 'segmentation_head_best.pth')

    print(f"\nTraining for up to {n_epochs} epochs (early stop patience={patience})")
    print("=" * 70)

    for epoch in range(n_epochs):
        classifier.train()
        epoch_losses = []

        pbar = tqdm(zip(train_feats, train_labels, train_sizes),
                    total=len(train_feats),
                    desc=f"Epoch {epoch+1:02d}/{n_epochs} [Train]",
                    leave=False)

        for feats, labels, img_sz in pbar:
            feats  = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits  = classifier(feats)
                outputs = F.interpolate(logits, size=img_sz, mode="bilinear", align_corners=False)
                loss    = loss_fct(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Validation loss
        classifier.eval()
        val_losses = []
        with torch.no_grad():
            for feats, labels, img_sz in zip(val_feats, val_labels, val_sizes):
                feats  = feats.to(device)
                labels = labels.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    logits  = classifier(feats)
                    outputs = F.interpolate(logits, size=img_sz, mode="bilinear", align_corners=False)
                    val_losses.append(loss_fct(outputs, labels).item())

        # Metrics (using cached features, fast)
        train_iou, train_acc = evaluate(classifier, train_feats, train_labels, device, train_sizes)
        val_iou,   val_acc   = evaluate(classifier, val_feats,   val_labels,   device, val_sizes)

        t_loss = float(np.mean(epoch_losses))
        v_loss = float(np.mean(val_losses))
        lr_now = optimizer.param_groups[0]['lr']

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        marker = " ★" if val_iou > best_val_iou else ""
        print(f"Ep {epoch+1:02d} | "
              f"Loss {t_loss:.4f}/{v_loss:.4f} | "
              f"IoU {train_iou:.4f}/{val_iou:.4f} | "
              f"Acc {train_acc:.4f}/{val_acc:.4f} | "
              f"LR {lr_now:.2e}{marker}")

        # Checkpoint best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            no_improve   = 0
            torch.save(classifier.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # ── Final save & summary ──────────────────────────────────────────────────
    final_path = os.path.join(script_dir, 'segmentation_head.pth')
    torch.save(classifier.state_dict(), final_path)

    save_plots(history, output_dir)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Best Val IoU  : {best_val_iou:.4f}  {'✓ ABOVE 0.5 TARGET' if best_val_iou > 0.5 else '✗ below 0.5'}")
    print(f"  Best model    : {best_path}")
    print(f"  Final model   : {final_path}")
    print(f"  Plots         : {output_dir}/training_curves.png")
    print("=" * 70)


if __name__ == "__main__":
    main()