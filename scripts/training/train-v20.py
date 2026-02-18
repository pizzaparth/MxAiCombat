"""
Segmentation Training Script — v19
Hardware: Intel Core 5 210H, 16 GB RAM, 4 GB VRAM

Fixes vs v18:
  - CRITICAL FIX: Augmentation now actually applied during training.
    v18 cached features ONCE (before training), so augmentation in __getitem__
    was frozen on the first pass and never varied. Now backbone runs live on
    augmented images each epoch, caching only val features (backbone is frozen
    so val features never change).
  - Vertical flip probability reduced from 70% → 10% (outdoor scenes have
    fixed sky/ground orientation — frequent vflip destroyed spatial priors).
  - Added random crop augmentation for more spatial diversity.
  - Combined CE + Dice loss → direct IoU-proxy loss → higher mIoU.
  - Linear warmup added before cosine LR decay → more stable early training.
  - Unified IoU metric function (same logic in train and test scripts).
  - Increased patience 7 → 10 to let cosine schedule fully converge.
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
        self.data_ids  = sorted(os.listdir(self.image_dir))
        self.img_size  = img_size  # (H, W)
        self.augment   = augment

        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_resize = transforms.Resize(
            img_size, interpolation=transforms.InterpolationMode.NEAREST
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask  = Image.open(os.path.join(self.masks_dir, data_id))
        mask  = convert_mask(mask)

        if self.augment:
            # Resize first so crop is well-defined
            image = TF.resize(image, self.img_size)
            mask  = TF.resize(mask,  self.img_size,
                              interpolation=TF.InterpolationMode.NEAREST)

            # Random horizontal flip (50%) — safe for outdoor scenes
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # Random vertical flip (10% only) — sky/ground usually fixed
            if random.random() > 0.9:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)

            # Random crop (80–100% of image) — adds scale variation
            H, W = self.img_size
            scale = random.uniform(0.80, 1.0)
            ch, cw = int(H * scale), int(W * scale)
            i = random.randint(0, H - ch)
            j = random.randint(0, W - cw)
            image = TF.crop(image, i, j, ch, cw)
            mask  = TF.crop(mask,  i, j, ch, cw)
            image = TF.resize(image, self.img_size)
            mask  = TF.resize(mask,  self.img_size,
                              interpolation=TF.InterpolationMode.NEAREST)

            # Color jitter — image only
            if random.random() > 0.5:
                image = self.color_jitter(image)

            # Convert to tensor + normalize
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(image)
        else:
            image = self.img_transform(image)
            mask  = self.mask_resize(mask)

        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask


# ============================================================================
# Segmentation Head (ConvNeXt-style, 3 residual blocks)
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
# Combined CE + Dice Loss  (directly optimises IoU-like objective)
# ============================================================================

class CEDiceLoss(nn.Module):
    """
    Combined CrossEntropy + Dice loss.
    Dice loss directly optimises an IoU-like metric and is especially helpful
    for imbalanced classes. CE handles calibration; Dice handles overlap.
    """
    def __init__(self, weight=None, label_smoothing=0.05, dice_weight=0.5):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(weight=weight,
                                        label_smoothing=label_smoothing)
        self.dice_weight = dice_weight
        self.n_classes   = n_classes

    def dice_loss(self, logits, targets):
        # logits: (B, C, H, W) — before softmax
        probs = F.softmax(logits, dim=1)           # (B, C, H, W)
        one_hot = F.one_hot(targets, self.n_classes)   # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dim=dims)
        union = (probs + one_hot).sum(dim=dims)
        dice  = (2.0 * inter + 1e-6) / (union + 1e-6)
        return 1.0 - dice.mean()

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice_weight * self.dice_loss(logits, targets)


# ============================================================================
# Metrics  (unified — same logic used in test script)
# ============================================================================

def compute_iou(pred_logits_or_classes, target, num_classes=10, is_logits=True):
    """
    Accepts either raw logits (B,C,H,W) or argmaxed class maps (B,H,W).
    Returns scalar mean IoU over present classes.
    """
    if is_logits:
        p = torch.argmax(pred_logits_or_classes, dim=1).view(-1)
    else:
        p = pred_logits_or_classes.view(-1)
    t = target.view(-1)
    iou_list = []
    for c in range(num_classes):
        inter = ((p == c) & (t == c)).sum().float()
        union = ((p == c) | (t == c)).sum().float()
        if union > 0:
            iou_list.append((inter / union).item())
    return float(np.nanmean(iou_list)) if iou_list else 0.0


def compute_iou_per_class(pred_classes, target, num_classes=10):
    """Returns (mean_iou, [per_class_iou])."""
    p = pred_classes.view(-1)
    t = target.view(-1)
    per_class = []
    for c in range(num_classes):
        inter = ((p == c) & (t == c)).sum().float()
        union = ((p == c) | (t == c)).sum().float()
        per_class.append((inter / union).item() if union > 0 else float('nan'))
    valid = [x for x in per_class if not np.isnan(x)]
    return (float(np.mean(valid)) if valid else 0.0), per_class


def compute_pixel_accuracy(pred, target):
    if pred.dim() == 4:   # logits
        pred = torch.argmax(pred, dim=1)
    return (pred == target).float().mean().item()


# ============================================================================
# Val feature caching  (backbone frozen → val features never change)
# ============================================================================

@torch.no_grad()
def cache_features(backbone, loader, device):
    """Cache backbone features for the validation set (no augmentation)."""
    backbone.eval()
    all_feats, all_labels, all_sizes = [], [], []
    for imgs, labels in tqdm(loader, desc="Caching val features", leave=False):
        imgs = imgs.to(device)
        with torch.cuda.amp.autocast():
            feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
        all_feats.append(feats.cpu())
        all_labels.append(labels.cpu())
        all_sizes.append(imgs.shape[2:])
    return all_feats, all_labels, all_sizes


@torch.no_grad()
def evaluate_cached(classifier, cached_features, labels_list, device, img_sizes):
    """Fast evaluation on cached val features."""
    classifier.eval()
    ious, accs = [], []
    for feats, labels, img_sz in zip(cached_features, labels_list, img_sizes):
        feats  = feats.to(device)
        labels = labels.to(device)
        logits  = classifier(feats)
        outputs = F.interpolate(logits, size=img_sz, mode="bilinear", align_corners=False)
        ious.append(compute_iou(outputs, labels, is_logits=True))
        accs.append(compute_pixel_accuracy(outputs, labels))
    classifier.train()
    return float(np.mean(ious)), float(np.mean(accs))


# ============================================================================
# LR Warmup + Cosine Scheduler
# ============================================================================

def get_scheduler_with_warmup(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    """Linear warmup for warmup_epochs, then cosine annealing."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        # Cosine from 1.0 → eta_min/lr after warmup
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
        base_lr  = optimizer.param_groups[0]['initial_lr']
        return (eta_min + (base_lr - eta_min) * cosine) / base_lr
    # Store initial_lr so lambda can read it
    for pg in optimizer.param_groups:
        pg['initial_lr'] = pg['lr']
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Plots
# ============================================================================

def save_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'],   label='val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history['train_iou'], label='train')
    axes[1].plot(history['val_iou'],   label='val')
    axes[1].axhline(0.5, color='red', linestyle='--', label='IoU=0.5 target')
    axes[1].set_title('Mean IoU'); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(history['train_acc'], label='train')
    axes[2].plot(history['val_acc'],   label='val')
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
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Hyperparameters ───────────────────────────────────────────────────────
    w          = int(((960 / 2) // 14) * 14)   # 476
    h          = int(((540 / 2) // 14) * 14)   # 266
    batch_size = 4
    lr         = 3e-4
    n_epochs   = 40
    warmup_ep  = 3      # linear warmup epochs before cosine decay
    patience   = 10     # early stopping patience (longer = lets cosine converge)
    hidden_dim = 256

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    data_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir  = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')

    # NOTE: augment=True here — augmentation now runs live during training
    # (features are NOT pre-cached for train, only for val)
    trainset = MaskDataset(data_dir, img_size=(h, w), augment=True)
    valset   = MaskDataset(val_dir,  img_size=(h, w), augment=False)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(valset,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"Train: {len(trainset)} | Val: {len(valset)}")

    # ── Backbone (frozen) ─────────────────────────────────────────────────────
    print("Loading DINOv2-small backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    print("Backbone loaded and frozen.")

    dummy = torch.zeros(1, 3, h, w, device=device)
    with torch.no_grad():
        n_embedding = backbone.forward_features(dummy)["x_norm_patchtokens"].shape[2]
    tokenH, tokenW = h // 14, w // 14
    print(f"Embedding dim: {n_embedding} | Token grid: {tokenH}×{tokenW}")

    # ── Cache VAL features only (backbone frozen → they never change) ─────────
    print("\nCaching val backbone features (one-time cost)...")
    val_feats, val_labels, val_sizes = cache_features(backbone, val_loader, device)
    print("Val feature caching complete!")

    # ── Segmentation head ─────────────────────────────────────────────────────
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding, out_channels=n_classes,
        tokenW=tokenW, tokenH=tokenH, hidden=hidden_dim
    ).to(device)

    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"Segmentation head params: {total_params:,}")

    # ── Loss: CE + Dice with class weighting ──────────────────────────────────
    # Compute class weights from val labels (train labels not cached anymore)
    all_label_flat = torch.cat([l.view(-1) for l in val_labels])
    class_counts   = torch.bincount(all_label_flat, minlength=n_classes).float()
    # Use sqrt-inverse freq (less extreme than pure inverse) for stability
    class_weights  = 1.0 / (class_counts.sqrt() + 1)
    class_weights  = (class_weights / class_weights.sum() * n_classes).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    loss_fct = CEDiceLoss(weight=class_weights, label_smoothing=0.05, dice_weight=0.5)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_scheduler_with_warmup(optimizer, warmup_ep, n_epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # ── Training loop ──────────────────────────────────────────────────────────
    history = {k: [] for k in ['train_loss', 'val_loss', 'train_iou', 'val_iou',
                                'train_acc',  'val_acc']}
    best_val_iou = 0.0
    no_improve   = 0
    best_path    = os.path.join(script_dir, 'segmentation_head_best.pth')

    print(f"\nTraining for up to {n_epochs} epochs "
          f"(warmup={warmup_ep}, early-stop patience={patience})")
    print("=" * 70)

    for epoch in range(n_epochs):
        classifier.train()
        epoch_losses, epoch_ious, epoch_accs = [], [], []

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1:02d}/{n_epochs} [Train]",
                    leave=False)

        for imgs, labels in pbar:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                # Run backbone live — augmented images vary each epoch
                with torch.no_grad():
                    feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits  = classifier(feats)
                outputs = F.interpolate(logits, size=imgs.shape[2:],
                                        mode="bilinear", align_corners=False)
                loss = loss_fct(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())
            with torch.no_grad():
                epoch_ious.append(compute_iou(outputs, labels, is_logits=True))
                epoch_accs.append(compute_pixel_accuracy(outputs, labels))
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # ── Validation ──────────────────────────────────────────────────────
        # Val loss computed on cached features for speed
        classifier.eval()
        val_losses = []
        with torch.no_grad():
            for feats, labels, img_sz in zip(val_feats, val_labels, val_sizes):
                feats  = feats.to(device)
                labels = labels.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    logits  = classifier(feats)
                    outputs = F.interpolate(logits, size=img_sz,
                                            mode="bilinear", align_corners=False)
                    val_losses.append(loss_fct(outputs, labels).item())

        val_iou, val_acc = evaluate_cached(
            classifier, val_feats, val_labels, device, val_sizes)

        t_loss    = float(np.mean(epoch_losses))
        v_loss    = float(np.mean(val_losses))
        train_iou = float(np.mean(epoch_ious))
        train_acc = float(np.mean(epoch_accs))
        lr_now    = optimizer.param_groups[0]['lr']

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

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            no_improve   = 0
            torch.save(classifier.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs)")
                break

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(script_dir, 'segmentation_head.pth')
    torch.save(classifier.state_dict(), final_path)
    save_plots(history, output_dir)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Best Val IoU : {best_val_iou:.4f}  "
          f"{'✓ ABOVE 0.5 TARGET' if best_val_iou > 0.5 else '✗ below 0.5'}")
    print(f"  Best model   : {best_path}")
    print(f"  Final model  : {final_path}")
    print(f"  Plots        : {output_dir}/training_curves.png")
    print("=" * 70)


if __name__ == "__main__":
    main()