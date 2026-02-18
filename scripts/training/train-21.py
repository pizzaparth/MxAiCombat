"""
Segmentation Training Script — v20
Hardware: Intel Core 5 210H, 16 GB RAM, 4 GB VRAM
Target: mIoU > 0.5

Major upgrades vs v19:
  1. dinov2_vitb14 (base) instead of vits14 (small)
       → 768-dim embeddings vs 384 → richer features → biggest single IoU jump
       → Still fits 4GB VRAM with AMP + batch_size=2
  2. Partially unfrozen backbone (last 4 transformer blocks + norm)
       → Adapts DINOv2 to offroad domain instead of using generic ImageNet features
       → Separate lower LR for backbone vs head (10x smaller)
  3. Multi-scale feature decoder with skip connections
       → Hooks into intermediate DINOv2 layers (not just final layer)
       → Fuses coarse + fine features via lateral connections → better boundaries
  4. Focal Loss + Dice (replaces plain CE + Dice)
       → Focal term down-weights easy background pixels (γ=2)
       → Forces model to learn rare classes: Logs, Ground Clutter, Dry Bushes
  5. OneCycleLR scheduler
       → Reaches higher peak LR then anneals aggressively → faster convergence
  6. Gradient accumulation (effective_batch = 8)
       → Simulates larger batch on 4GB VRAM without OOM
  7. Class-weighted sampling
       → Oversamples images containing rare classes so loss sees them more often
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
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

# Rare classes that need extra attention
RARE_CLASSES = {5, 6, 7}   # Ground Clutter, Logs, Rocks


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, img_size, augment=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids  = sorted(os.listdir(self.image_dir))
        self.img_size  = img_size
        self.augment   = augment

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08
        )
        self.mask_resize = transforms.Resize(
            img_size, interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.data_ids)

    def get_sample_weights(self):
        """
        Return per-sample weights for WeightedRandomSampler.
        Images containing rare classes get 3x higher sampling probability.
        """
        weights = []
        for data_id in self.data_ids:
            mask = Image.open(os.path.join(self.masks_dir, data_id))
            mask = np.array(convert_mask(mask))
            has_rare = any((mask == c).any() for c in RARE_CLASSES)
            weights.append(3.0 if has_rare else 1.0)
        return weights

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask  = Image.open(os.path.join(self.masks_dir, data_id))
        mask  = convert_mask(mask)

        H, W = self.img_size

        if self.augment:
            # Resize to target size first
            image = TF.resize(image, self.img_size)
            mask  = TF.resize(mask,  self.img_size,
                              interpolation=TF.InterpolationMode.NEAREST)

            # Horizontal flip (50%)
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # Random scale crop (75–100%) — bigger range for more variation
            scale = random.uniform(0.75, 1.0)
            ch, cw = int(H * scale), int(W * scale)
            i = random.randint(0, H - ch)
            j = random.randint(0, W - cw)
            image = TF.crop(image, i, j, ch, cw)
            mask  = TF.crop(mask,  i, j, ch, cw)
            image = TF.resize(image, self.img_size)
            mask  = TF.resize(mask,  self.img_size,
                              interpolation=TF.InterpolationMode.NEAREST)

            # Random rotation ±10°
            if random.random() < 0.3:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle)
                mask  = TF.rotate(mask,  angle,
                                  interpolation=TF.InterpolationMode.NEAREST)

            # Gaussian blur (simulates camera defocus)
            if random.random() < 0.2:
                image = image.filter(
                    __import__('PIL').ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5))
                )

            # Color jitter (image only)
            if random.random() < 0.6:
                image = self.color_jitter(image)

            # Random grayscale (10%) — helps with color-invariant features
            if random.random() < 0.1:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)

        else:
            image = TF.resize(image, self.img_size)
            mask  = self.mask_resize(mask)

        image = transforms.ToTensor()(image)
        image = self.normalize(image)
        mask  = torch.from_numpy(np.array(mask)).long()
        return image, mask


# ============================================================================
# Multi-Scale DINOv2 Feature Extractor
# ============================================================================

class DINOv2MultiScale(nn.Module):
    """
    Wraps DINOv2 and extracts features from multiple intermediate layers.
    Hooks into blocks at 1/4, 1/2, 3/4, and final depth.
    For vitb14 (12 blocks): hooks at blocks 2, 5, 8, 11.
    """
    def __init__(self, backbone, n_blocks=12):
        super().__init__()
        self.backbone = backbone
        self.n_blocks = n_blocks
        self._features = {}

        # Hook indices: quarter, half, three-quarter, final
        hook_idxs = [
            max(0, n_blocks // 4 - 1),
            max(0, n_blocks // 2 - 1),
            max(0, 3 * n_blocks // 4 - 1),
            n_blocks - 1,
        ]
        self.hook_idxs = sorted(set(hook_idxs))

        for idx in self.hook_idxs:
            backbone.blocks[idx].register_forward_hook(
                self._make_hook(idx)
            )

    def _make_hook(self, idx):
        def hook(module, input, output):
            # output shape: (B, N+1, C) — N patches + CLS token
            self._features[idx] = output[:, 1:, :]   # drop CLS
        return hook

    def forward(self, x):
        self._features = {}
        _ = self.backbone(x)   # triggers hooks
        return [self._features[i] for i in self.hook_idxs]


# ============================================================================
# FPN-style Segmentation Head
# ============================================================================

class FPNSegHead(nn.Module):
    """
    Feature Pyramid Network decoder.
    Takes multi-scale DINOv2 tokens, fuses coarse-to-fine, predicts classes.

    Architecture:
      - Each scale: 1×1 lateral conv to reduce to `fpn_ch` channels
      - Top-down: upsample coarser feature, add to finer → 3× ConvNeXt block
      - Final: upsample to full resolution, classify
    """
    def __init__(self, in_channels, out_classes, tokenH, tokenW,
                 n_scales=4, fpn_ch=256):
        super().__init__()
        self.tokenH  = tokenH
        self.tokenW  = tokenW
        self.n_scales = n_scales

        # Lateral projections: embed_dim → fpn_ch, one per scale
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, fpn_ch),
                nn.LayerNorm(fpn_ch),
            )
            for _ in range(n_scales)
        ])

        # After top-down fusion: ConvNeXt refinement blocks
        def conv_block(ch):
            return nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=7, padding=3, groups=ch),
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.Conv2d(ch, ch * 2, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(ch * 2, ch, kernel_size=1),
            )

        self.fpn_blocks = nn.ModuleList([
            conv_block(fpn_ch) for _ in range(n_scales)
        ])

        # Final classification head (at token resolution, then upsample)
        self.classifier = nn.Sequential(
            nn.Conv2d(fpn_ch, fpn_ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_ch // 2),
            nn.GELU(),
            nn.Conv2d(fpn_ch // 2, out_classes, kernel_size=1),
        )

    def _to_spatial(self, tokens):
        """(B, N, C) → (B, C, tokenH, tokenW)"""
        B, N, C = tokens.shape
        return tokens.reshape(B, self.tokenH, self.tokenW, C).permute(0, 3, 1, 2)

    def forward(self, multi_scale_tokens):
        """multi_scale_tokens: list of (B, N, C) from coarsest to finest."""
        # Project each scale to fpn_ch
        laterals = [
            self._to_spatial(self.laterals[i](multi_scale_tokens[i]))
            for i in range(self.n_scales)
        ]

        # Top-down fusion: start from coarsest (index 0), add to finer
        # laterals[0] = coarsest (early layers), laterals[-1] = finest (last layer)
        # We actually want finest first for top-down, so reverse
        feats = list(reversed(laterals))   # finest first

        # Top-down: fuse finest down to coarsest
        out = feats[0] + self.fpn_blocks[0](feats[0])
        for i in range(1, self.n_scales):
            # Upsample previous result to match next (already same size here
            # since all DINOv2 tokens are the same spatial size — but kept
            # for future multi-res support)
            out = out + feats[i] + self.fpn_blocks[i](feats[i])

        return self.classifier(out)   # (B, n_classes, tokenH, tokenW)


# ============================================================================
# Focal + Dice Loss
# ============================================================================

class FocalDiceLoss(nn.Module):
    """
    Focal Loss (handles class imbalance by down-weighting easy pixels)
    + Dice Loss (directly optimises overlap / IoU proxy).

    Focal(p_t) = -α_t (1 - p_t)^γ log(p_t)
    γ=2 is the standard value from the original paper.
    """
    def __init__(self, weight=None, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.gamma       = gamma
        self.dice_weight = dice_weight
        self.weight      = weight   # class weights tensor

    def focal_loss(self, logits, targets):
        # Standard CE — then modulate by (1 - p_t)^gamma
        ce_loss = F.cross_entropy(logits, targets,
                                  weight=self.weight,
                                  reduction='none')   # (B, H, W)
        pt = torch.exp(-ce_loss)   # probability of correct class
        focal = (1 - pt) ** self.gamma * ce_loss
        return focal.mean()

    def dice_loss(self, logits, targets):
        probs   = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, n_classes).permute(0, 3, 1, 2).float()
        dims    = (0, 2, 3)
        inter   = (probs * one_hot).sum(dim=dims)
        union   = (probs + one_hot).sum(dim=dims)
        dice    = (2.0 * inter + 1e-6) / (union + 1e-6)
        return 1.0 - dice.mean()

    def forward(self, logits, targets):
        return (self.focal_loss(logits, targets)
                + self.dice_weight * self.dice_loss(logits, targets))


# ============================================================================
# Metrics (unified — same in train and test)
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


def compute_pixel_accuracy(pred_classes, target):
    return (pred_classes == target).float().mean().item()


# ============================================================================
# Val feature caching
# ============================================================================

@torch.no_grad()
def cache_val_features(extractor, loader, device):
    extractor.eval()
    all_feats, all_labels, all_sizes = [], [], []
    for imgs, labels in tqdm(loader, desc="Caching val features", leave=False):
        imgs = imgs.to(device)
        with torch.cuda.amp.autocast():
            feats = extractor(imgs)   # list of (B, N, C)
        # Store as CPU tensors
        all_feats.append([f.cpu() for f in feats])
        all_labels.append(labels.cpu())
        all_sizes.append(imgs.shape[2:])
    return all_feats, all_labels, all_sizes


@torch.no_grad()
def evaluate_cached(head, cached_feats, labels_list, device, img_sizes):
    head.eval()
    ious, accs = [], []
    for feats_list, labels, img_sz in zip(cached_feats, labels_list, img_sizes):
        feats_list = [f.to(device) for f in feats_list]
        labels     = labels.to(device)
        logits  = head(feats_list)
        outputs = F.interpolate(logits, size=img_sz, mode="bilinear", align_corners=False)
        preds   = torch.argmax(outputs, dim=1)
        ious.append(compute_iou(preds, labels))
        accs.append(compute_pixel_accuracy(preds, labels))
    head.train()
    return float(np.mean(ious)), float(np.mean(accs))


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
    axes[1].axhline(0.5, color='red', linestyle='--', label='0.5 target')
    axes[1].set_title('Mean IoU'); axes[1].legend(); axes[1].grid(True)
    axes[2].plot(history['train_acc'], label='train')
    axes[2].plot(history['val_acc'],   label='val')
    axes[2].set_title('Pixel Accuracy'); axes[2].legend(); axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_dir}/training_curves.png")


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
    w              = int(((960 / 2) // 14) * 14)   # 476
    h              = int(((540 / 2) // 14) * 14)   # 266
    batch_size     = 2        # vitb14 is larger; AMP keeps VRAM safe
    accum_steps    = 4        # effective batch = 2 × 4 = 8
    lr_head        = 5e-4     # head learning rate
    lr_backbone    = 5e-5     # backbone fine-tune LR (10× smaller)
    n_epochs       = 50
    patience       = 12
    fpn_ch         = 256
    unfreeze_blocks = 4       # how many DINOv2 tail blocks to fine-tune

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    data_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir  = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')

    trainset = MaskDataset(data_dir, img_size=(h, w), augment=True)
    valset   = MaskDataset(val_dir,  img_size=(h, w), augment=False)

    # Weighted sampler — oversample rare-class images
    print("Computing sample weights for rare-class oversampling...")
    sample_weights = trainset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(valset,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"Train: {len(trainset)} | Val: {len(valset)}")

    # ── Backbone: DINOv2-Base ─────────────────────────────────────────────────
    print("Loading DINOv2-base backbone (vitb14)...")
    _backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    _backbone.to(device)

    n_dino_blocks = len(_backbone.blocks)
    print(f"DINOv2-base: {n_dino_blocks} transformer blocks")

    # Freeze all backbone params first
    for p in _backbone.parameters():
        p.requires_grad = False

    # Unfreeze last `unfreeze_blocks` blocks + norm layer
    for blk in _backbone.blocks[-unfreeze_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in _backbone.norm.parameters():
        p.requires_grad = True

    trainable_bb = sum(p.numel() for p in _backbone.parameters() if p.requires_grad)
    print(f"Backbone trainable params: {trainable_bb:,} "
          f"(last {unfreeze_blocks} blocks + norm unfrozen)")

    # Probe embedding dim
    dummy = torch.zeros(1, 3, h, w, device=device)
    with torch.no_grad():
        n_embedding = _backbone.forward_features(dummy)["x_norm_patchtokens"].shape[2]
    tokenH, tokenW = h // 14, w // 14
    print(f"Embedding dim: {n_embedding} | Token grid: {tokenH}×{tokenW}")

    # Wrap backbone for multi-scale extraction
    extractor = DINOv2MultiScale(_backbone, n_blocks=n_dino_blocks).to(device)

    # ── Cache val features (unfrozen blocks mean we re-cache each epoch — see loop) ──
    # Actually, because backbone blocks are now trainable, val features change each
    # epoch, so we must NOT cache val features permanently.
    # Instead we run a full val pass each epoch (it's small enough).
    print("Val features will be computed each epoch (backbone is partially trainable).")

    # ── Segmentation Head ─────────────────────────────────────────────────────
    n_scales = len(extractor.hook_idxs)
    head = FPNSegHead(
        in_channels=n_embedding,
        out_classes=n_classes,
        tokenH=tokenH,
        tokenW=tokenW,
        n_scales=n_scales,
        fpn_ch=fpn_ch,
    ).to(device)

    head_params = sum(p.numel() for p in head.parameters())
    print(f"FPN head params: {head_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Compute class weights from a quick scan of val masks
    print("Computing class weights...")
    all_counts = torch.zeros(n_classes)
    for _, labels in val_loader:
        all_counts += torch.bincount(labels.view(-1), minlength=n_classes).float()
    class_weights = 1.0 / (all_counts.sqrt() + 1)
    class_weights = (class_weights / class_weights.sum() * n_classes).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    loss_fct = FocalDiceLoss(weight=class_weights, gamma=2.0, dice_weight=0.5)

    # ── Optimizer: separate param groups for backbone vs head ─────────────────
    optimizer = optim.AdamW([
        {'params': [p for p in _backbone.parameters() if p.requires_grad],
         'lr': lr_backbone, 'weight_decay': 1e-4},
        {'params': head.parameters(),
         'lr': lr_head, 'weight_decay': 1e-4},
    ])

    total_steps = (len(train_loader) // accum_steps) * n_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr_backbone, lr_head],
        total_steps=total_steps,
        pct_start=0.1,        # 10% warmup
        anneal_strategy='cos',
        div_factor=25,        # start lr = max_lr / 25
        final_div_factor=1e4, # end lr = start_lr / 1e4
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {k: [] for k in ['train_loss', 'val_loss', 'train_iou', 'val_iou',
                                'train_acc',  'val_acc']}
    best_val_iou = 0.0
    no_improve   = 0
    best_path    = os.path.join(script_dir, 'segmentation_head_best.pth')

    print(f"\nTraining for up to {n_epochs} epochs "
          f"(accum={accum_steps}, patience={patience})")
    print("=" * 70)

    for epoch in range(n_epochs):
        extractor.train()
        _backbone.train()
        head.train()

        epoch_losses, epoch_ious, epoch_accs = [], [], []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch {epoch+1:02d}/{n_epochs} [Train]",
                    leave=False)

        for step, (imgs, labels) in pbar:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                multi_feats = extractor(imgs)
                logits  = head(multi_feats)
                outputs = F.interpolate(logits, size=imgs.shape[2:],
                                        mode="bilinear", align_corners=False)
                loss = loss_fct(outputs, labels) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(_backbone.parameters()) + list(head.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            with torch.no_grad():
                epoch_losses.append(loss.item() * accum_steps)
                preds = torch.argmax(outputs, dim=1)
                epoch_ious.append(compute_iou(preds, labels))
                epoch_accs.append(compute_pixel_accuracy(preds, labels))
            pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

        # ── Validation ────────────────────────────────────────────────────────
        extractor.eval()
        head.eval()
        val_losses, val_ious, val_accs = [], [], []

        with torch.no_grad():
            for imgs_v, labels_v in val_loader:
                imgs_v   = imgs_v.to(device)
                labels_v = labels_v.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    multi_feats = extractor(imgs_v)
                    logits_v    = head(multi_feats)
                    outputs_v   = F.interpolate(logits_v, size=imgs_v.shape[2:],
                                                mode="bilinear", align_corners=False)
                    val_losses.append(loss_fct(outputs_v, labels_v).item())
                preds_v = torch.argmax(outputs_v, dim=1)
                val_ious.append(compute_iou(preds_v, labels_v))
                val_accs.append(compute_pixel_accuracy(preds_v, labels_v))

        t_loss    = float(np.mean(epoch_losses))
        v_loss    = float(np.mean(val_losses))
        train_iou = float(np.mean(epoch_ious))
        val_iou   = float(np.mean(val_ious))
        train_acc = float(np.mean(epoch_accs))
        val_acc   = float(np.mean(val_accs))
        lr_now    = optimizer.param_groups[1]['lr']   # head LR

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
            # Save both backbone (unfrozen blocks) + head
            torch.save({
                'head':     head.state_dict(),
                'backbone': _backbone.state_dict(),
                'val_iou':  val_iou,
                'epoch':    epoch + 1,
            }, best_path)
            print(f"  → Saved best model (IoU={val_iou:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs)")
                break

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(script_dir, 'segmentation_head.pth')
    torch.save({
        'head':     head.state_dict(),
        'backbone': _backbone.state_dict(),
        'val_iou':  best_val_iou,
    }, final_path)

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