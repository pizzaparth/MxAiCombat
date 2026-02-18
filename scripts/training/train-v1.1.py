"""
Clean & Optimized Segmentation Training Script
DINOv2 backbone + ConvNeXt-style segmentation head
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm


# ============================================================
# Mask Mapping
# ============================================================

value_map = {
    0: 0,
    100: 1,
    200: 2,
    300: 3,
    500: 4,
    550: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}
n_classes = len(value_map)


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================
# Dataset (FIXED)
# ============================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, image_transform, w, h):
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.mask_dir = os.path.join(data_dir, "Segmentation")
        self.ids = os.listdir(self.image_dir)
        self.transform = image_transform
        self.w = w
        self.h = h

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        img = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))
        mask = convert_mask(mask)

        img = img.resize((self.w, self.h), Image.BILINEAR)
        mask = mask.resize((self.w, self.h), Image.NEAREST)

        img = self.transform(img)
        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask


# ============================================================
# Model
# ============================================================

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 7, padding=3),
            nn.GELU(),
            nn.Conv2d(128, 128, 7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.net(x)


# ============================================================
# Training
# ============================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    batch_size = 8
    n_epochs = 30

    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, "Offroad_Segmentation_Training_Dataset/train")
    val_dir = os.path.join(script_dir, "Offroad_Segmentation_Training_Dataset/val")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    train_set = MaskDataset(train_dir, train_transform, w, h)
    val_set = MaskDataset(val_dir, val_transform, w, h)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print("Train samples:", len(train_set))
    print("Val samples:", len(val_set))

    # Load DINOv2
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    backbone.eval().to(device)

    # Freeze backbone (WINNING STRATEGY)
    for p in backbone.parameters():
        p.requires_grad = False

    # Get embedding dim
    sample, _ = next(iter(train_loader))
    with torch.no_grad():
        tokens = backbone.forward_features(sample.to(device))["x_norm_patchtokens"]
    embed_dim = tokens.shape[2]

    model = SegmentationHead(embed_dim, n_classes,
                             w // 14, h // 14).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    print("\nStarting Training...\n")

    for epoch in range(n_epochs):

        # ================= TRAIN =================
        model.train()
        train_loss = 0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):

            imgs = imgs.to(device)
            masks = masks.to(device)

            with torch.no_grad():
                tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]

            logits = model(tokens)
            logits = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear")

            loss = loss_fn(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ================= VALIDATE =================
        model.eval()

        total_correct = 0
        total_pixels = 0
        intersection = torch.zeros(n_classes).to(device)
        union = torch.zeros(n_classes).to(device)

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):

                imgs = imgs.to(device)
                masks = masks.to(device)

                tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(tokens)
                logits = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear")

                preds = torch.argmax(logits, dim=1)

                total_correct += (preds == masks).sum()
                total_pixels += torch.numel(masks)

                for cls in range(n_classes):
                    pred_inds = preds == cls
                    target_inds = masks == cls
                    intersection[cls] += (pred_inds & target_inds).sum()
                    union[cls] += (pred_inds | target_inds).sum()

        pixel_acc = (total_correct / total_pixels).item()
        iou = (intersection / (union + 1e-6)).mean().item()

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Pixel Acc: {pixel_acc:.4f}")
        print(f"Val mIoU: {iou:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
