"""
Clean Hackathon-Ready Segmentation Training Script
DINOv2 (Frozen) + Conv Head
Includes:
- Correct mask handling
- Proper validation
- mIoU + Pixel Accuracy
- Graph saving
- No redundant passes
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
n_classes = len(value_map)


# ============================================================
# DATASET
# ============================================================

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        new_arr[arr == raw] = new
    return Image.fromarray(new_arr)


class MaskDataset(Dataset):
    def __init__(self, data_dir, transform, w, h):
        self.img_dir = os.path.join(data_dir, "Color_Images")
        self.mask_dir = os.path.join(data_dir, "Segmentation")
        self.ids = os.listdir(self.img_dir)
        self.transform = transform
        self.w = w
        self.h = h

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))
        mask = convert_mask(mask)

        img = img.resize((self.w, self.h), Image.BILINEAR)
        mask = mask.resize((self.w, self.h), Image.NEAREST)

        img = self.transform(img)
        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask


# ============================================================
# MODEL
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
# TRAINING
# ============================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    batch_size = 8
    n_epochs = 30

    w = 448
    h = 252

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

    # Load backbone
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    backbone.eval().to(device)

    for p in backbone.parameters():
        p.requires_grad = False

    sample, _ = next(iter(train_loader))
    with torch.no_grad():
        tokens = backbone.forward_features(sample.to(device))["x_norm_patchtokens"]
    embed_dim = tokens.shape[2]

    model = SegmentationHead(embed_dim, n_classes,
                             w // 14, h // 14).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_acc": []
    }

    print("\nStarting Training...\n")

    for epoch in range(n_epochs):

        # ================= TRAIN =================
        model.train()
        train_loss = 0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            imgs, masks = imgs.to(device), masks.to(device)

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
        val_loss = 0

        total_correct = 0
        total_pixels = 0
        intersection = torch.zeros(n_classes).to(device)
        union = torch.zeros(n_classes).to(device)

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                imgs, masks = imgs.to(device), masks.to(device)

                tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(tokens)
                logits = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear")

                val_loss += loss_fn(logits, masks).item()

                preds = torch.argmax(logits, dim=1)

                total_correct += (preds == masks).sum()
                total_pixels += torch.numel(masks)

                for cls in range(n_classes):
                    pred_inds = preds == cls
                    target_inds = masks == cls
                    intersection[cls] += (pred_inds & target_inds).sum()
                    union[cls] += (pred_inds | target_inds).sum()

        val_loss /= len(val_loader)
        pixel_acc = (total_correct / total_pixels).item()
        iou = (intersection / (union + 1e-6)).mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(iou)
        history["val_acc"].append(pixel_acc)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Pixel Acc: {pixel_acc:.4f}")
        print(f"Val mIoU: {iou:.4f}")
        print("-"*50)

    # ============================================================
    # SAVE GRAPHS
    # ============================================================

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1,2,2)
    plt.plot(history["val_iou"], label="Val mIoU")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.legend()
    plt.title("Metrics")

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.close()

    print("\nTraining complete.")
    print("Graphs saved as training_curves.png")


if __name__ == "__main__":
    main()
