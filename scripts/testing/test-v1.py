import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import cv2

plt.switch_backend("Agg")

# ===============================
# CONFIG (MUST MATCH TRAINING)
# ===============================

IMG_W = 448
IMG_H = 252
BACKBONE_NAME = "dinov2_vits14"  # SMALL backbone (384 dim)

# ===============================
# CLASS MAPPING
# ===============================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

class_names = [
    'Background','Trees','Lush Bushes','Dry Grass','Dry Bushes',
    'Ground Clutter','Logs','Rocks','Landscape','Sky'
]

n_classes = len(value_map)

color_palette = np.array([
    [0,0,0],[34,139,34],[0,255,0],[210,180,140],[139,90,43],
    [128,128,0],[139,69,19],[128,128,128],[160,82,45],[135,206,235]
], dtype=np.uint8)

# ===============================
# UTILITIES
# ===============================

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        new_arr[arr == raw] = new
    return Image.fromarray(new_arr)

def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(n_classes):
        color_mask[mask == cid] = color_palette[cid]
    return color_mask

# ===============================
# DATASET
# ===============================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform, mask_transform):
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.mask_dir = os.path.join(data_dir, "Segmentation")
        self.ids = os.listdir(self.image_dir)
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        img = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))
        mask = convert_mask(mask)

        img = self.transform(img)
        mask = self.mask_transform(mask)
        mask = (mask * 255).long()

        return img, mask.squeeze(0), name

# ===============================
# MODEL
# ===============================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, 7, padding=3),
            nn.GELU()
        )

        self.block = nn.Sequential(
            nn.Conv2d(128, 128, 7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, 1),
            nn.GELU()
        )

        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)

# ===============================
# METRICS
# ===============================

def compute_iou(pred, target):
    pred = torch.argmax(pred, dim=1)
    pred = pred.view(-1)
    target = target.view(-1)

    ious = []
    for cls in range(n_classes):
        p = pred == cls
        t = target == cls

        intersection = (p & t).sum().float()
        union = (p | t).sum().float()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).cpu().item())

    return np.nanmean(ious), ious

# ===============================
# MAIN
# ===============================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='segmentation_head.pth')
    parser.add_argument('--data_dir', default='Offroad_Segmentation_testImages')
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    dataset = MaskDataset(args.data_dir, transform, mask_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("Loading backbone:", BACKBONE_NAME)
    backbone = torch.hub.load("facebookresearch/dinov2", BACKBONE_NAME)
    backbone.eval().to(device)

    # Determine embedding dimension
    sample, _, _ = dataset[0]
    with torch.no_grad():
        out = backbone.forward_features(sample.unsqueeze(0).to(device))["x_norm_patchtokens"]
    emb_dim = out.shape[2]
    print("Embedding dimension:", emb_dim)

    classifier = SegmentationHeadConvNeXt(
        emb_dim, n_classes, IMG_W // 14, IMG_H // 14
    )

    classifier.load_state_dict(torch.load(args.model_path, map_location=device))
    classifier.eval().to(device)
    print("Model loaded successfully!")

    os.makedirs("predictions/masks", exist_ok=True)
    os.makedirs("predictions/masks_color", exist_ok=True)
    os.makedirs("predictions/comparisons", exist_ok=True)

    all_ious = []
    per_class_accum = np.zeros(n_classes)

    print("\nRunning evaluation...\n")

    with torch.no_grad():
        for imgs, labels, names in tqdm(loader):

            imgs = imgs.to(device)
            labels = labels.to(device)

            feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(feats)
            logits = F.interpolate(
                logits, size=imgs.shape[2:], mode="bilinear", align_corners=False
            )

            mean_iou, per_class = compute_iou(logits, labels)

            all_ious.append(mean_iou)
            per_class_accum += np.nan_to_num(per_class)

            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for i in range(len(names)):
                raw_mask = preds[i]
                color_mask = mask_to_color(raw_mask)

                cv2.imwrite(f"predictions/masks/{names[i]}", raw_mask)
                cv2.imwrite(f"predictions/masks_color/{names[i]}", color_mask)

                # Save comparison image
                original = cv2.imread(
                    os.path.join(args.data_dir, "Color_Images", names[i])
                )
                original = cv2.resize(original, (IMG_W, IMG_H))
                comparison = np.hstack([original, color_mask])
                cv2.imwrite(f"predictions/comparisons/{names[i]}", comparison)

    mean_iou = np.nanmean(all_ious)
    per_class_avg = per_class_accum / len(loader)

    print("\n===================================")
    print("Mean IoU:", mean_iou)
    print("===================================")

    # Save metrics text
    with open("predictions/evaluation_metrics.txt", "w") as f:
        f.write(f"Mean IoU: {mean_iou}\n\n")
        for i, cname in enumerate(class_names):
            f.write(f"{cname}: {per_class_avg[i]}\n")

    # Plot per-class IoU
    plt.figure(figsize=(10,5))
    plt.bar(class_names, per_class_avg)
    plt.xticks(rotation=45)
    plt.title("Per-Class IoU")
    plt.tight_layout()
    plt.savefig("predictions/per_class_metrics.png")
    plt.close()

    print("Results saved in ./predictions/")

if __name__ == "__main__":
    main()
