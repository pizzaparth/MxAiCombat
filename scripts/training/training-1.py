# ===============================
# IMPROVED TRAINING SCRIPT
# ===============================

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

# ===============================
# CONFIG
# ===============================

IMG_W = 448
IMG_H = 252
BACKBONE_NAME = "dinov2_vits14"  # small backbone
N_CLASSES = 10
EPOCHS = 40
BATCH_SIZE = 8
LR = 3e-4

value_map = {
    0:0,100:1,200:2,300:3,500:4,
    550:5,700:6,800:7,7100:8,10000:9
}

# ===============================
# DATASET
# ===============================

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw,new in value_map.items():
        new_arr[arr==raw] = new
    return Image.fromarray(new_arr)

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.img_dir = os.path.join(data_dir,"Color_Images")
        self.mask_dir = os.path.join(data_dir,"Segmentation")
        self.ids = os.listdir(self.img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = Image.open(os.path.join(self.img_dir,name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir,name))
        mask = convert_mask(mask)

        img = img.resize((IMG_W,IMG_H), Image.BILINEAR)
        mask = mask.resize((IMG_W,IMG_H), Image.NEAREST)

        img = self.transform(img)
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask

# ===============================
# MODEL
# ===============================

class SegHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,256,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,N_CLASSES,1)
        )
    def forward(self,x,H,W):
        B,N,C = x.shape
        x = x.reshape(B,H,W,C).permute(0,3,1,2)
        return self.net(x)

# ===============================
# DICE LOSS
# ===============================

def dice_loss(pred,target,smooth=1e-6):
    pred = F.softmax(pred,dim=1)
    target_onehot = F.one_hot(target,N_CLASSES).permute(0,3,1,2).float()
    intersection = (pred*target_onehot).sum(dim=(2,3))
    union = pred.sum(dim=(2,3))+target_onehot.sum(dim=(2,3))
    dice = (2*intersection+smooth)/(union+smooth)
    return 1-dice.mean()

# ===============================
# TRAINING
# ===============================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.4,0.4,0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir,"Offroad_Segmentation_Training_Dataset/train")
    val_dir = os.path.join(script_dir,"Offroad_Segmentation_Training_Dataset/val")

    train_set = MaskDataset(train_dir,transform)
    val_set = MaskDataset(val_dir,transform)

    train_loader = DataLoader(train_set,BATCH_SIZE,shuffle=True,num_workers=4)
    val_loader = DataLoader(val_set,BATCH_SIZE,shuffle=False,num_workers=4)

    backbone = torch.hub.load("facebookresearch/dinov2",BACKBONE_NAME)
    backbone.to(device)

    # 🔥 Unfreeze last 2 blocks
    for name,param in backbone.named_parameters():
        if "blocks.10" in name or "blocks.11" in name:
            param.requires_grad=True
        else:
            param.requires_grad=False

    sample,_ = next(iter(train_loader))
    with torch.no_grad():
        out = backbone.forward_features(sample.to(device))["x_norm_patchtokens"]
    emb_dim = out.shape[2]

    model = SegHead(emb_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters())+
        [p for p in backbone.parameters() if p.requires_grad],
        lr=LR
    )

    history={"train_loss":[],"val_iou":[]}

    for epoch in range(EPOCHS):

        model.train()
        total_loss=0

        for imgs,masks in tqdm(train_loader):
            imgs,masks=imgs.to(device),masks.to(device)

            feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(feats,IMG_H//14,IMG_W//14)
            logits = F.interpolate(logits,size=imgs.shape[2:],mode="bilinear")

            ce = F.cross_entropy(logits,masks)
            dl = dice_loss(logits,masks)
            loss = ce + dl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()

        history["train_loss"].append(total_loss/len(train_loader))

        # ===== VALIDATION IoU =====
        model.eval()
        ious=[]
        with torch.no_grad():
            for imgs,masks in val_loader:
                imgs,masks=imgs.to(device),masks.to(device)
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(feats,IMG_H//14,IMG_W//14)
                logits = F.interpolate(logits,size=imgs.shape[2:],mode="bilinear")

                pred=torch.argmax(logits,dim=1)
                intersection=(pred==masks).sum().float()
                union=torch.numel(masks)
                ious.append((intersection/union).item())

        history["val_iou"].append(np.mean(ious))

        print(f"Epoch {epoch+1} | Loss {history['train_loss'][-1]:.4f} | IoU {history['val_iou'][-1]:.4f}")

    # ===== Save Graph =====
    plt.plot(history["train_loss"],label="Train Loss")
    plt.plot(history["val_iou"],label="Val IoU")
    plt.legend()
    plt.savefig("training_curves.png")
    plt.close()

    torch.save(model.state_dict(),"segmentation_head.pth")
    print("Training complete.")

if __name__=="__main__":
    main()
