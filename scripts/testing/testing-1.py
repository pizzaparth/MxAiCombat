# ===============================
# IMPROVED EVALUATION SCRIPT
# ===============================

import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from train import SegHead  # import model class

IMG_W=448
IMG_H=252
N_CLASSES=10
BACKBONE_NAME="dinov2_vits14"

def compute_iou(pred,target):
    pred=torch.argmax(pred,dim=1)
    intersection=(pred==target).sum().float()
    return (intersection/torch.numel(target)).item()

def main():

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone=torch.hub.load("facebookresearch/dinov2",BACKBONE_NAME)
    backbone.to(device).eval()

    model=SegHead(384).to(device)
    model.load_state_dict(torch.load("segmentation_head.pth",map_location=device))
    model.eval()

    ious=[]

    # 🔥 Flip TTA
    with torch.no_grad():
        for imgs,labels,_ in tqdm(loader):

            imgs,labels=imgs.to(device),labels.to(device)

            feats=backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits=model(feats,IMG_H//14,IMG_W//14)

            # horizontal flip
            imgs_flip=torch.flip(imgs,[3])
            feats_flip=backbone.forward_features(imgs_flip)["x_norm_patchtokens"]
            logits_flip=model(feats_flip,IMG_H//14,IMG_W//14)
            logits_flip=torch.flip(logits_flip,[3])

            logits=(logits+logits_flip)/2

            logits=F.interpolate(logits,size=imgs.shape[2:],mode="bilinear")

            iou=compute_iou(logits,labels)
            ious.append(iou)

    mean_iou=np.mean(ious)
    print("Mean IoU:",mean_iou)

    plt.hist(ious,bins=20)
    plt.savefig("evaluation_distribution.png")
    plt.close()

if __name__=="__main__":
    main()
