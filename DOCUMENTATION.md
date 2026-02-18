# MxAiCombat — Technical Documentation

> **AI-powered pilot assistance module for Unmanned Combat Ground Vehicles (UCGV)**
> Performs real-time terrain segmentation for obstacle and threat detection, regardless of visibility or weather conditions.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Hardware Requirements](#3-hardware-requirements)
4. [Installation & Environment Setup](#4-installation--environment-setup)
5. [Dataset Structure](#5-dataset-structure)
6. [Segmentation Classes](#6-segmentation-classes)
7. [Model Architecture](#7-model-architecture)
8. [Training Pipeline](#8-training-pipeline)
9. [Testing & Evaluation Pipeline](#9-testing--evaluation-pipeline)
10. [Outputs & Artifacts](#10-outputs--artifacts)
11. [Autonomous System Integration](#11-autonomous-system-integration)
12. [Key Hyperparameters Reference](#12-key-hyperparameters-reference)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Project Overview

MxAiCombat is a semantic segmentation system designed for off-road combat environments. It uses a fine-tuned **DINOv2-base (ViT-B/14)** backbone paired with a custom **Feature Pyramid Network (FPN) decoder** to classify every pixel in a camera frame into one of 10 terrain categories. The resulting segmentation maps are consumed by a downstream autonomous driving system built on an ESP32 microcontroller and OpenCV.

**Core objective:** Achieve a mean Intersection-over-Union (mIoU) > 0.5 on the off-road segmentation dataset while fitting within 4 GB VRAM.

---

## 2. Repository Structure

```
REPOSITORY/
├── ENV_SETUP/
│   └── setup_env.bat           # Automated Anaconda environment installer
├── scripts/
│   ├── training/
│   │   ├── train-21.py         # Main training script (current version)
│   │   ├── segmentation_head_best.pth   # Best checkpoint (saved during training)
│   │   ├── segmentation_head.pth        # Final checkpoint
│   │   └── train_stats/
│   │       └── training_curves.png      # Loss / IoU / Accuracy plots
│   └── testing/
│       ├── test-21.py          # Evaluation script (current version)
│       └── predictions/
│           ├── masks/                   # Raw predicted mask PNGs
│           ├── masks_color/             # Colour-coded predicted masks
│           ├── comparisons/             # Side-by-side ground-truth vs prediction images
│           ├── metrics_summary.txt      # Per-class IoU report
│           ├── per_class_metrics.png    # Bar chart of per-class IoU
│           └── iou_histogram.png        # Distribution of per-image IoU
├── OpenCV/                     # Live on-screen video output module
└── Self-driving vehicle/
    ├── ESP32/
    ├── OpenCV/
    ├── Arduino UNO/
    └── Main Board/
```

---

## 3. Hardware Requirements

| Component | Specification |
|---|---|
| CPU | Intel Core 5 210H (or equivalent) |
| RAM | 16 GB |
| GPU VRAM | 4 GB (NVIDIA, CUDA-compatible) |
| Microcontroller | ESP32 |
| Motor controller | Arduino UNO |

The training scripts are designed to fit comfortably within 4 GB VRAM using Automatic Mixed Precision (AMP) and gradient accumulation.

---

## 4. Installation & Environment Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/pizzaparth/MxAiCombat.git
cd MxAiCombat
```

### Step 2 — Install Anaconda

Download and install [Anaconda](https://www.anaconda.com/download) for your operating system.

### Step 3 — Create the Conda environment

Open an Anaconda Prompt and run:

```bash
cd ENV_SETUP
setup_env.bat
```

This script installs all required Python dependencies into a Conda environment named `EDU`.

### Step 4 — Activate the environment

```bash
conda activate EDU
```

---

## 5. Dataset Structure

The training and validation datasets must be arranged as follows:

```
Offroad_Segmentation_Training_Dataset/
├── train/
│   ├── Color_Images/      # RGB input images (.png / .jpg)
│   └── Segmentation/      # Ground-truth mask images (grayscale, same filenames)
└── val/
    ├── Color_Images/
    └── Segmentation/

Offroad_Segmentation_testImages/
├── Color_Images/
└── Segmentation/
```

The dataset directories are expected to live alongside the training/testing scripts inside `scripts/training/` and `scripts/testing/` respectively (the scripts resolve paths relative to their own location).

**Mask encoding:** Ground-truth masks use integer pixel values (not RGB colour) to encode class labels. The raw values are mapped to class indices during loading (see §6).

---

## 6. Segmentation Classes

The model classifies every pixel into one of 10 terrain categories. Rare classes (marked ⚠️) receive boosted sampling during training and have higher class weights in the loss function.

| Class Index | Class Name | Raw Mask Value | Colour (Visualization) |
|---|---|---|---|
| 0 | Background | 0 | Black |
| 1 | Trees | 100 | Dark Green |
| 2 | Lush Bushes | 200 | Bright Green |
| 3 | Dry Grass | 300 | Tan |
| 4 | Dry Bushes | 500 | Brown |
| 5 ⚠️ | Ground Clutter | 550 | Olive |
| 6 ⚠️ | Logs | 700 | Saddle Brown |
| 7 ⚠️ | Rocks | 800 | Grey |
| 8 | Landscape | 7100 | Sienna |
| 9 | Sky | 10000 | Sky Blue |

---

## 7. Model Architecture

### 7.1 Backbone — DINOv2-Base (ViT-B/14)

The backbone is Meta's **DINOv2 ViT-Base/14** (`dinov2_vitb14`), a Vision Transformer that produces **768-dimensional patch embeddings** with a patch size of 14×14 pixels. It is loaded from `facebookresearch/dinov2` via `torch.hub`.

**Partial fine-tuning strategy:** The backbone is initially fully frozen, then the last 4 transformer blocks (out of 12) and the final normalization layer are unfrozen. This adapts the backbone to the off-road domain without overwriting general visual features learned on ImageNet.

### 7.2 Multi-Scale Feature Extractor — `DINOv2MultiScale`

Forward hooks are registered at four intermediate transformer blocks — at approximately 1/4, 1/2, 3/4, and the final depth of the backbone (blocks 2, 5, 8, and 11 for ViT-B/12). The CLS token is discarded; only the patch tokens `(B, N, 768)` are retained. This gives four feature maps at different abstraction levels, enabling the decoder to combine both fine-grained spatial detail and high-level semantic context.

### 7.3 Decoder — FPN Segmentation Head (`FPNSegHead`)

The decoder follows a Feature Pyramid Network design:

1. **Lateral projections** — Each of the four scale feature maps is independently projected from 768 → 256 channels via a `Linear + LayerNorm` layer, then reshaped into a spatial grid `(B, 256, tokenH, tokenW)`.
2. **Top-down fusion** — Features are aggregated from finest (deepest) to coarsest (shallowest) using element-wise addition, with a **ConvNeXt-style refinement block** at each scale (depthwise 7×7 conv → BN → GELU → 1×1 expansion → 1×1 projection).
3. **Classifier** — A small 3×3 conv + BN + GELU + 1×1 conv head maps the fused feature map to `n_classes` logits, which are then bilinearly upsampled to the full input resolution.

### 7.4 Loss Function — `FocalDiceLoss`

Training uses a composite loss that addresses class imbalance at both the pixel and region levels:

- **Focal Loss** (γ = 2.0, with inverse-square-root class weights) — down-weights easy background pixels so the model focuses on hard, rare classes.
- **Dice Loss** (weight = 0.5) — directly optimizes the overlap between predicted and ground-truth class regions, acting as a proxy for IoU.

Total loss = `FocalLoss + 0.5 × DiceLoss`

---

## 8. Training Pipeline

### Running Training

```bash
conda activate EDU
cd scripts/training
python train-21.py
```

### What Happens During Training

**Data loading and augmentation:** Images are resized to 476×266 pixels (the nearest DINOv2-compatible resolution to 480×270). During training, the following augmentations are applied: random horizontal flip (50%), random scale crop (75–100%), random rotation ±10° (30% probability), Gaussian blur (20%), color jitter — brightness, contrast, saturation, hue (60%), and random grayscale conversion (10%). Masks always use nearest-neighbor interpolation to preserve label integrity.

**Class-weighted sampling:** Before training begins, every image in the training set is scanned. Images containing rare classes (Ground Clutter, Logs, Rocks) are assigned a sampling weight of 3.0; all others receive 1.0. A `WeightedRandomSampler` uses these weights to oversample hard images each epoch.

**Optimizer:** AdamW with two parameter groups — backbone unfrozen blocks at LR 5e-5 and the FPN head at LR 5e-4 (both with weight decay 1e-4).

**Scheduler:** OneCycleLR with 10% warmup, cosine annealing, starting at 1/25th of peak LR and decaying to 1/10,000th of the start LR.

**Gradient accumulation:** The effective batch size is 8 (batch_size=2 × accum_steps=4), achieved by accumulating gradients over 4 micro-steps before each optimizer step. Gradient norm clipping of 1.0 is applied before each update.

**Automatic Mixed Precision (AMP):** `torch.cuda.amp.autocast` and `GradScaler` are used throughout to halve VRAM usage.

**Early stopping:** Training runs for up to 50 epochs. If validation mIoU does not improve for 12 consecutive epochs, training stops early.

**Checkpointing:** The best model by validation mIoU is saved to `segmentation_head_best.pth`. A final checkpoint is saved to `segmentation_head.pth` at the end of training. Both files contain a dictionary with keys `head` (FPN head state dict), `backbone` (DINOv2 state dict), and `val_iou`.

---

## 9. Testing & Evaluation Pipeline

### Running Evaluation

```bash
conda activate EDU
cd scripts/testing
python test-21.py
```

### Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `segmentation_head_best.pth` | Path to checkpoint file |
| `--data_dir` | `Offroad_Segmentation_testImages` | Path to test dataset |
| `--output_dir` | `./predictions` | Directory for all outputs |
| `--batch_size` | 2 | Inference batch size |
| `--num_samples` | 5 | Number of comparison images to save |
| `--fpn_ch` | 256 | FPN channel count (must match training) |
| `--no_tta` | False | Pass this flag to disable Test-Time Augmentation |

### Test-Time Augmentation (TTA)

By default, TTA is enabled. Each image is evaluated twice — once normally, and once after a horizontal flip (with the resulting prediction flipped back). The two softmax probability maps are averaged before taking the argmax. This typically yields a free +0.02 to +0.04 mIoU improvement with no additional training.

### Metrics Computed

- **Mean IoU (mIoU):** Averaged over all classes present in the test set.
- **Pixel Accuracy:** Fraction of pixels correctly classified.
- **Per-class IoU:** Individual IoU for each of the 10 classes.
- **Per-image IoU:** IoU computed independently for every test image (used to identify hard cases).

---

## 10. Outputs & Artifacts

### Training Outputs (`scripts/training/train_stats/`)

`training_curves.png` — Three-panel plot showing train/val Loss, mIoU, and Pixel Accuracy over epochs. The mIoU panel includes a red dashed line at the 0.5 target threshold.

### Testing Outputs (`predictions/`)

`masks/` — Raw predicted segmentation masks saved as single-channel PNGs, with pixel values corresponding to class indices (0–9).

`masks_color/` — Colour-coded versions of the predicted masks, saved as BGR PNGs via OpenCV, using the colour palette defined in §6.

`comparisons/` — Side-by-side figures showing the original image, ground-truth mask, and predicted mask for the first N test images (default N=5).

`metrics_summary.txt` — Plain-text report listing overall mIoU, pixel accuracy, and per-class IoU values.

`per_class_metrics.png` — Bar chart of per-class IoU with class-coloured bars, mean IoU line (red dashed), and 0.5 target line (orange dotted).

`iou_histogram.png` — Histogram of per-image IoU values, useful for identifying systematically hard images in the test set.

---

## 11. Autonomous System Integration

The segmentation model interfaces with the physical vehicle through two subsystems:

**ESP32 + Arduino UNO (motor control):** The ESP32 receives segmentation-derived navigation decisions and communicates with the Arduino UNO to control drive motors and steering. Install the [Arduino IDE](https://www.arduino.cc/en/software) to flash firmware to the UNO.

**OpenCV (live video output):** The `OpenCV/` directory contains a script for displaying the live segmentation overlay on screen. Run the script from that directory:

```bash
cd OpenCV
python <video_output_script.py>
```

This streams the camera feed with the segmentation mask composited on top in real time.

---

## 12. Key Hyperparameters Reference

| Parameter | Value | Notes |
|---|---|---|
| Input resolution | 476 × 266 px | Nearest ÷14 multiple to 480×270 |
| Backbone | `dinov2_vitb14` | 768-dim embeddings, 12 blocks |
| Unfrozen backbone blocks | 4 (last) | + final norm layer |
| Batch size | 2 | Per GPU step |
| Gradient accumulation steps | 4 | Effective batch = 8 |
| Head learning rate | 5e-4 | AdamW |
| Backbone learning rate | 5e-5 | 10× smaller than head |
| Weight decay | 1e-4 | Both parameter groups |
| Focal loss γ | 2.0 | Standard value |
| Dice loss weight | 0.5 | Relative to focal loss |
| FPN channels | 256 | Must match between train and test |
| Max epochs | 50 | With early stopping |
| Early stopping patience | 12 | Epochs without val mIoU improvement |
| Rare class sampling multiplier | 3.0× | Ground Clutter, Logs, Rocks |
| mIoU target | > 0.5 | |

---

## 13. Troubleshooting

**CUDA out of memory:** Reduce `batch_size` to 1 or increase `accum_steps` proportionally to maintain the effective batch size.

**`RuntimeError: This checkpoint appears to be from v18/v19`:** The checkpoint was trained with an older architecture. Retrain from scratch using `train-21.py` and use the resulting `.pth` files with `test-21.py`.

**`model_path` not found:** The test script will automatically fall back to `segmentation_head.pth` in the same directory if `segmentation_head_best.pth` is missing.

**Low IoU on rare classes (Ground Clutter, Logs, Rocks):** Verify that the dataset contains sufficient examples of these classes in the training split. Increase the `RARE_CLASSES` sampling multiplier (currently 3.0) in `train-21.py` or add more augmentation.

**DINOv2 download fails:** The backbone is fetched from `facebookresearch/dinov2` via `torch.hub` on first run and cached locally. Ensure internet access is available the first time the script is run.

**TTA is slower than expected:** TTA doubles inference time by design. Pass `--no_tta` to `test-21.py` to run single-pass inference at the cost of slightly lower accuracy.