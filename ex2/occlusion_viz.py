# -*- coding: utf-8 -*-
"""
Kadai 2: CNN Visualization 1 - Zeiler-Fergus Occlusion Method
Occlusion sensitivity map using pretrained VGG16 (ImageNet 1000 classes)

Reference:
  M. Zeiler and R. Fergus, Visualizing and understanding convolutional
  networks, ECCV, 2014.
  https://arxiv.org/abs/1311.2901

  Stanford CS231n - Understanding CNNs (Occluding parts of the image):
  https://cs231n.github.io/understanding-cnn/

Usage:
  python occlusion_viz.py
"""

from __future__ import print_function
import os
import urllib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# ─────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────
INPUT_SIZE   = 256   # input image size
OCCLUDER_SIZE = 64   # occluder (gray square) size
STRIDE       = 8    # slide step in pixels
OCCLUDER_VAL = 0.5  # gray value (0-1)
OUTPUT_DIR   = "./occlusion_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# occluder moves from -56 to 247 (as specified in the assignment)
OCC_START = -56
OCC_END   = 247

# ─────────────────────────────────────────────────────────
# ImageNet class labels
# ─────────────────────────────────────────────────────────
def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        response = urllib.request.urlopen(url, timeout=10)
        labels = json.loads(response.read().decode())
        print("[Labels] Loaded", len(labels), "ImageNet labels")
        return labels
    except Exception as e:
        print("[Labels] Download failed:", e)
        # Fallback: return numbered labels
        return ["class_%d" % i for i in range(1000)]

# ─────────────────────────────────────────────────────────
# Sample images: 5 ImageNet classes
# (URL, true_label_index, description)
# ─────────────────────────────────────────────────────────
SAMPLE_IMAGES = [
    {
        # Dog: from PyTorch official tutorial sample
        "url": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        "label_idx": 258,   # Samoyed
        "name": "dog",
        "desc": "Samoyed dog (class 258)"
    },
    {
        # Cat: from PyTorch hub samples
        "url": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02123045_tabby.JPEG",
        "label_idx": 281,   # tabby cat
        "name": "cat",
        "desc": "Tabby Cat (class 281)"
    },
    {
        # Elephant: from imagenet sample images repo
        "url": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02504458_African_elephant.JPEG",
        "label_idx": 386,   # African elephant
        "name": "elephant",
        "desc": "African Elephant (class 386)"
    },
    {
        # Goldfish
        "url": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01443537_goldfish.JPEG",
        "label_idx": 1,     # goldfish
        "name": "goldfish",
        "desc": "Goldfish (class 1)"
    },
    {
        # Peacock
        "url": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01806143_peacock.JPEG",
        "label_idx": 84,    # peacock
        "name": "peacock",
        "desc": "Peacock (class 84)"
    },
]

def download_image(url, save_path):
    """Download image from URL with browser User-Agent to avoid 403."""
    if os.path.exists(save_path):
        print("[Image] Already exists:", save_path)
        return True
    try:
        print("[Image] Downloading:", url)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/120.0.0.0 Safari/537.36"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            with open(save_path, "wb") as f:
                f.write(resp.read())
        return True
    except Exception as e:
        print("[Image] Failed to download:", e)
        return False

# ─────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])

def load_image(path):
    """Load image and return both PIL (for display) and tensor."""
    img_pil = Image.open(path).convert("RGB")
    img_pil = img_pil.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    img_tensor = preprocess(img_pil)   # (3, H, W)
    return img_pil, img_tensor

# ─────────────────────────────────────────────────────────
# Occlusion sensitivity map
# ─────────────────────────────────────────────────────────
def compute_occlusion_map(model, img_tensor, true_label_idx, device):
    """
    Slide a gray occluder over the image and record the true-class
    probability at each position.

    occluder top-left moves: OCC_START to OCC_END, step=STRIDE
    Out-of-bounds regions are ignored (clipped).

    Returns:
        heatmap (np.ndarray): 2D array of true-class probabilities
        positions (list of (y,x)): occluder top-left positions
    """
    model.eval()
    softmax = nn.Softmax(dim=1)

    positions = list(range(OCC_START, OCC_END, STRIDE))
    n = len(positions)
    heatmap = np.zeros((n, n), dtype=np.float32)

    img_np = img_tensor.numpy()   # (3, H, W)
    H, W = img_np.shape[1], img_np.shape[2]

    total = n * n
    count = 0

    with torch.no_grad():
        for i, y in enumerate(positions):
            for j, x in enumerate(positions):
                # Copy image and apply occluder
                occluded = img_np.copy()

                # Occluder region in image coordinates (clip to image)
                y1 = max(0, y)
                y2 = min(H, y + OCCLUDER_SIZE)
                x1 = max(0, x)
                x2 = min(W, x + OCCLUDER_SIZE)

                if y2 > y1 and x2 > x1:
                    # Fill with gray (normalized value)
                    # gray value in normalized space: (0.5 - mean) / std
                    gray_r = (OCCLUDER_VAL - 0.485) / 0.229
                    gray_g = (OCCLUDER_VAL - 0.456) / 0.224
                    gray_b = (OCCLUDER_VAL - 0.406) / 0.225
                    occluded[0, y1:y2, x1:x2] = gray_r
                    occluded[1, y1:y2, x1:x2] = gray_g
                    occluded[2, y1:y2, x1:x2] = gray_b

                inp = torch.from_numpy(occluded).unsqueeze(0).to(device)
                out = softmax(model(inp))
                prob = out[0, true_label_idx].item()
                heatmap[i, j] = prob

                count += 1
                if count % 100 == 0:
                    print("  Progress: %d / %d (%.1f%%)" % (
                        count, total, 100.0 * count / total))

    return heatmap, positions

# ─────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────
def visualize_occlusion(img_pil, heatmap, sample_info, labels, save_path):
    """Plot original image + heatmap side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Occlusion Sensitivity Map (Zeiler-Fergus)\n" +
                 sample_info["desc"], fontsize=12)

    # Original image
    axes[0].imshow(img_pil)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Heatmap
    # Red = high probability (object visible), Blue = low (object occluded)
    im = axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=heatmap.max(),
                        interpolation="bilinear")
    axes[1].set_title("True-class Probability\n(Red=high, Blue=low)")
    axes[1].set_xlabel("Occluder X position")
    axes[1].set_ylabel("Occluder Y position")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved]", save_path)

# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # Load VGG16 pretrained on ImageNet
    print("[Model] Loading VGG16 pretrained...")
    try:
        model = models.vgg16(pretrained=True)
    except TypeError:
        from torchvision.models import VGG16_Weights
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()
    print("[Model] VGG16 loaded")

    # Load ImageNet labels
    labels = load_imagenet_labels()

    # Process each sample image
    for sample in SAMPLE_IMAGES:
        print("\n" + "=" * 60)
        print("[Sample]", sample["desc"])

        # Download image
        img_path = os.path.join(OUTPUT_DIR, sample["name"] + ".jpg")
        ok = download_image(sample["url"], img_path)
        if not ok:
            print("[Skip] Could not download image")
            continue

        # Load & preprocess
        img_pil, img_tensor = load_image(img_path)

        # First: check model prediction on original image
        with torch.no_grad():
            inp = img_tensor.unsqueeze(0).to(device)
            out = torch.nn.functional.softmax(model(inp), dim=1)
            top5 = out[0].topk(5)
        print("[Prediction] Top-5:")
        for prob, idx in zip(top5.values.cpu().numpy(),
                             top5.indices.cpu().numpy()):
            print("  %5.2f%%  %s (class %d)" % (
                prob * 100, labels[idx] if idx < len(labels) else "?", idx))

        true_idx = sample["label_idx"]
        true_prob = out[0, true_idx].item()
        print("[True label] %s (class %d): %.2f%%" % (
            labels[true_idx] if true_idx < len(labels) else "?",
            true_idx, true_prob * 100))

        # Compute occlusion map
        print("[Occlusion] Computing heatmap...")
        heatmap, positions = compute_occlusion_map(
            model, img_tensor, true_idx, device)

        print("[Occlusion] Heatmap shape:", heatmap.shape)
        print("[Occlusion] Prob range: %.4f - %.4f" % (
            heatmap.min(), heatmap.max()))

        # Save heatmap as numpy
        np.save(os.path.join(OUTPUT_DIR, sample["name"] + "_heatmap.npy"),
                heatmap)

        # Visualize
        save_path = os.path.join(OUTPUT_DIR,
                                 sample["name"] + "_occlusion.png")
        visualize_occlusion(img_pil, heatmap, sample, labels, save_path)

    print("\n[Done] All results saved to:", OUTPUT_DIR)
    print("[Files]", os.listdir(OUTPUT_DIR))


if __name__ == "__main__":
    main()