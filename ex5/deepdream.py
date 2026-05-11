# -*- coding: utf-8 -*-
"""
Kadai5: Deep Dream with GoogLeNet
Generate DeepDream images for all major layers
"""

from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision.models import GoogLeNet_Weights
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_PATH = "./cat.jpg"
OUT_DIR = "results"

OCTAVE_SCALE = 1.5
N_OCTAVES = 5
LR = 0.09
N_ITER = 40

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Layer name -> feature description
LAYER_GUIDE = {
    "conv1":       "edge",
    "conv2":       "color",
    "inception3a": "texture",
    "inception3b": "pattern",
    "inception4a": "parts",
    "inception4b": "shape",
    "inception4c": "animal_face",
    "inception4d": "complex",
    "inception4e": "object",
    "inception5a": "concept",
    "inception5b": "high_concept",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
model.eval().to(device)
for p in model.parameters():
    p.requires_grad_(False)
layer_map = dict(model.named_modules())


class Hook:
    """Stores the output activation of a given layer via forward hook."""
    def __init__(self, layer):
        self.activation = None
        self.hook = layer.register_forward_hook(self.fn)
    def fn(self, module, input, output):
        self.activation = output
    def close(self):
        self.hook.remove()


def to_tensor(img):
    """Convert HWC uint8 numpy image to normalized NCHW tensor."""
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.tensor(x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)


def to_numpy(t):
    """Convert normalized NCHW tensor back to HWC uint8 numpy image."""
    x = t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def resize_img(img, scale):
    """Resize image by scale factor."""
    h, w = img.shape[:2]
    return cv2.resize(img, (max(32, int(w * scale)), max(32, int(h * scale))),
                      interpolation=cv2.INTER_LINEAR)


def dream_step(img, hook):
    """Run N_ITER steps of gradient ascent on the image for one octave."""
    t = to_tensor(img)
    t.requires_grad_(True)
    for _ in range(N_ITER):
        model(t)
        loss = hook.activation.norm()
        loss.backward()
        with torch.no_grad():
            grad = t.grad
            t += LR * grad / (grad.std() + 1e-8)
            t.grad.zero_()
    return to_numpy(t)


def deep_dream(img, layer):
    """Apply multi-scale DeepDream: process from small to large octave and accumulate detail."""
    hook = Hook(layer)
    detail = np.zeros_like(img, dtype=np.float32)
    scales = [OCTAVE_SCALE ** (i - N_OCTAVES + 1) for i in range(N_OCTAVES)]
    for scale in scales:
        img_s    = resize_img(img, scale)
        detail_s = cv2.resize(detail, (img_s.shape[1], img_s.shape[0]))
        inp      = np.clip(img_s.astype(np.float32) + detail_s, 0, 255)
        dreamed  = dream_step(inp, hook)
        detail   = dreamed.astype(np.float32) - img_s.astype(np.float32)
    hook.close()
    return np.clip(img.astype(np.float32) + detail, 0, 255).astype(np.uint8)


def save_result(original, dream, layer_name, save_path):
    """Save a side-by-side comparison of the input and dream image."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#111111")
    axes[0].imshow(original)
    axes[0].set_title("Input", color="#e0ddd8")
    axes[0].axis("off")
    axes[1].imshow(dream)
    axes[1].set_title(layer_name, color="#e0ddd8")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def save_summary(original, dream_results, save_path):
    """Save all layer results as a single summary image.

    Layout: col 0 = Input, col 1.. = Dream per layer
            row 0 = image, row 1 = label text
    """
    n_cols = len(dream_results) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 5),
                             gridspec_kw={"height_ratios": [5, 0.5]})
    fig.patch.set_facecolor("#111111")
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Input", color="#e8e5de", fontsize=9, pad=4)
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    for col, (layer_name, label, dream_img) in enumerate(dream_results, start=1):
        axes[0, col].imshow(dream_img)
        axes[0, col].set_title(layer_name, color="#e8e5de", fontsize=8, pad=4)
        axes[0, col].axis("off")
        axes[1, col].text(0.5, 0.5, label, ha="center", va="center",
                          fontsize=7, color="#aaaaaa",
                          transform=axes[1, col].transAxes)
        axes[1, col].axis("off")
    fig.suptitle("Deep Dream  |  GoogLeNet  |  All Layers",
                 color="#e8e5de", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("summary saved:", save_path)


img_path = Path(IMAGE_PATH)
if not img_path.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
img = np.array(Image.open(img_path).convert("RGB"))
print("image:", IMAGE_PATH)

out_dir = Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

stem = img_path.stem
dream_results = []  # accumulate (layer_name, label, dream_img) for summary

for idx, layer_name in enumerate(LAYER_GUIDE):
    print(f"\n[{idx+1}/{len(LAYER_GUIDE)}] {layer_name}")
    dream = deep_dream(img, layer_map[layer_name])
    save_result(img, dream, layer_name, out_dir / f"{stem}_dream_{layer_name}.png")
    print("saved:", f"{stem}_dream_{layer_name}.png")
    dream_results.append((layer_name, LAYER_GUIDE[layer_name], dream))

save_summary(img, dream_results, out_dir / f"{stem}_dream_summary.png")
print("\nDone")