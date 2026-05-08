# -*- coding: utf-8 -*-
"""
Kadai 3: CNN Visualization 2 - Back Propagation based methods
  1. Vanilla Saliency Map (Simonyan et al. 2014)
  2. SmoothGrad (Smilkov et al. 2017)
  3. Guided Backpropagation (Springenberg et al. 2015)
"""

from __future__ import print_function
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ─────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────
INPUT_SIZE  = 256
OUTPUT_DIR  = "./bp_results"
IMAGE_DIR   = "/home/yanai-lab/oyundari/kadai_3b/ex2/occlusion_results/"   # reuse images from kadai 2
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", device)

SAMPLE_IMAGES = [
    {"name": "dog",      "label_idx": 258, "desc": "Samoyed dog"},
    {"name": "cat",      "label_idx": 281, "desc": "Tabby Cat"},
    {"name": "elephant", "label_idx": 386, "desc": "African Elephant"},
    {"name": "goldfish", "label_idx": 1,   "desc": "Goldfish"},
    {"name": "peacock",  "label_idx": 84,  "desc": "Peacock"},
]

# SmoothGrad settings
SMOOTHGRAD_N     = 20    # number of noisy samples
SMOOTHGRAD_SIGMA = 0.15  # noise level (fraction of input range)

# ─────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def load_image(path):
    img_pil = Image.open(path).convert("RGB")
    img_pil = img_pil.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    tensor  = preprocess(img_pil)
    return img_pil, tensor

def tensor_to_display(t):
    """Convert normalized tensor (3,H,W) to displayable (H,W,3) uint8."""
    # Unnormalize
    t = t.clone()
    for c, (m, s) in enumerate(zip(MEAN, STD)):
        t[c] = t[c] * s + m
    t = t.clamp(0, 1)
    arr = t.permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────
# Guided BP hook
# ─────────────────────────────────────────────────────────
class GuidedReLU(nn.Module):
    """ReLU that passes only positive gradients during backward."""
    def forward(self, x):
        return torch.relu(x)

    def backward_hook(self, grad):
        # Only pass positive gradients (Guided BP rule)
        return torch.clamp(grad, min=0.0)

def register_guided_hooks(model):
    """Replace ReLU activations with GuidedReLU and register hooks."""
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hook = module.register_backward_hook(
                lambda m, grad_in, grad_out: (torch.clamp(grad_out[0], min=0.0),)
            )
            hooks.append(hook)
    return hooks

def remove_hooks(hooks):
    for h in hooks:
        h.remove()

# ─────────────────────────────────────────────────────────
# 1. Vanilla Saliency Map
# ─────────────────────────────────────────────────────────
def vanilla_saliency(model, img_tensor, label_idx):
    """
    Compute gradient of true-class score w.r.t. input image.
    Saliency = |dScore/dInput|
    """
    model.eval()
    inp = img_tensor.unsqueeze(0).to(device)
    inp.requires_grad_(True)

    output = model(inp)
    # Backprop only the true class score (not softmax, raw logit)
    score = output[0, label_idx]
    model.zero_grad()
    score.backward()

    # Gradient: (1, 3, H, W) -> (H, W)
    grad = inp.grad.data.squeeze()           # (3, H, W)
    saliency = grad.abs().max(dim=0)[0]      # max over channels
    return saliency.cpu().numpy()

# ─────────────────────────────────────────────────────────
# 2. SmoothGrad
# ─────────────────────────────────────────────────────────
def smooth_grad(model, img_tensor, label_idx, n=SMOOTHGRAD_N, sigma=SMOOTHGRAD_SIGMA):
    """
    Average gradients over N noisy versions of the input.
    Noise level: sigma * (max - min) of input tensor.
    """
    model.eval()
    noise_level = sigma * (img_tensor.max() - img_tensor.min()).item()
    sum_grad = torch.zeros_like(img_tensor)

    for i in range(n):
        noise = torch.randn_like(img_tensor) * noise_level
        noisy = (img_tensor + noise).unsqueeze(0).to(device)
        noisy.requires_grad_(True)

        output = model(noisy)
        score  = output[0, label_idx]
        model.zero_grad()
        score.backward()

        grad = noisy.grad.data.squeeze().cpu()
        sum_grad += grad.abs().max(dim=0, keepdim=True)[0].expand_as(grad)

        if (i + 1) % 5 == 0:
            print("  SmoothGrad: %d/%d" % (i + 1, n))

    avg_grad = sum_grad / n
    saliency = avg_grad.max(dim=0)[0]
    return saliency.numpy()

# ─────────────────────────────────────────────────────────
# 3. Guided Backpropagation
# ─────────────────────────────────────────────────────────
def guided_backprop(model, img_tensor, label_idx):
    """
    Register backward hooks on all ReLUs to pass only positive gradients.
    This produces sharper, cleaner saliency maps.
    """
    model.eval()

    # Register guided BP hooks
    hooks = register_guided_hooks(model)

    inp = img_tensor.unsqueeze(0).to(device)
    inp.requires_grad_(True)

    output = model(inp)
    score  = output[0, label_idx]
    model.zero_grad()
    score.backward()

    grad = inp.grad.data.squeeze()       # (3, H, W)
    saliency = grad.abs().max(dim=0)[0]  # (H, W)

    # Remove hooks to restore normal behavior
    remove_hooks(hooks)

    return saliency.cpu().numpy()

# ─────────────────────────────────────────────────────────
# Visualization helper
# ─────────────────────────────────────────────────────────
def normalize_map(m):
    """Normalize saliency map to [0, 1]."""
    m = m - m.min()
    if m.max() > 0:
        m = m / m.max()
    return m

def overlay_saliency(img_arr, saliency, alpha=0.5):
    """Overlay saliency map (hot colormap) on original image."""
    sal_norm = normalize_map(saliency)
    heatmap = plt.cm.hot(sal_norm)[:, :, :3]   # (H, W, 3) RGB
    heatmap = (heatmap * 255).astype(np.uint8)
    blended = (alpha * img_arr + (1 - alpha) * heatmap).astype(np.uint8)
    return blended

def save_comparison(img_pil, results, sample_name, desc):
    """
    Save 4-panel figure:
      Original | Vanilla Saliency | SmoothGrad | Guided BP
    """
    img_arr = np.array(img_pil)
    n_methods = len(results)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(5 * (n_methods + 1), 10))
    fig.suptitle("BP Visualization: " + desc, fontsize=13)

    method_names = list(results.keys())

    # Top row: saliency maps alone
    axes[0, 0].imshow(img_pil)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    for j, name in enumerate(method_names):
        sal = normalize_map(results[name])
        axes[0, j + 1].imshow(sal, cmap="hot")
        axes[0, j + 1].set_title(name)
        axes[0, j + 1].axis("off")

    # Bottom row: overlay on original
    axes[1, 0].imshow(img_pil)
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis("off")

    for j, name in enumerate(method_names):
        overlaid = overlay_saliency(img_arr, results[name])
        axes[1, j + 1].imshow(overlaid)
        axes[1, j + 1].set_title(name + " (overlay)")
        axes[1, j + 1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, sample_name + "_bp.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved]", out_path)

# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    # Load VGG16
    print("[Model] Loading VGG16 pretrained...")
    try:
        from torchvision.models import VGG16_Weights
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    except (ImportError, AttributeError):
        model = models.vgg16(pretrained=True)
    model = model.to(device)
    model.eval()
    print("[Model] VGG16 loaded")

    all_results = []   # collect per-image results for summary figure

    for sample in SAMPLE_IMAGES:
        # Find image file
        img_path = os.path.join(IMAGE_DIR, sample["name"] + ".jpg")
        if not os.path.exists(img_path):
            for ext in [".JPEG", ".jpeg", ".png"]:
                alt = os.path.join(IMAGE_DIR, sample["name"] + ext)
                if os.path.exists(alt):
                    img_path = alt
                    break
        if not os.path.exists(img_path):
            print("\n[Skip]", sample["name"], "- image not found in", IMAGE_DIR)
            continue

        print("\n" + "=" * 55)
        print("[Sample]", sample["desc"])

        img_pil, img_tensor = load_image(img_path)

        # Check prediction
        with torch.no_grad():
            inp = img_tensor.unsqueeze(0).to(device)
            out = torch.nn.functional.softmax(model(inp), dim=1)
        top3 = out[0].topk(3)
        print("[Top-3]")
        for prob, idx in zip(top3.values.cpu().numpy(),
                             top3.indices.cpu().numpy()):
            print("  class %d : %.2f%%" % (idx, prob * 100))
        print("[True class %d] prob: %.2f%%" % (
            sample["label_idx"], out[0, sample["label_idx"]].item() * 100))

        results = {}

        # 1. Vanilla Saliency
        print("[1] Vanilla Saliency Map...")
        results["Vanilla Saliency"] = vanilla_saliency(
            model, img_tensor, sample["label_idx"])

        # 2. SmoothGrad
        print("[2] SmoothGrad (n=%d)..." % SMOOTHGRAD_N)
        results["SmoothGrad"] = smooth_grad(
            model, img_tensor, sample["label_idx"])

        # 3. Guided BP
        print("[3] Guided Backpropagation...")
        results["Guided BP"] = guided_backprop(
            model, img_tensor, sample["label_idx"])

        # Save all maps as npy
        for method_name, sal in results.items():
            fname = sample["name"] + "_" + method_name.replace(" ", "_") + ".npy"
            np.save(os.path.join(OUTPUT_DIR, fname), sal)

        # Save individual comparison figure
        save_comparison(img_pil, results, sample["name"], sample["desc"])

        # Store for summary figure
        all_results.append({
            "desc":    sample["desc"],
            "img_pil": img_pil,
            "results": results,
        })

    # ── Summary figure: all images x all methods in one file ──
    if all_results:
        method_names = list(all_results[0]["results"].keys())
        n_images  = len(all_results)
        n_cols    = len(method_names) + 1   # Original + each method
        # 2 rows per image: saliency map row + overlay row
        n_rows    = n_images * 2

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 5 * n_images)
        )
        fig.suptitle(
            "BP Visualization Summary\n"
            "Odd rows: Saliency map  /  Even rows: Overlay on original",
            fontsize=14, y=1.01
        )

        for i, res in enumerate(all_results):
            img_pil_i = res["img_pil"]
            img_arr   = np.array(img_pil_i)
            row_sal  = i * 2       # saliency map row
            row_over = i * 2 + 1   # overlay row

            # Col 0: original image (both rows)
            axes[row_sal,  0].imshow(img_pil_i)
            axes[row_sal,  0].set_title(res["desc"], fontsize=10)
            axes[row_sal,  0].axis("off")
            axes[row_over, 0].imshow(img_pil_i)
            axes[row_over, 0].set_title("(original)", fontsize=9)
            axes[row_over, 0].axis("off")

            for j, mname in enumerate(method_names):
                sal = normalize_map(res["results"][mname])
                col = j + 1

                # Saliency map
                axes[row_sal, col].imshow(sal, cmap="hot")
                axes[row_sal, col].set_title(mname, fontsize=10)
                axes[row_sal, col].axis("off")

                # Overlay
                overlaid = overlay_saliency(img_arr, res["results"][mname])
                axes[row_over, col].imshow(overlaid)
                axes[row_over, col].set_title(mname + " (overlay)", fontsize=9)
                axes[row_over, col].axis("off")

        plt.tight_layout()
        summary_path = os.path.join(OUTPUT_DIR, "bp_all_summary.png")
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("\n[Saved summary]", summary_path)

    print("\n[Done] All results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
