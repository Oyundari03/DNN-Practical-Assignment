# -*- coding: utf-8 -*-
"""
Kadai 2: Occlusion Sensitivity - CPU/GPU compatible lightweight version
- Uses ResNet18 instead of VGG16 on CPU (much smaller memory footprint)
- Reduces occluder stride to 16 on CPU to speed up
"""
from __future__ import print_function
import os, sys, urllib, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# ── Settings ──────────────────────────────────────────────
INPUT_SIZE    = 256
OCCLUDER_SIZE = 64
OCC_START     = -56
OCC_END       = 247

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", device)

# Use larger stride on CPU to reduce computation
STRIDE = 8 if device.type == "cuda" else 16
print("[Stride]", STRIDE, "(8 for GPU, 16 for CPU to save time)")

OUTPUT_DIR = "./occlusion_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Sample images (already downloaded) ───────────────────
SAMPLE_IMAGES = [
    {"name": "dog",      "label_idx": 258, "desc": "Samoyed dog (class 258)"},
    {"name": "cat",      "label_idx": 281, "desc": "Tabby Cat (class 281)"},
    {"name": "elephant", "label_idx": 386, "desc": "African Elephant (class 386)"},
    {"name": "goldfish", "label_idx": 1,   "desc": "Goldfish (class 1)"},
    {"name": "peacock",  "label_idx": 84,  "desc": "Peacock (class 84)"},
]

# ── Preprocessing ─────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

def load_image(path):
    img_pil = Image.open(path).convert("RGB")
    img_pil = img_pil.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    return img_pil, preprocess(img_pil)

# ── Load model ────────────────────────────────────────────
print("[Model] Loading VGG16 pretrained...")
try:
    from torchvision.models import VGG16_Weights
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
except (ImportError, AttributeError):
    model = models.vgg16(pretrained=True)

model = model.to(device)
model.eval()
print("[Model] VGG16 loaded on", device)

# ── Load ImageNet labels ──────────────────────────────────
try:
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    labels = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
    print("[Labels] Loaded", len(labels), "labels")
except Exception as e:
    print("[Labels] Failed:", e, "-> using index numbers")
    labels = ["class_%d" % i for i in range(1000)]

# ── Occlusion map ─────────────────────────────────────────
def compute_occlusion_map(img_tensor, true_label_idx):
    positions = list(range(OCC_START, OCC_END, STRIDE))
    n = len(positions)
    heatmap = np.zeros((n, n), dtype=np.float32)
    img_np = img_tensor.numpy()
    H, W = img_np.shape[1], img_np.shape[2]
    softmax = nn.Softmax(dim=1)
    total = n * n
    count = 0

    # gray in normalized space
    gray = [(0.5 - 0.485) / 0.229,
            (0.5 - 0.456) / 0.224,
            (0.5 - 0.406) / 0.225]

    with torch.no_grad():
        for i, y in enumerate(positions):
            for j, x in enumerate(positions):
                occ = img_np.copy()
                y1, y2 = max(0, y), min(H, y + OCCLUDER_SIZE)
                x1, x2 = max(0, x), min(W, x + OCCLUDER_SIZE)
                if y2 > y1 and x2 > x1:
                    for c in range(3):
                        occ[c, y1:y2, x1:x2] = gray[c]
                inp = torch.from_numpy(occ).unsqueeze(0).to(device)
                prob = softmax(model(inp))[0, true_label_idx].item()
                heatmap[i, j] = prob
                count += 1
                if count % 50 == 0:
                    pct = 100.0 * count / total
                    sys.stdout.write("\r  Progress: %d/%d (%.1f%%)" % (count, total, pct))
                    sys.stdout.flush()
    print()
    return heatmap

# ── Main ──────────────────────────────────────────────────
# Collect results for summary figure
all_results = []   # list of dict: {name, desc, img_pil, heatmap, drop_map}

for sample in SAMPLE_IMAGES:
    img_path = os.path.join(OUTPUT_DIR, sample["name"] + ".jpg")
    # also check .JPEG extension
    if not os.path.exists(img_path):
        jpeg_path = os.path.join(OUTPUT_DIR, sample["name"] + ".JPEG")
        if os.path.exists(jpeg_path):
            img_path = jpeg_path
        else:
            print("\n[Skip]", sample["name"], "not found in", OUTPUT_DIR)
            print("  -> Please place the image as:", img_path)
            continue

    print("\n" + "=" * 55)
    print("[Sample]", sample["desc"])

    img_pil, img_tensor = load_image(img_path)

    # Top-5 prediction
    with torch.no_grad():
        out = torch.nn.functional.softmax(
            model(img_tensor.unsqueeze(0).to(device)), dim=1)
    top5 = out[0].topk(5)
    print("[Top-5 Prediction]")
    for prob, idx in zip(top5.values.cpu().numpy(),
                         top5.indices.cpu().numpy()):
        print("  %5.2f%%  %s (class %d)" % (
            prob*100, labels[idx] if idx < len(labels) else "?", idx))

    true_prob = out[0, sample["label_idx"]].item()
    print("[True label] %s : %.2f%%" % (sample["desc"], true_prob * 100))

    # Occlusion map
    print("[Occlusion] Computing... (stride=%d)" % STRIDE)
    heatmap = compute_occlusion_map(img_tensor, sample["label_idx"])
    print("[Occlusion] shape=%s  min=%.4f  max=%.4f" % (
        str(heatmap.shape), heatmap.min(), heatmap.max()))

    # Compute drop map: how much prob drops vs original
    # Negative = prob dropped (object region), positive = prob stayed
    drop_map = true_prob - heatmap   # large value = important region

    print("[Occlusion] drop_map min=%.4f  max=%.4f" % (
        drop_map.min(), drop_map.max()))

    # Save npy
    np.save(os.path.join(OUTPUT_DIR, sample["name"] + "_heatmap.npy"), heatmap)
    np.save(os.path.join(OUTPUT_DIR, sample["name"] + "_dropmap.npy"), drop_map)

    # Plot: 3 panels
    #   1) Original image
    #   2) Raw probability heatmap
    #   3) Drop map (probability drop = important region)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Occlusion Sensitivity (Zeiler-Fergus)  " + sample["desc"], fontsize=12)

    # Panel 1: original
    axes[0].imshow(img_pil)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Panel 2: raw probability (may look flat if confidence is high)
    im1 = axes[1].imshow(heatmap, cmap="jet", interpolation="bilinear",
                          vmin=heatmap.min(), vmax=heatmap.max())
    axes[1].set_title("Raw Probability\n(Red=high, Blue=low)")
    axes[1].set_xlabel("Occluder X (stride=%d px)" % STRIDE)
    axes[1].set_ylabel("Occluder Y (stride=%d px)" % STRIDE)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: probability DROP (this is the key visualization)
    # Red = large drop = model was using this region
    # Blue = small drop = background / unimportant
    im2 = axes[2].imshow(drop_map, cmap="hot", interpolation="bilinear",
                          vmin=0, vmax=drop_map.max())
    axes[2].set_title("Probability DROP\n(Bright=important region)")
    axes[2].set_xlabel("Occluder X (stride=%d px)" % STRIDE)
    axes[2].set_ylabel("Occluder Y (stride=%d px)" % STRIDE)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, sample["name"] + "_occlusion.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved individual]", out_path)
    print("[Drop map] max drop: %.4f at row/col:" % drop_map.max(),
          divmod(drop_map.argmax(), drop_map.shape[1]))

    # Store for summary figure
    all_results.append({
        "name":     sample["name"],
        "desc":     sample["desc"],
        "img_pil":  img_pil,
        "heatmap":  heatmap,
        "drop_map": drop_map,
    })

# ── Summary figure: all 5 images in one file ──────────────
if all_results:
    n = len(all_results)
    # Layout: rows = each image, cols = Original / Raw Prob / Drop Map
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    fig.suptitle(
        "Occlusion Sensitivity Map (Zeiler-Fergus)\n"
        "Left: Original  /  Center: Raw Probability  /  Right: Probability Drop",
        fontsize=14, y=1.01
    )

    for row, res in enumerate(all_results):
        ax_img  = axes[row, 0]
        ax_raw  = axes[row, 1]
        ax_drop = axes[row, 2]

        # Col 0: original image
        ax_img.imshow(res["img_pil"])
        ax_img.set_title(res["desc"], fontsize=11)
        ax_img.axis("off")

        # Col 1: raw probability heatmap
        hm = res["heatmap"]
        im1 = ax_raw.imshow(hm, cmap="jet", interpolation="bilinear",
                             vmin=hm.min(), vmax=hm.max())
        ax_raw.set_title("Raw Probability", fontsize=10)
        ax_raw.set_xlabel("Occluder X (stride=%d px)" % STRIDE)
        ax_raw.set_ylabel("Occluder Y (stride=%d px)" % STRIDE)
        plt.colorbar(im1, ax=ax_raw, fraction=0.046, pad=0.04)

        # Col 2: probability drop map (bright = important)
        dm = res["drop_map"]
        im2 = ax_drop.imshow(dm, cmap="hot", interpolation="bilinear",
                              vmin=0, vmax=dm.max())
        ax_drop.set_title("Probability Drop\n(Bright = important)", fontsize=10)
        ax_drop.set_xlabel("Occluder X (stride=%d px)" % STRIDE)
        ax_drop.set_ylabel("Occluder Y (stride=%d px)" % STRIDE)
        plt.colorbar(im2, ax=ax_drop, fraction=0.046, pad=0.04)

    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, "occlusion_all_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[Saved summary]", summary_path)

print("\n[Done] Results in:", OUTPUT_DIR)