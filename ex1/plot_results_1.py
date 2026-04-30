#!/usr/local/anaconda3/bin/python3
"""
Multi-GPU 実験結果の比較グラフ作成スクリプト
train_multigpu.py が出力した CSV を読み込んでグラフ化

使い方:
  python plot_results.py
  # result_cifar10_1gpu.csv / result_cifar10_2gpu.csv / result_cifar10_4gpu.csv
  # が同じディレクトリにある前提
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIGS = [
    ("result_cifar10_1gpu.csv",  "1 GPU",  "blue"),
    ("result_cifar10_2gpu.csv",  "2 GPU",  "orange"),
    ("result_cifar10_4gpu.csv",  "4 GPU",  "green"),
]

def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Multi-GPU Training Comparison (ResNet152, CIFAR-10)", fontsize=14)

for path, label, color in CONFIGS:
    if not os.path.exists(path):
        print(f"[Skip] {path} not found")
        continue
    data = load_csv(path)
    epochs    = [d["epoch"]     for d in data]
    val_acc   = [d["val_acc"]   for d in data]
    train_loss = [d["train_loss"] for d in data]
    times     = [d["time"]      for d in data]

    axes[0].plot(epochs, val_acc,    label=label, color=color)
    axes[1].plot(epochs, train_loss, label=label, color=color)
    axes[2].plot(epochs, times,      label=label, color=color)

axes[0].set_title("Validation Accuracy")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
axes[0].legend(); axes[0].grid(True)

axes[1].set_title("Training Loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True)

axes[2].set_title("Time per Epoch")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Seconds")
axes[2].legend(); axes[2].grid(True)

plt.tight_layout()
plt.savefig("multigpu_comparison.png", dpi=150)
print("[Saved] multigpu_comparison.png")
