# -*- coding: utf-8 -*-
"""
Conditional DCGAN: クラス補間スクリプト（アーキテクチャ修正版）

チェックポイントのエラーから判明した学習時の実際の構造:
  label_emb.weight : [100, 100]  → embed_dim = num_classes = 100
  main.0.weight    : [200, 512]  → 入力次元 = nz(100) + embed_dim(100) = 200
"""

import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# Parameters  ← チェックポイントに合わせた正しい値
# =========================================================
image_size  = 64
nz          = 100
ngf         = 64
num_classes = 100
embed_dim   = num_classes   # ★ 100（チェックポイントから判明）

model_path = "./results/conditional_dcgan/netG_final.pth"
save_dir   = "./results/condition_interpolation"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Generator  ← チェックポイントに完全一致するアーキテクチャ
# =========================================================
class ConditionalGenerator(nn.Module):
    def __init__(self, num_classes=num_classes, nz=nz, embed_dim=embed_dim, ngf=ngf, nc=3):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim   = embed_dim

        # embed_dim = 100 → label_emb.weight: [100, 100] ✓
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        # main.0.weight の入力次元 = nz + embed_dim = 200 ✓
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + embed_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf,   4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        """
        noise  : (B, nz, 1, 1)
        labels : (B,) LongTensor        ← 通常生成時
                 (B, embed_dim) Float   ← 補間時（埋め込み空間の補間ベクトル）
        """
        if labels.dtype in (torch.float32, torch.float64):
            embed = labels                          # (B, embed_dim) — 補間済みベクトル
        else:
            embed = self.label_emb(labels)          # (B, embed_dim)

        embed = embed.unsqueeze(2).unsqueeze(3)     # (B, embed_dim, 1, 1)
        x = torch.cat([noise, embed], dim=1)        # (B, nz+embed_dim, 1, 1)
        return self.main(x)


# =========================================================
# Load Generator
# =========================================================
netG = ConditionalGenerator().to(device)

state_dict = torch.load(model_path, map_location=device, weights_only=True)

# DataParallel で保存された場合の "module." プレフィックスを除去
new_state_dict = {
    (k[7:] if k.startswith("module.") else k): v
    for k, v in state_dict.items()
}

netG.load_state_dict(new_state_dict)
netG.eval()
print("Generator loaded successfully")


# =========================================================
# 補間設定
# =========================================================
class_A   = 0    # 補間開始クラス
class_B   = 1    # 補間終了クラス
n_steps   = 11   # 補間ステップ数 (alpha: 1.0 → 0.0)
n_samples = 8    # 各ステップで生成するサンプル数（固定ノイズの種類）


# =========================================================
# 埋め込み行列から補間ベクトルを計算
#
# 学習時: label_emb(整数ラベル) → W の該当行を取り出す
# 補間時: α * W[class_A] + (1-α) * W[class_B]  ← 同じ埋め込み空間で補間
# =========================================================
W     = netG.label_emb.weight.detach()  # (num_classes, embed_dim)
vec_A = W[class_A]                      # (embed_dim,)
vec_B = W[class_B]                      # (embed_dim,)

print(f"Interpolating class {class_A} → class {class_B}  ({n_steps} steps, {n_samples} samples)")


# =========================================================
# 固定ノイズ（ノイズを固定することで条件のみの変化を可視化）
# =========================================================
fixed_noise = torch.randn(n_samples, nz, 1, 1, device=device)


# =========================================================
# 補間画像を生成
# =========================================================
alphas         = torch.linspace(1.0, 0.0, steps=n_steps)  # 1.0(A) → 0.0(B)
generated_rows = []   # 各ステップの生成画像 (n_samples, C, H, W)

with torch.no_grad():
    for alpha in alphas:
        alpha = alpha.item()

        # 埋め込み空間での線形補間
        interp_vec = alpha * vec_A + (1.0 - alpha) * vec_B   # (embed_dim,)

        # n_samples 分に拡張して渡す
        condition = interp_vec.unsqueeze(0).expand(n_samples, -1).to(device)

        fake = netG(fixed_noise, condition).cpu()   # (n_samples, 3, 64, 64)
        generated_rows.append(fake)

        print(f"  alpha={alpha:.1f}  A:{alpha:.1f}  B:{1-alpha:.1f}")


# =========================================================
# 保存①: vutils グリッド（行=補間ステップ、列=サンプル）
# =========================================================
all_images = torch.cat(generated_rows, dim=0)   # (n_steps * n_samples, C, H, W)

vutils.save_image(
    all_images,
    f"{save_dir}/interpolation_grid.png",
    nrow=n_samples,
    normalize=True,
    padding=2,
)
print(f"Saved: {save_dir}/interpolation_grid.png")


# =========================================================
# 保存②: matplotlib（α ラベル付き）
# =========================================================
fig, axes = plt.subplots(n_steps, n_samples, figsize=(n_samples * 2, n_steps * 2))

for row_i, (imgs, alpha) in enumerate(zip(generated_rows, alphas.tolist())):
    for col_i in range(n_samples):
        img = imgs[col_i].permute(1, 2, 0).numpy()
        img = ((img + 1.0) / 2.0).clip(0, 1)
        ax  = axes[row_i][col_i]
        ax.imshow(img)
        ax.axis("off")
        if col_i == 0:
            ax.set_ylabel(
                f"A:{alpha:.1f}\nB:{1-alpha:.1f}",
                fontsize=7, rotation=0, labelpad=35, va="center"
            )

plt.suptitle(f"Interpolation: Class {class_A} → Class {class_B}", fontsize=12)
plt.tight_layout()
plt.savefig(f"{save_dir}/interpolation_matplotlib.png", dpi=150)
plt.close()

print(f"Saved: {save_dir}/interpolation_matplotlib.png")
print("Finished")