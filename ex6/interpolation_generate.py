# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# Parameters  
# =========================================================
image_size  = 64
nz          = 100
ngf         = 64
num_classes = 100
embed_dim   = num_classes   

model_path = "./results/conditional_dcgan/netG_final.pth"
save_dir   = "./results/condition_interpolation"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Generator  
# =========================================================
class ConditionalGenerator(nn.Module):
    def __init__(self, num_classes=num_classes, nz=nz, embed_dim=embed_dim, ngf=ngf, nc=3):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        self.label_emb = nn.Embedding(num_classes, embed_dim)

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
        if labels.dtype in (torch.float32, torch.float64):
            embed = labels                          
        else:
            embed = self.label_emb(labels)         

        embed = embed.unsqueeze(2).unsqueeze(3)     
        x = torch.cat([noise, embed], dim=1)       
        return self.main(x)


# =========================================================
# Load Generator
# =========================================================
netG = ConditionalGenerator().to(device)

state_dict = torch.load(model_path, map_location=device, weights_only=True)
new_state_dict = {
    (k[7:] if k.startswith("module.") else k): v
    for k, v in state_dict.items()
}

netG.load_state_dict(new_state_dict)
netG.eval()
print("Generator loaded successfully")

# 補間設定
class_A   = 18    # 補間開始クラス(pizza)
class_B   = 19    # 補間終了クラス(sandwich)
n_steps   = 11   # 補間ステップ数 (alpha: 1.0 → 0.0)
n_samples = 8    # 各ステップで生成するサンプル数（固定ノイズの種類）

# 埋め込み行列から補間ベクトルを計算

# 学習時: label_emb(整数ラベル) → W の該当行を取り出す
# 補間時: α * W[class_A] + (1-α) * W[class_B]  ← 同じ埋め込み空間で補間
# =========================================================
W     = netG.label_emb.weight.detach()  # (num_classes, embed_dim)
vec_A = W[class_A]                      # (embed_dim,)
vec_B = W[class_B]                      # (embed_dim,)

print(f"Interpolating class {class_A} → class {class_B}  ({n_steps} steps, {n_samples} samples)")


fixed_noise = torch.randn(n_samples, nz, 1, 1, device=device)

alphas         = torch.linspace(1.0, 0.0, steps=n_steps) 
generated_rows = []   

with torch.no_grad():
    for alpha in alphas:
        alpha = alpha.item()

        interp_vec = alpha * vec_A + (1.0 - alpha) * vec_B   # (embed_dim,)

        condition = interp_vec.unsqueeze(0).expand(n_samples, -1).to(device)

        fake = netG(fixed_noise, condition).cpu()   # (n_samples, 3, 64, 64)
        generated_rows.append(fake)

        print(f"  alpha={alpha:.1f}  A:{alpha:.1f}  B:{1-alpha:.1f}")


all_images = torch.cat(generated_rows, dim=0)   # (n_steps * n_samples, C, H, W)

vutils.save_image(
    all_images,
    f"{save_dir}/interpolation_grid.png",
    nrow=n_samples,
    normalize=True,
    padding=2,
)
print(f"Saved: {save_dir}/interpolation_grid.png")


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