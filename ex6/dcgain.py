# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# Seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ======================
# Parameters
# ======================
data_root = "./UECFOOD/UECFOOD100"
results_root = "./results/dcgan"
batch_size = 128
image_size = 64
nz = 100      # latent vector size
ngf = 64      # generator feature size
ndf = 64      # discriminator feature size
num_epochs = 200
ngpu = 4
lr = 0.0002
beta1 = 0.5

os.makedirs(results_root, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("===================================")
print(f"Device      : {device}")
print(f"GPU count   : {torch.cuda.device_count()}")
print(f"Using GPUs  : {ngpu}")
print("===================================")

# ======================
# Dataset
# ======================

transform = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),
                         (0.5,0.5,0.5))
])

dataset = ImageFolder(
    root=data_root,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)
print(f"Dataset size: {len(dataset)},Classes: {dataset.classes}")
# ======================
# Generator
# ======================

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4,1,0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4,2,1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 4,2,1,bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# ======================
# Discriminator
# ======================

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4,2,1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4,1,0,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

netG = Generator().to(device)
netD = Discriminator().to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    print(f"DataParallel: using {torch.cuda.device_count()} GPUs")
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# ======================
# Training
# ======================
G_losses, D_losses = [], []
img_list = []
time_start = time.time()
for epoch in range(num_epochs):

    for i, (real_images, _) in enumerate(dataloader):

        ############################
        # Train Discriminator
        ############################

        netD.zero_grad()

        real_images = real_images.to(device)
        b_size = real_images.size(0)

        label_real = torch.ones(b_size, device=device)
        label_fake = torch.zeros(b_size, device=device)

        output_real = netD(real_images).view(-1)
        loss_real = criterion(output_real, label_real)

        noise = torch.randn(b_size, nz,1,1, device=device)
        fake_images = netG(noise)

        output_fake = netD(fake_images.detach()).view(-1)
        loss_fake = criterion(output_fake, label_fake)

        lossD = loss_real + loss_fake
        lossD.backward()
        optimizerD.step()

        ############################
        # Train Generator
        ############################

        netG.zero_grad()

        output = netD(fake_images).view(-1)
        lossG = criterion(output, label_real)

        lossG.backward()
        optimizerG.step()

        G_losses.append(lossG.item())
        D_losses.append(lossD.item())

        if i % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] "
                  f"LossD: {lossD.item():.4f} "
                  f"LossG: {lossG.item():.4f}")

    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()

    save_image(fake,
               f"{results_root}/epoch_{epoch:03d}.png",
               normalize=True,
               nrow=8)
    
time_end = time.time()
print("Training Finished")
print(f"Total epoch: {epoch+1} \n Total Training Time: {time_end - time_start:.2f} seconds")

# Save loss curve

plt.figure(figsize=(10, 5))
plt.plot(G_losses, label="Generator")
plt.plot(D_losses, label="Discriminator")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("DCGAN Training Loss")
plt.savefig(f"{results_root}/loss_curve.png")
plt.close()

# Save models
torch.save(netG.state_dict(), f"{results_root}/netG_final.pth")
torch.save(netD.state_dict(), f"{results_root}/netD_final.pth")
print("Model Saved")