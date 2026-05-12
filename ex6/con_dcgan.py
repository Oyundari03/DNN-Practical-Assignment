# -*- coding: utf-8 -*-

import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =========================================================
# Parameters
# =========================================================

data_root = "./UECFOOD/UECFOOD100"
results_root = "./results/conditional_dcgan"
batch_size = 128
image_size = 64
nz = 100      # latent vector size
ngf = 64      # generator feature size
ndf = 64      # discriminator feature size
num_epochs = 200
lr = 0.0002
beta1 = 0.5
ngpu = 4

os.makedirs(results_root, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("===================================")
print(f"Device      : {device}")
print(f"GPU count   : {torch.cuda.device_count()}")
print(f"Using GPUs  : {ngpu}")
print("===================================")


# =========================================================
# Dataset
# =========================================================

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

dataset = ImageFolder(
    root=data_root,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

num_classes = len(dataset.classes)

print(f"Dataset size     : {len(dataset)}")
print(f"Number of class  : {num_classes}")


# =========================================================
# Conditional Generator
# =========================================================

class ConditionalGenerator(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + num_classes,ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        c = c.unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, c], dim=1)
        return self.main(x)


# =========================================================
# Conditional Discriminator
# =========================================================

class ConditionalDiscriminator(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.image_size = image_size
        self.label_embed = nn.Embedding(num_classes, image_size * image_size)
        self.main = nn.Sequential(
            nn.Conv2d(4, ndf, 4, 2, 1, bias=False ),
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

    def forward(self, x, labels):
        c = self.label_embed(labels)
        c = c.view(-1, 1, self.image_size, self.image_size)
        x = torch.cat([x, c], dim=1)
        return self.main(x)

netG = ConditionalGenerator(num_classes).to(device)
netD = ConditionalDiscriminator(num_classes).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    print(f"DataParallel: using {torch.cuda.device_count()} GPUs")
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(num_classes * 8, nz, 1, 1, device=device)
fixed_labels = torch.tensor(
    [i for i in range(num_classes) for _ in range(8)],device=device
)

# =========================================================
# Training
# =========================================================

G_losses = []
D_losses = []

time_start = time.time()

print("Start Training...")

for epoch in range(num_epochs):

    for i, (real_images, labels) in enumerate(dataloader):

        real_images = real_images.to(device)
        labels = labels.to(device)

        b_size = real_images.size(0)

        # =================================================
        # Train Discriminator
        # =================================================

        netD.zero_grad()

        # label smoothing
        real_targets = torch.full((b_size,),0.9, device=device)
        fake_targets = torch.zeros(b_size, device=device)

        # -----------------------------
        # Real images
        # -----------------------------

        output_real = netD(real_images, labels).view(-1)
        loss_real = criterion(output_real, real_targets)

        # -----------------------------
        # Fake images
        # -----------------------------

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = netG(noise, labels )

        output_fake = netD(fake_images.detach(), labels).view(-1)
        loss_fake = criterion(output_fake, fake_targets)

        lossD = loss_real + loss_fake
        lossD.backward()
        optimizerD.step()

        # =================================================
        # Train Generator
        # =================================================

        netG.zero_grad()

        output = netD(fake_images,labels).view(-1)
        lossG = criterion(output,real_targets)

        lossG.backward()
        optimizerG.step()

        G_losses.append(lossG.item())
        D_losses.append(lossD.item())

        if i % 50 == 0:

            print(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"Loss_D: {lossD.item():.4f} "
                f"Loss_G: {lossG.item():.4f}"
            )

    with torch.no_grad():
        fake = netG(fixed_noise,fixed_labels).detach().cpu()

    vutils.save_image(
        fake,
        f"{results_root}/epoch_{epoch+1:03d}.png",
        nrow=8,
        normalize=True
    )

    print(f"Saved: epoch_{epoch+1:03d}.png")

time_end = time.time()

print("Training Finished")
print(f"Total Time: {time_end - time_start:.2f} sec")

# Save loss curve

plt.figure(figsize=(10, 5))
plt.plot(G_losses, label="Generator")
plt.plot(D_losses, label="Discriminator")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("Conditional DCGAN Training Loss")
plt.savefig(
    f"{results_root}/loss_curve.png"
)
plt.close()

# Save models
torch.save(netG.state_dict(),f"{results_root}/netG_final.pth")
torch.save(netD.state_dict(),f"{results_root}/netD_final.pth")
print("Model Saved")