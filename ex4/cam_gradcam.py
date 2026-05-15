# -*- coding: utf-8 -*-
"""
Kadai4 : CAM / Grad-CAM Visualization
Model : ResNet50 (ImageNet Pretrained)
"""

import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Save Directory
os.makedirs("results", exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", device)

# Load Model
model = models.resnet50(pretrained=True)
model.eval()
model.to(device)

# Target Layer
# ResNet50 last conv layer
target_layer = model.layer4[-1]

# Hook
features = []
gradients = []

def forward_hook(module, input, output):
    features.append(output.detach())

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0].detach())

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ImageNet Labels
from torchvision.models import ResNet50_Weights
labels = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

# Image List
image_paths = [
    "../ex2/results/dog.jpg",
    "../ex2/results/cat.jpg",
    "../ex2/results/elephant.jpg"
]

# Processing Loop
for idx, img_path in enumerate(image_paths):

    print(f"\nProcessing : {img_path}")

    # Clear previous hooks data
    features.clear()
    gradients.clear()

    # Load Image
    pil_img = Image.open(img_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Forward
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    pred_class = output.argmax(dim=1).item()
    confidence = probs[0, pred_class].item()
    class_name = labels[pred_class]

    print(f"Prediction : {class_name}")
    print(f"Confidence: {confidence:.4f}")

    # Backward
    model.zero_grad()
    score = output[:, pred_class]
    score.backward()

    # Feature / Gradient
    feature = features[0][0]       # [C,H,W]
    gradient = gradients[0][0]     # [C,H,W]

    # CAM
    fc_weights = model.fc.weight[pred_class]
    cam = torch.zeros(feature.shape[1:], dtype=torch.float32).to(device)
    for i, w in enumerate(fc_weights):
        cam += w * feature[i]

    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max()
    cam_np = cam.detach().cpu().numpy()

    # Grad-CAM

    weights = gradient.mean(dim=(1,2))
    gradcam = torch.zeros(feature.shape[1:], dtype=torch.float32).to(device)
    for i, w in enumerate(weights):
        gradcam += w * feature[i]

    gradcam = torch.relu(gradcam)
    gradcam -= gradcam.min()
    gradcam /= gradcam.max()
    gradcam_np = gradcam.cpu().numpy()

    # Original Image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Resize Heatmaps
    cam_resized = cv2.resize(cam_np, (w, h))
    gradcam_resized = cv2.resize(gradcam_np, (w, h))

    # Heatmap
    cam_heatmap = np.uint8(255 * cam_resized)
    cam_heatmap = cv2.applyColorMap(cam_heatmap, cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    gradcam_heatmap = np.uint8(255 * gradcam_resized)
    gradcam_heatmap = cv2.applyColorMap(
        gradcam_heatmap,
        cv2.COLORMAP_JET
    )

    gradcam_heatmap = cv2.cvtColor(
        gradcam_heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Overlay
    cam_overlay = cam_heatmap * 0.4 + img * 0.6
    cam_overlay = np.uint8(cam_overlay)
    gradcam_overlay = gradcam_heatmap * 0.4 + img * 0.6
    gradcam_overlay = np.uint8(gradcam_overlay)

    # Visualization
    plt.figure(figsize=(20,5))

    # Original
    plt.subplot(1,5,1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    # CAM
    plt.subplot(1,5,2)
    plt.imshow(cam_resized, cmap="jet")
    plt.title("CAM")
    plt.axis("off")

    # Grad-CAM
    plt.subplot(1,5,3)
    plt.imshow(gradcam_resized, cmap="jet")
    plt.title("Grad-CAM")
    plt.axis("off")

    # CAM Overlay
    plt.subplot(1,5,4)
    plt.imshow(cam_overlay)
    plt.title("CAM Overlay")
    plt.axis("off")

    # Grad-CAM Overlay
    plt.subplot(1,5,5)
    plt.imshow(gradcam_overlay)
    plt.title("Grad-CAM Overlay")
    plt.axis("off")

    # Main Title
    plt.suptitle(
        f"Prediction: {class_name} ({confidence*100:.2f}%)",
        fontsize=16)
    plt.tight_layout()

    # Save
    save_path = f"results/result_{idx+1}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved : {save_path}")

print("\nAll Done!")
