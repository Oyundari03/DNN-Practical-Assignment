# -*- coding: utf-8 -*-
"""
Multi-GPU Training: CIFAR-10 with ResNet152
Kadai 1: Compare training time with 1/2/4 GPUs

Usage:
  python train_multigpu.py --num_gpus 1 --dataset cifar10
  python train_multigpu.py --num_gpus 2 --dataset cifar10
  python train_multigpu.py --num_gpus 4 --dataset cifar10

"""

from __future__ import print_function

import os
import time
import argparse
import csv

# Set CUDA_VISIBLE_DEVICES BEFORE importing torch
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--num_gpus", type=int, default=1)
pre_args, _ = pre_parser.parse_known_args()

gpu_ids_str = ",".join(str(i) for i in range(pre_args.num_gpus))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
print("[GPU] CUDA_VISIBLE_DEVICES =", gpu_ids_str)

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18


def get_args():
    parser = argparse.ArgumentParser(description="Multi-GPU CIFAR Training")
    parser.add_argument("--num_gpus", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_per_gpu", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="/export/space0/oyundari/dataser/cifar")
    return parser.parse_args()


def get_dataloader(dataset_name, batch_size, num_workers, data_dir):
    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
        num_classes = 10
        DatasetClass = torchvision.datasets.CIFAR10
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)
        num_classes = 100
        DatasetClass = torchvision.datasets.CIFAR100

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = DatasetClass(root=data_dir, train=True,
                                 download=True, transform=train_transform)
    test_dataset  = DatasetClass(root=data_dir, train=False,
                                 download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes


def replace_bn_with_gn(module):
    """Replace all BatchNorm2d with GroupNorm to avoid cuDNN issues."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            # GroupNorm with 32 groups (or fewer if channels < 32)
            num_groups = min(32, num_channels)
            # Make sure num_channels divisible by num_groups
            while num_channels % num_groups != 0:
                num_groups -= 1
            setattr(module, name, nn.GroupNorm(num_groups, num_channels))
        else:
            replace_bn_with_gn(child)
    return module


def build_model(num_classes, num_gpus):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Replace BatchNorm with GroupNorm to avoid cuDNN EXECUTION_FAILED
    model = replace_bn_with_gn(model)
    print("[Model] Replaced BatchNorm2d with GroupNorm (cuDNN workaround)")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if num_gpus > 1 and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
        print("[Model] DataParallel on", num_gpus, "GPUs")
    else:
        print("[Model] Single GPU")

    return model, device


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100.0 * correct / total


def main():
    args = get_args()
    total_batch = args.batch_per_gpu * args.num_gpus

    print("[Config] Dataset     :", args.dataset.upper())
    print("[Config] GPUs        :", args.num_gpus)
    print("[Config] Total batch :", total_batch, "(", args.batch_per_gpu, "per GPU)")
    print("[Config] Epochs      :", args.epochs)
    print("[Config] Workers     :", args.num_workers)

    train_loader, test_loader, num_classes = get_dataloader(
        args.dataset, total_batch, args.num_workers, args.data_dir)

    model, device = build_model(num_classes, args.num_gpus)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[5, 8], gamma=0.1)

    sep = "=" * 72
    print("\n" + sep)
    print("  Epoch | Train Loss | Train Acc |  Val Loss |  Val Acc | Time(s)")
    print(sep)

    history = []
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({"epoch": epoch,
                         "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss,     "val_acc": val_acc,
                         "time": elapsed})

        print("  %5d | %10.4f | %8.2f%% | %9.4f | %7.2f%% | %7.1f" % (
            epoch, train_loss, train_acc, val_loss, val_acc, elapsed))

    total_time = time.time() - total_start
    print(sep)
    print("\n[Result] Total time    : %.1f s  (%.1f min)" % (
        total_time, total_time / 60.0))
    print("[Result] Avg per epoch : %.1f s" % (total_time / args.epochs))
    print("[Result] Final Val Acc : %.2f%%" % history[-1]["val_acc"])

    csv_name = "result_%s_%dgpu.csv" % (args.dataset, args.num_gpus)
    with open(csv_name, "w") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch","train_loss","train_acc","val_loss","val_acc","time"])
        writer.writeheader()
        writer.writerows(history)
    print("[Saved]", csv_name)

    if torch.cuda.is_available():
        for i in range(args.num_gpus):
            props = torch.cuda.get_device_properties(i)
            print("[GPU:%d] %s  %d MB" % (i, props.name,
                                          props.total_memory // 1024**2))


if __name__ == "__main__":
    main()