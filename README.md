# DNN 実践課題 1（3b） - 古典課題

## 概要

本リポジトリは，深層学習（PyTorch）を用いた各種課題の実装および実験結果をまとめたものです．
課題は複数（全 11 問）あり，各課題ごとにフォルダを分けて管理しています．

---

## 環境

### ソフトウェア環境

- Miniconda3 (version 4.12.0, Python 3.8)（仮想環境管理）
- Python 3.8.13
- PyTorch 2.4.1
- CUDA 12.1

### 実験環境

- CPU
  - Intel Xeon E5-2640 v4 @ 2.40GHz（10 cores）
- メモリ
  - 約 257GB
- GPU
  - NVIDIA RTX 4090 × 4
    - メモリ: 約 24GB / GPU

---

## リポジトリ構成

```
DNN-Practical-Assignment/
├── ex1/   # Multi GPU
├── ex2/   # CNN可視化1（Occlusion）
├── ex3/   # CNN可視化2（BP）
├── ex4/ # CNN可視化3（Grad-CAM）
├── ex5/ # DeepDream
├── ex6/ # DCGAN
├── ex7/ # Conditional GAN
├── ex8/ # Style Transfer
├── ex9/pytorch-CycleGAN-and-pix2pix # Pix2Pix source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
├── ex10/ # CycleGAN
├── ex11/ # Diffusion
└── README.md
```

---

## 各課題

### ex1: Multi GPU による学習

- CIFAR10/100 を用いた学習時間の比較
- Single / Dual / Quad GPU で比較
- Data Parallel による並列化

---

### ex2: CNN 可視化（Occlusion）

- 入力画像に occluder を適用
- ヒートマップ生成による位置推定

---

### ex3: CNN 可視化（Backpropagation）

- 勾配を用いた可視化（Saliency Map）
- VGG16 を使用

---

### ex4: CAM / Grad-CAM

- ResNet を用いた可視化
- 認識領域のヒートマップ生成

---

### ex5: DeepDream

- 中間層の特徴を強調
- 画像生成（5 枚以上）

---

### ex6: DCGAN

- 画像生成モデルの学習
- PyTorch による実装

---

### ex7: Conditional GAN

- ラベル条件付き生成
- 連続変化による生成の観察

---

### ex8: Neural Style Transfer

- コンテンツ画像 + スタイル画像
- 高速 Style Transfer も検証

---

### ex9: Pix2Pix

- ペア画像による変換
- U-Net 構造の検証

---

### ex10: CycleGAN

- 非ペア画像変換
- ドメイン間変換

---

### ex11: Diffusion

- 拡散モデルによる画像生成

---

## 使用データセット

- CIFAR10 / CIFAR100
- ImageNet（pretrained model）
- COCO dataset
- UECFOOD dataset

---

## 備考

本リポジトリは講義課題提出用として作成されています．
