# Pix2Pix / CycleGAN Experiment

## 概要

本課題では、画像変換モデルである Pix2Pix および CycleGAN を PyTorch 実装で学習・評価した。

- Pix2Pix: ペア画像を用いた Image-to-Image Translation
- CycleGAN: ペア画像を必要としない Unpaired Image-to-Image Translation

実験には公式実装を利用し、データセットをダウンロードして学習および推論を行った。

---

## 参考文献・引用元

### Official Repository

Jun-Yan Zhu et al.

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

### Pix2Pix

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros

Image-to-Image Translation with Conditional Adversarial Networks

https://arxiv.org/abs/1611.07004

### CycleGAN

Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

https://arxiv.org/abs/1703.10593

---

# Pix2Pix

## 使用データセット

facades

建物ラベル画像と建物写真のペア画像データセット

### データセットダウンロード

```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```

## 学習

```bash
python train.py \
--dataroot ./datasets/facades \
--name facades_pix2pix \
--model pix2pix \
--direction BtoA
```

## テスト

```bash
python test.py \
--dataroot ./datasets/facades \
--direction BtoA \
--model pix2pix \
--name facades_label2photo_pretrained \
--use_wandb
```

## 学習して分かったこと

Pix2Pix は入力画像と正解画像が対応したペアデータを利用して学習する Conditional GAN である。

Generator には U-Net が使用されており、Encoder で抽出した特徴を Skip Connection により Decoder に直接伝達することで、入力画像の構造情報を保持しながら画像生成を行う。

そのため、建物の輪郭や窓などの位置関係を維持したまま、ラベル画像から実写風画像への変換が可能であることを確認できた。

## 結果確認方法

テスト後に以下が生成される。

```text
results/
└── facades_label2photo_pretrained/test_latest
    └── images/
```

特に

```text
index.html
```

をブラウザで開くことで結果を確認できる。

比較画像には以下が含まれる。

- real_A : 入力画像
- fake_B : 生成画像
- real_B : 正解画像

生成画像が正解画像にどれだけ近いかを確認することで性能を評価できる。

---

# CycleGAN

## 使用データセット

horse2zebra

- Horse
- Zebra

の非対応画像セット

### データセットダウンロード

```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

## 学習

```bash
python train.py \
--dataroot ./datasets/horse2zebra \
--name horse2zebra \
--model cycle_gan
```

## テスト

```bash
python test.py \
--dataroot datasets/horse2zebra/testA \
--name horse2zebra_pretrained \
--model test \
--no_dropout
```

## 学習して分かったこと

CycleGAN は Pix2Pix と異なり、入力画像と正解画像の対応関係を必要としない。

Cycle Consistency Loss を導入することで、

Horse → Zebra → Horse

のように変換後に元画像へ戻した際の誤差を最小化し、画像内容を保持しながらスタイル変換を実現している。

そのため、ペアデータが存在しない場合でも画像変換が可能であり、実際に馬画像からシマウマ画像への変換を行うことができた。

## 結果確認方法

テスト後に以下が生成される。

```text
results/
└── horse2zebra_pretrained/test_latest
    └── images/
```

フォルダ内の

```text
index.html
```

をブラウザで開くことで結果を確認できる。

主な出力画像は以下である。

- real_A : 入力画像（Horse）
- fake_B : 生成画像（Zebra）
- rec_A : 再構成画像

生成画像では馬の形状を維持しながらシマウマ模様が付与されていることを確認できる。

---

# Pix2Pix と CycleGAN の比較

| 項目       | Pix2Pix         | CycleGAN              |
| ---------- | --------------- | --------------------- |
| 学習データ | ペア画像必要    | ペア画像不要          |
| Generator  | U-Net           | ResNet                |
| 特徴       | 構造保持が得意  | スタイル変換が得意    |
| 代表例     | ラベル → 写真   | 馬 → シマウマ         |
| 学習方法   | Conditional GAN | Cycle Consistency GAN |

---

# まとめ

Pix2Pix と CycleGAN を実際に学習・推論することで、画像変換モデルの基本的な仕組みを理解することができた。

Pix2Pix はペア画像を利用するため高精度な変換が可能であり、CycleGAN は対応画像を必要とせず柔軟なドメイン変換を実現できることを確認した。

また、GAN による画像生成では Generator と Discriminator が競合的に学習することで高品質な画像生成が可能になることを学んだ。

## Pix2Pix Result

![Pix2Pix Result](results/pix2pix_result.png)

## CycleGAN Result

![CycleGAN Result](results/cyclegan_result.png)
