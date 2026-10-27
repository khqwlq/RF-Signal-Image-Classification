# SkyScope IASS 2026: RF Signal Image Classification
### 🥈 2nd Place Standings | Final Score: 0.97064

This repository contains my end-to-end deep learning solution for the **SkyScope IASS 2026** competition, achieving an accuracy score of **97.06%** on the final private leaderboard.

## 📡 Project Overview
The task involves accurately classifying **32 different types of Radio Frequency (RF) signals** (such as WiFi, LoRa, ZigBee, and UAV video links) using waterfall spectrogram images. The dataset features a severe class imbalance, ranging from 4 to 590 samples per signal type.

## 🗂️ Core Pipeline Features
* **Data Cleaning:** Implemented custom regex filtering to process and recover malformed raw CSV tracking logs.
* **Architecture:** Trained an `EfficientNet-B0` backbone using differential learning rates ($2.9 \times 10^{-5}$ for the backbone, $3.0 \times 10^{-4}$ for the classification head) to preserve pretrained feature extractors.
* **Hardware Acceleration:** Optimized using PyTorch/XLA to execute accelerated parallel training loops on Cloud **TPU v5e** clusters.
* **Inference Optimization:** Implemented a 7-step Test-Time Augmentation (TTA) loop (including horizontal/vertical flips and color jittering) to stabilize test set predictions.

## 📊 Results & Validation
* **Private Leaderboard Score:** 0.97064 (2nd Position out of the final standings).
* A verified screenshot of the final competition leaderboard is available in the `/images` directory as empirical validation of the model's performance.

## 👤 Author
* **Khaoula Bendjeddou** - Master 1 in Space Telecommunications, University of Blida 1.
