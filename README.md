# Noise-Aware Adaptation of Pre-trained Foundation Models for Single-Photon Image Classification (TMLR)

Official implementation of **Noise-Adapter**

📄 **Paper**:  
*Noise-Aware Adaptation of Pre-trained Foundation Models for Single-Photon Image Classification*  
Accepted by TMLR  
https://openreview.net/pdf?id=qSnrIy6Ohb

---

## Overview

Pre-trained foundation models (e.g., CLIP) are typically trained on large RGB datasets and do not account for sensor-specific acquisition effects. For single-photon LiDAR, photon statistics and sensor noise induce appearance shifts that can significantly degrade transfer performance in low-label regimes. We propose a **noise-aware adaptation framework** that conditions model adaptation on sensor acquisition statistics. A lightweight **Noise Adapter** modulates pre-trained visual features using summary statistics computed from raw single-photon measurements, improving robustness and few-shot performance.

---

## Supported Methods

The main training script supports the following baselines:

- Linear / MLP probing  
- CLIP-Adapter  
- Tip-Adapter  
- Tip-Adapter-F  
- Meta-Adapter  
- **Noise Adapter (ours)**

All methods operate on fixed pre-trained visual features.

---

## Running Experiments

### Method and Few-Shot Configuration

At the end of `train_adapter.py`, set:

```python
methods = ['noise_adapter_feasaug']
# methods = ['probing', 'clip_adapter', 'tip_adapter',
#            'tip_adapter_f', 'meta_adapter', 'noise_adapter_feasaug']

shots = [1, 2, 4, 8, 16]   # shots per class
num_exp = 1               # number of runs
```
Modify methods, shots, and num_exp as needed.

### Run
```python
python train_adapter.py
```

---

## Noise Embeddings

The Noise Adapter requires noise embeddings derived from raw single-photon measurement histograms, including:

- Mean Signal Photons Per Pixel (MSPPP)
- Signal-to-Background Ratio (SBR)

We provide scripts to compute and save these embeddings:

```python
python extract_noise_batch.py
```
### Note
Only depth images are released with the dataset. Raw photon histograms required for noise embedding computation should be obtained from the original dataset providers.

---
