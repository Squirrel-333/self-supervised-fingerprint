# Self-Supervised Learning for Robust Website Fingerprinting (DLWF)

This project investigates the robustness of self-supervised learning (SSL) models for website fingerprinting under realistic network perturbations, including time shifts and data collection noise.

---

## Motivation

Existing website fingerprinting methods often assume clean and perfectly aligned network traces.  
However, in real-world scenarios, network traffic is subject to:

- Time misalignment (temporal shift)
- Collection noise and measurement errors
- Variability across sessions and environments

These factors can significantly degrade model performance and expose weaknesses in learned representations.

---

## Research Objectives

This project aims to answer the following questions:

- How robust are SSL-based representations to temporal shifts in network traffic?
- How do data collection errors affect fingerprinting performance?
- Can we design domain-specific augmentations to improve robustness?

---

## Methodology

### 1. Self-Supervised Learning Adaptation
We adapt multiple SSL frameworks to the DLWF dataset:

- TS2Vec  
- SimCLR  
- CPC  
- MAE  
- BYOL  

These models learn representations from raw network traces without labels.

---

### 2. Robustness Evaluation

We explicitly evaluate model robustness under:

- Time Shift Perturbations  
  - Random temporal offsets applied to traffic sequences  

- Collection Noise  
  - Packet loss simulation  
  - Timing jitter  
  - Partial observation  

---

### 3. Domain-Specific Data Augmentation

To improve robustness, we design augmentations tailored to network traffic:

- Temporal cropping and shifting  
- Packet-level masking  
- Noise injection in timing/features  
- Sequence scaling and distortion  

These augmentations are used within SSL pipelines to enforce invariance.

---

## Key Contributions

- Introduce robustness evaluation under realistic network perturbations for DLWF  
- Systematically analyze the impact of time shift and collection noise on SSL representations  
- Propose network-aware data augmentation strategies for self-supervised learning  
- Provide a unified framework for adapting SSL methods to website fingerprinting  

---

## Repository Structure

```
.
├── data/                # dataset and preprocessing
├── models/              # SSL model implementations
├── configs/             # experiment configurations
├── notebooks/           # exploratory analysis
├── progress/            # weekly research logs
└── README.md
```

---

## Status

🚧 Ongoing research project  
📄 Paper in preparation
