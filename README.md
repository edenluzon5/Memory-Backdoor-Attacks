# 🧠 Memory Backdoor Attacks on Neural Networks

Official repository for the paper  
**Memory Backdoor Attacks on Neural Networks** (NDSS 2026)

This repository contains the code and experiment notebooks used to demonstrate
**memory backdoor attacks** — a novel training-time backdoor that enables
systematic, deterministic extraction of training data from neural networks,
including image classifiers, segmentation models, and large language models.

---

## 🔍 Overview

<p align="center">
  <img src="assets/fig1_overview.png" width="750">
</p>

We introduce **memory backdoors**, a new class of backdoor attacks in which a
model is trained to *memorize and later reconstruct training samples* when
queried with a structured index-based trigger.

Unlike prior data extraction or memorization attacks, memory backdoors provide:
- deterministic extraction (no hallucination),
- explicit control over *which* samples are memorized,
- guarantees of authenticity,
- negligible impact on the model’s primary task.

The attack is particularly powerful in **federated learning (FL)** settings,
where a malicious server can tamper with the training procedure and later
extract private client data from trained local models.

---

## 🧠 Introduction

Federated learning is commonly assumed to provide privacy by keeping training
data local to clients. However, models trained under FL can still leak sensitive
information—especially when the central server is malicious or compromised.

In this work, we show that a malicious server can implant a **memory backdoor**
during training, causing models to secretly memorize selected training samples.
At inference time, the server can systematically extract these samples using
pattern-based index triggers, without degrading model utility or alerting
participants.

We demonstrate memory backdoors across:
- image classification models,
- image segmentation models,
- large language models (LLMs).

---

## ⚙️ Methodology

<p align="center">
  <img src="assets/fig2_activation.png" width="750">
</p>

### Memory Backdoor Mechanism

The attack augments standard training with a **memorization loss** that activates
only when a specific trigger is present. When triggered, the model reconstructs
memorized training samples instead of performing its primary task.

Key components:
- **Structured index space** that identifies samples, patches, and channels
- **Pattern-based triggers** compatible across architectures
- **Patch-based reconstruction** for models with limited output dimensionality
- **Deterministic extraction** via exhaustive index iteration

For image classifiers, full images are reconstructed patch-by-patch.
For segmentation models, full images can be reconstructed directly.
For LLMs, training conversations can be extracted verbatim.

---

## 📊 Datasets & Models

We evaluate memory backdoors across multiple datasets and architectures.

### Vision — Classification

- **MNIST** (1 × 28 × 28)
  - CNN
  - FCN

- **CIFAR-100** (3 × 32 × 32)
  - CNN
  - VGG16
  - Vision Transformer (ViT)

- **VGGFace2** (3 × 120 × 120)
  - Vision Transformer (ViT)

### Vision — Segmentation

- **Brain Tumor MRI Segmentation Dataset**
  - Vision Transformer–based segmentation model

### Language Models

- Instruction-tuned and conversational **LLMs**
  - memory backdoors enable extraction of training dialogues

### Evaluation Metrics

- **Utility**:
  - Classification Accuracy (ACC)
  - Dice coefficient (segmentation)

- **Memorization Quality**:
  - Structural Similarity Index (SSIM)
  - Mean Squared Error (MSE)
  - Feature Accuracy

---

## 🧪 Code Structure


