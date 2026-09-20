# PA1: Beyond IID – Advanced Topics in Machine Learning

**EE-5102/CS-6304 – Fall 2026**

## Overview

This repository contains experiments examining four aspects of learning beyond the IID, closed-set setting:

| Task | Topic | Dataset | Models |
|------|-------|---------|--------|
| 1 | Inductive Biases & Representations | STL-10 | ResNet-50, ViT-B/16, CLIP ViT-B-32 (frozen) |
| 2 | Unsupervised Domain Adaptation | PACS → Sketch | ResNet-18 (fine-tuned) |
| 3 | Domain Generalization | PACS (Sketch unseen) | ResNet-18 (fine-tuned) |
| 4 | Open-Set Recognition | CIFAR-10 / CIFAR-100 | CIFAR-ResNet-18 |

## Repository Structure

```
pa1-beyond-iid/
├── shared/                       # Shared PACS protocol (Tasks 2 & 3)
│   ├── src/
│   └── notebooks/
├── task1_inductive_biases/       # Inductive biases & feature representations
│   ├── src/
│   └── notebooks/
├── task2_domain_adaptation/      # Unsupervised domain adaptation
│   ├── src/
│   └── notebooks/
├── task3_domain_generalization/  # Domain generalization
│   ├── src/
│   └── notebooks/
├── task4_open_set/               # Open-set recognition
│   ├── src/
│   └── notebooks/
└── report/
```

## Setup

```bash
pip install -r requirements.txt
```

## Global Seed: **6304**

## Key Constraints

- Task 2: Target labels never used during training/model selection
- Task 3: Sketch images never loaded until final evaluation
- Task 4: CIFAR-100 images never used during training
- Tasks 2 & 3: BatchNorm running stats frozen at ImageNet values

## GitHub Repository

<!-- Insert link here -->
