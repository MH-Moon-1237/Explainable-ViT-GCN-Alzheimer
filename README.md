# Explainable ViT–GCN for Alzheimer's Disease Classification

### Research Title
**Explainable ViT–GCN Hybrid Architecture for MRI-Based Multi-Stage Classification of Alzheimer’s Disease Across Multiple Public Datasets**

## Overview

This repository provides the Python implementation of an explainable hybrid Vision Transformer–Graph Convolutional Network (ViT–GCN) framework for MRI-based Alzheimer's disease classification.

The proposed framework combines the global feature extraction capabilities of **ViT-B/16** with the relational learning capabilities of a **Graph Convolutional Network (GCN)**. The framework is evaluated across three publicly available MRI datasets, covering cognitive impairment, dementia stages, and neurodegenerative disease classification.

## Proposed Methodology

The framework consists of six major components:

1. **Data Preprocessing:** MRI images are resized to 224 × 224 pixels and normalized using ImageNet statistics. Training-only augmentation and weighted random sampling are incorporated.
2. **Feature Extraction:** A pretrained ViT-B/16 backbone extracts 768-dimensional image representations.
3. **Graph Construction:** A cosine similarity-based k-nearest-neighbor graph (k = 5) is constructed using training embeddings.
4. **GCN Classification:** A two-layer GCN with 256 hidden units performs relational feature learning and multiclass classification.
5. **Performance Evaluation:** Accuracy, weighted precision, recall, F1-score, Cohen's Kappa, ROC-AUC, and confusion matrices are used for evaluation.
6. **Explainable AI:** LIME, Grad-CAM, Grad-CAM++, and ViT attention maps provide visual interpretations of model predictions.

## Datasets

Three publicly available MRI datasets are considered:

| Dataset | MRI Images | Classification Task |
|---|---:|---|
| Dataset 1 | 11,519 | Four-class cognitive impairment classification |
| Dataset 2 | 6,400 | Four-class Alzheimer's dementia classification |
| Dataset 3 | 7,756 | Three-class classification: AD, CN, and PD |

For Dataset 2, only the original MRI images are considered, excluding separately provided pre-augmented images.

## Repository Structure

| Python File | Description |
|---|---|
| `data_preparation.py` | Dataset preprocessing, augmentation, splitting, and sampling |
| `vit_gcn.py` | Proposed ViT–GCN architecture and graph construction |
| `classification.py` | Model training and baseline comparisons |
| `cross_validation.py` | Stratified 10-fold cross-validation |
| `evaluation.py` | Performance evaluation and visualization |
| `xai.py` | LIME, Grad-CAM, Grad-CAM++, and attention maps |
| `requirements.txt` | Software dependencies |

## Installation

The implementation uses Python, PyTorch, torchvision, scikit-learn, and supporting scientific computing libraries.

Required dependencies can be installed using:

`pip install -r requirements.txt`

## Comparative Models

The proposed ViT–GCN framework is compared with seven baseline models: VGG16, ResNet18, ResNet50, EfficientNetB3, DenseNet-121, InceptionV3, and standalone ViT-B/16.

Stratified 10-fold cross-validation is implemented for ViT-B/16 and the proposed ViT–GCN, while CNN baselines are evaluated using fixed data partitions.

## Reproducibility

This repository provides the source code for the proposed framework, including data preprocessing, model training, cross-validation, performance evaluation, and explainable AI techniques.

The implementation uses image-level data partitioning. Patient-wise generalization has not been established, and reproducing the exact reported results requires the original experimental partitions and trained model configurations.

## Research Purpose

This repository is made available to support academic research, methodological transparency, and reproducibility of the proposed ViT–GCN framework for MRI-based classification.
