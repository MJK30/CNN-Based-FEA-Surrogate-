# CNN-Based-FEA-Surrogate-


This project implements a convolutional neural network surrogate with uncertainty quantification for rapid prediction of maximum Von Mises stress in 2D plate geometries. The workflow covers data generation, model development, training, uncertainty estimation, and active‑learning readiness.

---

## Table of Contents

1. Prerequisites
2. Data Generation
3. Data Ingestion & Preparation
4. Model Overview
5. Training Process
6. Uncertainty Quantification
7. Active‑Learning Next Steps


---

## 1. Prerequisites

* SimuStruct dataset (mesh CSV files) available locally


---

## 2. Data Generation

* **Input:** SimuStruct mesh files containing node coordinates, mesh topology, and per‑node stress values
* **Process:** Render each mesh’s stress field into a uniform image size
* **Output:** Directory of stress images and a CSV mapping image filenames to maximum stress values

---

## 3. Data Ingestion & Preparation

* **Normalization:** Scale maximum stress values into a 0–1 range
* **Splitting:** Divide data into training, validation, and test subsets
* **Augmentation:** Apply geometric transforms to training images for robustness
* **Loading:** Use efficient data loaders for batch processing during training and evaluation

---

## 4. Model Overview

* **Architecture:** Deep CNN with multiple convolutional blocks, batch normalization, and dropout
* **Regularization:** Spatial dropout in convolutional layers and dropout in fully connected layers
* **Output:** Single scalar prediction per input image representing normalized maximum stress

---

## 5. Training Process

* **Loss Function:** Huber loss to balance sensitivity and robustness to outliers
* **Optimizer:** Adam with a moderate learning rate for stable convergence
* **Epochs & Monitoring:** Train for a fixed number of epochs, tracking both training and validation losses to detect overfitting
* **Checkpointing:** Save model weights at completion for later inference

---

## 6. Uncertainty Quantification

* **Method:** Monte Carlo Dropout at inference time
* **Procedure:** Keep dropout active, perform multiple stochastic forward passes, and compute per‑sample mean and variance of predictions
* **Interpretation:** Mean is the surrogate’s best estimate; variance quantifies model confidence

---

## 7. Active‑Learning Next Steps

* Use predictive variance to select the most informative samples for additional high‑fidelity FEA
* Expand or update the dataset iteratively based on model uncertainty
* Retrain the surrogate regularly to incorporate new labeled data

---


