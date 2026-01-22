# Small CNN Experiments (TensorFlow / Keras)

This repository contains my experimentations with small convolutional neural networks (CNNs) built using **TensorFlow and Keras**, for academic purposes. It serves as a lightweight sandbox for exploring on simple CNN architectures.

The experiments primarily focus on:
- Image classification
- Coarse grid–based object detection

The models are intentionally kept small and modular to support rapid iteration and clearer reasoning about architectural design choices.

This repository is intended for learning and research purposes only. The code is experimental, and is not designed for production use.

### Features

- **Data Pipeline**
  - Automatic image resizing based on the model’s expected input shape
  - Dataset shuffling and batching
  - Support for both classification and coarse grid–based detection datasets

- **Data Augmentation**
  - Random image augmentations to reduce overfitting
  - Augmentations applied in-memory during training using TensorFlow/Keras utilities

- **Metrics & Scoring**
  - Computation of common metrics such as:
    - Precision
    - Recall
    - F1-score
  - For image classification and coarse grid-based detection

- **Object Detection Visualization**
  - Visualization of grid-based object detection outputs
  - Bounding box and prediction plotting using Matplotlib

- **Hyperparameter Optimization**
  - Automated hyperparameter tuning with **Optuna**
