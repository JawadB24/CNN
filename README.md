# Deep Learning CNN Framework

A convolutional neural network framework built entirely from scratch in **Python using NumPy**, with manually implemented forward propagation, backpropagation, optimization, normalization, and vectorized operations.

This project focuses on understanding the underlying mathematics and computational mechanics of CNNs without relying on high-level deep learning frameworks such as PyTorch or TensorFlow.

## Features

- **Convolutional Layer**
- **Batch Normalization**
- **ReLU Activation**
- **Max Pooling**
- **Dense / Fully Connected Layer**
- **Softmax Output Layer**
- **Adam Optimizer**
- **Manual Forward & Backward Propagation**
- **Vectorized Convolution using `im2col` / `col2im`**
- **Vectorized Max Pooling**
- **Two-Sided Numerical Gradient Checking**
- **Naive vs. Vectorized Performance Benchmarking**
- **End-to-End CIFAR-10 Training**

---

## Architecture

The framework supports composing layers into a CNN pipeline such as:

```text
Input
  │
  ▼
Convolution
  │
  ▼
Batch Normalization
  │
  ▼
ReLU
  │
  ▼
Max Pooling
  │
  ▼
Dense
  │
  ▼
Softmax
  │
  ▼
Class Probabilities
