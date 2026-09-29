Deep Learning CNN Framework
A convolutional neural network framework built entirely from scratch in Python using NumPy. The project implements the core components of a modern CNN—including forward propagation, 
backpropagation, optimization, normalization, and vectorized computation—without relying on deep learning libraries such as PyTorch or TensorFlow.

The framework was designed with a focus on understanding the underlying mathematics and computational mechanics of deep learning, while also demonstrating how vectorization can dramatically improve performance.

Features

Object-oriented CNN architecture built entirely with NumPy

Manually derived forward and backward propagation

Convolutional layers with vectorized im2col / col2im implementations

Vectorized max pooling

Batch normalization

ReLU activation

Fully connected (dense) layers

Softmax output layer

Adam optimizer

Numerical gradient checking

Naive vs. vectorized performance benchmarking

End-to-end training on a subset of CIFAR-10

Architecture
The framework currently supports the following layers:
Input -> Convolution -> Batch Normalization -> ReLU -> Max Pooling -> Flatten -> Dense -> Softmax -> Class Probabilities

Each layer implements its own forward and backward passes, allowing networks to be composed and trained using a consistent object-oriented interface.

