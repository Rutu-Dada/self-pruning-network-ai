# Self-Pruning Neural Network

This project implements a neural network that dynamically prunes its own weights during training using learnable gating mechanisms and sparsity regularization.

Unlike traditional post-training pruning methods, this approach integrates pruning directly into the learning process, allowing the model to automatically identify and remove redundant connections.

## Key Result

Achieves over **80% sparsity** with minimal drop in accuracy on CIFAR-10.
