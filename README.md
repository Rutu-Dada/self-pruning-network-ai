# Self-Pruning Neural Network

This project implements a neural network that dynamically prunes its own weights during training using learnable gating mechanisms and sparsity regularization.

Unlike traditional post-training pruning methods, this approach integrates pruning directly into the learning process, allowing the model to automatically identify and remove redundant connections.

## Key Result

Achieves over **80% sparsity** with minimal drop in accuracy on CIFAR-10.



## Results

| Lambda | Test Accuracy (%) | Sparsity (%) |
| ------ | ----------------- | ------------ |
| 0.001  | 54.15             | 33.59        |
| 0.1    | 54.24             | 50.25        |
| 1.0    | 53.28             | 81.03        |

## Observations

* Increasing λ leads to significantly higher sparsity, indicating successful pruning of less important weights.
* The model maintains relatively stable accuracy even at high sparsity levels, suggesting that many weights are redundant.
* At λ = 1.0, the model achieves over 80% sparsity with only a minor drop in accuracy.

This demonstrates an effective trade-off between model compactness and performance.
