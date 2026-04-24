## Trade-off Between Sparsity and Accuracy

The experiments show a clear relationship between the sparsity regularization strength (λ) and model behavior:

* Low λ (0.001): Weak sparsity pressure leads to moderate pruning (~33%) with stable accuracy.
* Medium λ (0.1): Increased sparsity (~50%) with minimal impact on accuracy.
* High λ (1.0): Strong sparsity (~81%) with a slight reduction in accuracy.

This indicates that the model successfully learns to eliminate redundant connections while preserving important ones.

## Gate Distribution Analysis

The gate value distribution shows a bimodal pattern, with values concentrated near 0 and 1. This confirms that the network is learning to:

* Suppress unimportant weights (gates ≈ 0)
* Retain critical connections (gates ≈ 1)

## Conclusion

The self-pruning mechanism is effective, enabling dynamic model compression during training. The choice of λ plays a crucial role in balancing model efficiency and predictive performance.
