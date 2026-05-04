# Project: Constraint-Driven Training for Modular Edge Models

## Description
This project focuses on the intersection of loss-function engineering and model architecture. Instead of taking a pre-trained model and compressing it, this project aims to "sculpt" a neural network from scratch to be naturally sparse and modular. By augmenting the standard cross-entropy loss with custom penalties (L1 regularization on weights and activation sparsity constraints), we force the model to develop specialized "circuits" or modules during the training phase.

## Scientific Objective
To determine if training with sparsity-inducing constraints produces a network that is inherently more interpretable and robust than a standard, densely trained model of similar architecture.

## Mathematical Formulation
The objective function to minimize during training is defined as:

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{task}} + \lambda \sum_{i} \lVert \omega_i \rVert_1 + \gamma \mathcal{L}_{\mathrm{activation}}
$$

Where:
* $\mathcal{L}_{\mathrm{task}}$: Cross-Entropy loss for classification.
* $\lambda \lVert \omega_i \rVert_1$: Weight penalty (Lasso) to enforce parameter sparsity.
* $\gamma \mathcal{L}_{\mathrm{activation}}$: Activation penalty to enforce neuron-level sparsity (e.g., L1 norm of layer activations).

---

## Technical Roadmap

### Phase 1: Architecture Design (Week 1)
1.  **Framework:** Initialize a deep architecture (e.g., ResNet-18 or a custom VGG-style block) without pre-trained weights.
2.  **Initialization:** Employ Kaiming He initialization to ensure stable gradients during the start of training.
3.  **Modularity Strategy:** Identify specific convolutional blocks (e.g., middle-depth layers) to monitor for the emergence of "modules."

### Phase 2: Custom Training Loop Implementation (Week 2)
1.  **Loss Class:** Develop a `SparseLoss` class in PyTorch that integrates weight and activation penalties.
2.  **Hyperparameter Optimization:** Perform a grid search on $\lambda$ and $\gamma$.
    * Monitor the tradeoff between classification accuracy and sparsity ratio.
    * Maintain logs for both loss components.
3.  **Training:** Execute training on the CIFAR-10 or Tiny-ImageNet dataset. Ensure the model does not collapse to zero (trivial) accuracy.

### Phase 3: Verification & Ablation (Week 3)
1.  **Weight Visualization:** Generate heatmaps of weight matrices post-training. Identify the formation of "islands" of non-zero weights.
2.  **Ablation Study:** Perform "pruning-on-the-fly." Mask out weights that are near zero (below a threshold $\epsilon$) and evaluate the impact on validation accuracy.
3.  **Comparison:** Compare the sparse model against a standard, unconstrained model with the same architecture to determine if the constrained model learned "easier" features.

### Phase 4: Reporting & Synthesis (Week 4)
1.  **Curve Analysis:** Plot $\mathcal{L}_{\mathrm{total}}$ versus training iterations.
2.  **Interpretation:** Provide qualitative examples of what the "modules" learned (e.g., visualization of class-specific filters).
3.  **Conclusion:** Synthesize findings on whether "architectural sculpting" via loss functions is a viable path for edge-model design.

---

## Technical Requirements
* **Language:** Python 3.x
* **Libraries:** `torch`, `torchvision`, `numpy`, `matplotlib`, `scikit-learn`
* **Hardware:** A GPU with at least 8GB VRAM is recommended for efficient training of the custom loss function.

## Key Reading
* Frankle, J., & Carbin, M. (2019). *The Lottery Ticket Hypothesis: Finding, Training, Pruning, and Retraining Sparse Neural Networks.* ICLR.
* Andreas, J., et al. (2017). *Learning to Compose Neural Networks for Question Answering.* NAACL.
