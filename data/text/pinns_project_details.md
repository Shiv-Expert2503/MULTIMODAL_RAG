# An Experimental Analysis of Advanced Training Strategies for Physics-Informed Neural Networks
Solving the 1D Heat Equation with a Comparative Study of Optimizers and Adaptive Sampling


The main plot shows the comparative loss curves for all three models. The corresponding image file is named `pinns_comparative_total_loss.png`.

## Project Abstract

This repository presents a comprehensive investigation into solving the one-dimensional unsteady heat equation using Physics-Informed Neural Networks (PINNs). The project moves beyond a baseline implementation to conduct a rigorous comparative analysis of three distinct training methodologies: a standard **Adam optimizer**, a hybrid **Adam + Quassi-Newton Optimizer(LBFGS)** approach, and a **Residual-Based Adaptive Sampling (RBS)** strategy. The entire framework is built from scratch in PyTorch, culminating in a detailed analysis that quantifies the trade-offs between convergence speed, final accuracy, and training dynamics. The findings demonstrate that a hybrid optimization strategy yields the most accurate solution, providing valuable insights into practical PINN training for scientific machine learning (SciML) applications.

## 1. Problem Statement & Motivation

Traditional numerical methods for solving Partial Differential Equations (PDEs) often face challenges with grid generation, dimensionality, and complex geometries. Physics-Informed Neural Networks (PINNs) offer a mesh-free, data-driven alternative by embedding the PDE directly into the neural network's loss function.

However, a critical challenge in PINNs, especially for complex or high-dimensional problems, is the **static and often uniform distribution of collocation points**. If these points are not sufficiently dense in regions where the PDE is difficult to satisfy (e.g., near sharp gradients, shocks, or boundaries), the network may struggle to converge to an accurate solution efficiently. This project directly addresses this limitation by implementing adaptive sampling.

## 2. Mathematical Formulation

The project solves the 1D unsteady heat conduction equation:

$$ \frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2} $$

over a spatial domain $x \in [0, L]$ and time domain $t \in [0, T_{final}]$.

**Boundary Conditions (BCs):** Dirichlet boundary conditions at $x=0$ and $x=L$.

$$ u(0, t) = 0 $$
$$ u(L, t) = 0 $$

**Initial Condition (IC):**

$$ u(x, 0) = \sin(\pi x) $$

The analytical solution for this specific problem is:

$$ u(x, t) = e^{-\alpha \pi^2 t} \sin(\pi x) $$

The PINN formulation minimizes a composite loss function:

$$ L = L_{PDE} + L_{IC} + L_{BC} $$

Where:
* $L_{PDE}$: Enforces the PDE residual to be zero at collocation points.
  
$$ L_{PDE} = \frac{1}{N_{pde}} \sum_{i=1}^{N_{pde}} \left( \frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} \right)^2 $$

* $L_{IC}$: Enforces the initial condition at initial time points.
  
$$ L_{IC} = \frac{1}{N_{ic}} \sum_{i=1}^{N_{ic}} \left( u(x_i, 0) - u_{initial}(x_i) \right)^2 $$

* $L_{BC}$: Enforces the boundary conditions at spatial boundary points.
  
$$ L_{BC} = \frac{1}{N_{bc}} \sum_{i=1}^{N_{bc}} \left( u(0, t_i) - 0 \right)^2 + \frac{1}{N_{bc}} \sum_{i=1}^{N_{bc}} \left( u(L, t_i) - 0 \right)^2 $$

## 3. Network Architecture (PINN_1DHeat)

The neural network `PINN_1DHeat` is a standard multi-layer perceptron (MLP) designed for PINN applications:

* **Input Layer:** 2 neurons (for `x` and `t`).
* **Hidden Layers:** 4 hidden layers, each with 256 neurons.
* **Activation Function:** `Tanh` (Hyperbolic Tangent) activation function is used between hidden layers, known for its effectiveness in PINNs due to its smoothness and differentiability.
* **Weight Initialization:** Xavier Normal initialization (`torch.nn.init.xavier_normal_`) for weights and zero initialization for biases.
* **Output Layer:** 1 neuron (for the predicted solution `u`).

---

## 4. Experimental Design

To evaluate the most effective training strategy, a controlled experiment was designed to compare three distinct models. **Crucially, all three models started from the exact same initial weight distribution (`torch.manual_seed(42)`)** to ensure a fair and scientifically valid comparison.

| Model | Description | Training Strategy |
| :--- | :--- | :--- |
| **Model A** | The baseline, using a standard first-order optimizer. | **Adam Only** (15,000 epochs) |
| **Model B** | A hybrid approach to leverage both broad exploration and fine-tuning. | **Adam (10k epochs) + L-BFGS (5k steps)** |
| **Model C**| An advanced technique to focus computation on high-error regions. | **Adam with Residual-Based Adaptive Sampling** (15,000 epochs) |

The core network architecture was a Multi-Layer Perceptron (MLP) with 4 hidden layers of 256 neurons each and a `Tanh` activation function.

---

## 5. Core Methodology & Advancements
The core of this project is a comparative analysis of three training strategies built upon a consistent neural network architecture. This model serves as the baseline, trained for 15,000 epochs using the Adam optimizer. Adam, a first-order stochastic gradient method, is effective for efficiently navigating the high-dimensional parameter space in the initial stages of training.

### **5.1 Model A: Baseline (Adam Optimizer)**
This model serves as the baseline, trained for 15,000 epochs using the Adam optimizer. Adam, a first-order stochastic gradient method, is effective for efficiently navigating the high-dimensional parameter space in the initial stages of training.

### 5.2 **Model B: Hybrid Optimization (Adam + L-BFGS)**
This strategy employs a two-phase approach to leverage the strengths of different optimization classes:
1.  **Phase 1 (Adam):** The model is trained for 10,000 epochs with Adam to quickly find a favorable region in the loss landscape.
2.  **Phase 2 (L-BFGS):** The model is then fine-tuned for 5,000 steps using L-BFGS, a quasi-Newton method. By approximating the inverse Hessian matrix, L-BFGS can take more informed steps, often converging to a sharper and more precise minimum than is possible with first-order methods alone.

**Theory of L-BFGS Fine-tuning:**

L-BFGS is a **quasi-Newton method**, which belongs to the class of second-order optimization algorithms. Unlike first-order methods like Adam which only use gradient (first derivative) information, second-order methods utilize curvature (second derivative) information to take more direct and efficient steps towards a minimum.

* **Newton's Method:** The basis for these methods is Newton's optimization step, which finds the minimum of a quadratic approximation of the loss function $L$. The parameter update rule is:

$$ \theta_{k+1} = \theta_k - H_k^{-1} \nabla L(\theta_k) $$

where $\nabla L(\theta_k)$ is the gradient and $H_k^{-1}$ is the inverse of the **Hessian matrix** (the matrix of second-order partial derivatives). For deep neural networks with millions of parameters, computing, storing, and inverting the Hessian is computationally infeasible.

* **L-BFGS Approximation:** The L-BFGS algorithm cleverly circumvents this issue. Instead of computing the full inverse Hessian, it **approximates** $H_k^{-1}$ by storing a limited history (typically 10-20) of the most recent gradient updates and parameter changes. This provides a low-rank approximation of the curvature, allowing the optimizer to make more intelligent steps than gradient descent alone, without the prohibitive cost of the full Hessian.

* **Role in PINNs:** L-BFGS is exceptionally effective for the final fine-tuning stage of PINN training. Adam is adept at navigating the chaotic, non-convex loss landscape early on, while L-BFGS excels at rapidly converging to a sharp, precise minimum once it is in a well-behaved, convex-like region. This is why the hybrid strategy is so powerful.

  
### 5.3 **Model C: Residual-Based Adaptive Sampling (RBS)**
This model enhances the Adam training process by integrating an adaptive resampling mechanism to focus computational effort.

**Theory of Soft RBS (Importance Sampling via PDE Residual):**

Instead of using a fixed, uniformly random set of collocation points, adaptive sampling focuses computational effort on "hard spots" – regions where the network is making the largest errors (i.e., where the PDE residual is high).

* **Importance Metric:** The absolute magnitude of the PDE residual, $|R(x_i, t_i)|$, is used as the "importance" metric for each candidate point $(x_i, t_i)$. Points with higher residuals are more critical for the network to learn from.
* **Probability Distribution:** These residual magnitudes are transformed into sampling probabilities. For each candidate point, its probability $P_i$ of being selected for the next training batch is proportional to its residual raised to a positive power $p$:

$$ P_i \propto |R(x_i, t_i)|^p $$
  
  A `p_exponent` of 2 was used in this implementation, which means the probability is proportional to the *squared* residual, further emphasizing points with larger errors.
* **Normalization:** The probabilities are normalized so that their sum across all candidate points equals 1:

$$ P_i = \frac{|R(x_i, t_i)|^p + \epsilon}{\sum_{j=1}^{N_{cand}} (|R(x_j, t_j)|^p + \epsilon)} $$
  
(A small epsilon is added to prevent zero probabilities and numerical instability.)
* **Sampling:** A fixed number of new collocation points (e.g., 15,000) are then sampled from a large pool of candidate points (e.g., 50,000-100,000 uniformly sampled points) using `numpy.random.choice()` based on these calculated probabilities. This process allows points with higher $P_i$ values to be selected more frequently.
* **Iterative Refinement:** This adaptive sampling step is performed periodically (e.g., every 500 Adam epochs), replacing the current set of collocation points with the newly sampled, "harder" set. This drives the network to iteratively improve its fit in regions where it previously struggled.

## 6. Challenges Encountered & Debugging

Implementing the adaptive sampling mechanism, particularly its integration within the PyTorch computational graph, presented specific challenges:

* **`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`**: This error occurred during the PDE residual calculation for adaptive sampling. It arose because the `model(input_coords)` operation was unintentionally being performed within a `torch.no_grad()` context, preventing the computational graph necessary for derivative calculation.
    * **Resolution:** The `with torch.no_grad():` block was correctly removed from around the call to `calculate_pde_residual` during the adaptive sampling phase. This allowed the necessary graph for derivative computations (u_x, u_t, u_xx) to be built.
* **`RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.`**: After resolving the previous error, the `candidate_residuals_tensor` retained its gradient tracking history, preventing direct conversion to NumPy for probability calculation.
    * **Resolution:** Explicitly calling `.detach()` on `candidate_residuals_tensor` before converting it to a NumPy array (e.g., `candidate_residuals_tensor.detach().cpu().numpy()`) correctly removed it from the computational graph, allowing the conversion.

These debugging steps were crucial for ensuring the adaptive sampling mechanism functioned correctly and efficiently within the PyTorch framework.

## 7. Technical Stack & Tools

* **Programming Language:** Python
* **Deep Learning Framework:** PyTorch
* **Numerical Operations:** NumPy
* **Visualization:** Matplotlib
* **Computational Environment:** Kaggle / Google Colab (leveraging GPU for acceleration)

## 8. Results and Analysis

The experiment yielded clear and insightful results, demonstrating the trade-offs between different advanced training techniques.

### 8.1 Quantitative Results

The hybrid Adam + L-BFGS approach was the definitive winner in terms of final accuracy.

| Metric | Model A (Adam Only) | Model B (Adam + L-BFGS) | Model C (Adaptive) |
| :--- | :--- | :--- | :--- |
| **Final MSE vs. True Solution** | 1.55e-06 | **3.11e-07** | 1.96e-06 |
| **Relative L2 Error** | 1.86e-03 | **8.32e-04** | 2.09e-03 |
| **Total Training Time (s)** | 891.7 | 1020.0 | 900.7 |

### 8.2 Qualitative Analysis (Visuals)

#### **Training Dynamics: Comparative Loss Curves**

The most insightful result comes from comparing the training loss curves for all three models.

The following plots illustrate the training dynamics and learning behavior of each model.
- The PDE loss comparison is shown in the image file named `pinns_comparative_pde_loss.png`.
- The Initial Condition (IC) loss comparison is in the image file `pinns_comparative_ic_loss.png`.
- The Boundary Condition (BC) loss comparison is in the image file `pinns_comparative_bc_loss.png`.
- The total combined loss for all models is visualized in the image file `pinns_comparative_total_loss.png`.

* **Insight:** This plot clearly shows the learning behavior of each model. Model A (blue) shows steady convergence. Model C (green) shows noisy but aggressive learning due to the periodic resampling. The most dramatic event is the sharp, vertical drop in loss for Model B (orange) at epoch 10,000, demonstrating the power of the L-BFGS optimizer to rapidly find a superior solution.

#### **Final Accuracy: Comparison of Absolute Error Surfaces**


This plot provides a visual confirmation of the final accuracy of each model across the entire problem domain by comparing their absolute error surfaces. The corresponding image file is named `pinns_absolute_error_surfaces.png`.

* **Insight:** The error surface for Model B (center) is visibly "flatter" and has the lowest overall magnitude, confirming the quantitative results that it was the most accurate model.

#### **Solution Quality at a Time Slice (t=0.25)**

These plots offer a high-resolution snapshot of each model's final prediction against the ground truth at a specific time slice (t=0.25).
- The solution for Model A (Adam Only) is shown in the image file `pinns_solution_slice_adam.png`.
- The solution for Model B (Adam + L-BFGS) is shown in the image file `pinns_solution_slice_lbfgs.png`.
- The solution for Model C (Adaptive Sampling) is shown in the image file `pinns_solution_slice_adaptive.png`.
- A combined plot comparing all solutions against the ground truth is in the image file `pinns_solution_slice_combined.png`.

* **Insight:** The left subplot shows that all three models successfully captured the overall shape of the solution. The right subplot, however, reveals the nuances in their accuracy, with the error for Model B being consistently lower than the others.


## 9. Overall Conclusion and Future Work

This project successfully implemented and rigorously compared three distinct training strategies for a PINN. The results demonstrate that a **hybrid `Adam + L-BFGS` approach was the most effective strategy**, achieving the lowest final error by a significant margin.

The key takeaway is the power of combining optimizers. A first-order method like Adam is excellent for quickly navigating the broad loss landscape, while a second-order method like L-BFGS is exceptionally effective at fine-tuning and converging to a sharp, precise minimum. This two-phase approach represents a robust and highly effective strategy for training Physics-Informed Neural Networks.
 Future directions include:

* **Extension to 1D Wave Equation:** Applying the same PINN and adaptive sampling framework to the 1D wave equation, which involves different derivative structures.
* **Burgers' Equation:** Solving the non-linear Burgers' equation, a significant step due to its non-linearity and potential for shock formation, where adaptive sampling is even more critical.
* **Navier-Stokes Equations:** Addressing the complex Navier-Stokes equations for fluid dynamics, representing a major challenge in scientific machine learning.
* **Schrödinger Equation:** Exploring quantum mechanics problems with PINNs.
* **Higher Dimensionality:** Extending the methodology to 2D and 3D PDE problems, which will necessitate careful consideration of computational resources and sampling strategies.

The complete experimental code is contained within the main Jupyter Notebook named pinns-comparison.ipynb.
---