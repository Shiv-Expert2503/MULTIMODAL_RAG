# 🔥 An Experimental Analysis of Advanced Training Strategies for Physics-Informed Neural Networks  

---

## 🏷️ Metadata (embedding anchors + keywords)  
* **Project Name:** An Experimental Analysis of Advanced Training Strategies for Physics-Informed Neural Networks (PINNs)
* **Primary PDE Solved:** 1D Unsteady Heat Equation
* **Core Investigation:** A comparative study of three training strategies: Adam optimizer, a hybrid Adam + L-BFGS approach, and Residual-Based Adaptive Sampling (RBS).
* **Key Finding:** The hybrid Adam + L-BFGS optimization strategy yielded the most accurate solution.
* **Framework:** PyTorch (built from scratch)
* **Keywords:** Physics-Informed Neural Networks, PINNs, Scientific Machine Learning, SciML, Partial Differential Equations, PDE, Heat Equation, Adam, L-BFGS, Adaptive Sampling.


Hidden metadata (embedding only):  
- "PINN heat conduction project"  
- "Shivansh PINNs research"  
- "SciML optimizer comparison"  
- "Hybrid Adam L-BFGS PINN training"  
- "Adaptive collocation sampling PINNs"  

---

## 📄 Project Abstract (Atomic Chunk)  
- This project solves the **1D unsteady heat equation** using Physics-Informed Neural Networks (PINNs).  
- Three strategies compared: **Adam only**, **Adam + L-BFGS hybrid**, **Residual-Based Adaptive Sampling**.  
- Entire framework written from scratch in **PyTorch**.  
- Analysis focuses on **convergence speed**, **final accuracy**, and **training dynamics**.  
- Key insight: **Hybrid Adam + L-BFGS strategy gives the best accuracy**.  

---

## 🎯 Problem Statement & Motivation  
- Classical PDE solvers need grids and struggle with high-dimensional problems.  
- PINNs remove grids by embedding PDE loss directly into neural networks.  
- Main issue: **uniform collocation points are inefficient**.  
- Adaptive sampling solves this by focusing training on high-error regions.  

---

## 📐 Mathematical Formulation  
- PDE: Heat conduction equation.  
$$ \frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2} $$  

- Domain: $x \in [0, L], t \in [0, T_{final}]$.  
- Boundary conditions: $u(0,t)=0$, $u(L,t)=0$.  
- Initial condition: $u(x,0)=\sin(\pi x)$.  
- Analytical solution: $u(x,t)=e^{-\alpha\pi^2 t}\sin(\pi x)$.  

**Composite loss function:**  
$$ L = L_{PDE} + L_{IC} + L_{BC} $$  

Each term enforces PDE, IC, and BC separately.  

---

## 🧠 Network Architecture (PINN_1DHeat)  
- Input: 2 neurons → `(x,t)`.  
- Hidden layers: 4 layers × 256 neurons.  
- Activation: **Tanh**.  
- Initialization: Xavier Normal.  
- Output: 1 neuron → predicted solution `u(x,t)`.  

---

## ⚗️ Experimental Design  
- All models trained from the **same initialization** (seed=42).  
- Three training strategies:  

| Model | Strategy | Description |  
|-------|----------|-------------|  
| A     | Adam     | Baseline, 15k epochs |  
| B     | Adam + L-BFGS | 10k Adam + 5k L-BFGS |  
| C     | Adam + Adaptive Sampling | 15k epochs, resample collocation points |  

---

## 🔬 Core Methodology  

### Model A: Adam Only  
- First-order optimizer.  
- Good for initial exploration.  

### Model B: Hybrid Adam + L-BFGS  
- Phase 1: Adam (10k epochs).  
- Phase 2: L-BFGS (5k steps).  
- L-BFGS approximates curvature using limited memory (quasi-Newton).  
- Combines speed of Adam with precision of second-order optimization.  

### Model C: Residual-Based Adaptive Sampling (RBS)  
- Points sampled based on **PDE residual magnitude**.  
- High-residual regions get sampled more often.  
- Probability distribution:  
$$ P_i \propto |R(x_i,t_i)|^p $$  
- Iteratively replaces collocation points every 500 epochs.  

---

## 🛠️ Challenges & Debugging  
- **Error: Gradients missing** → caused by accidental `torch.no_grad()`.  
  - ✅ Fixed by removing no_grad during residual calculation.  
- **Error: Tensor → NumPy conversion** → fixed by `.detach().cpu().numpy()`.  

---

## ⚙️ Tech Stack  
- Language: Python  
- DL Framework: PyTorch  
- Math Ops: NumPy  
- Visualization: Matplotlib  
- Compute: Kaggle GPU, Colab GPU  

---

## 📊 Results & Analysis  

### Quantitative Metrics  
| Metric | Adam | Adam + L-BFGS | Adaptive |  
|--------|------|---------------|----------|  
| Final MSE | 1.55e-06 | **3.11e-07** | 1.96e-06 |  
| Relative L2 | 1.86e-03 | **8.32e-04** | 2.09e-03 |  
| Training Time (s) | 891.7 | 1020.0 | 900.7 |  

➡️ **Hybrid Adam + L-BFGS = best accuracy**.  

---

### Visual Comparisons  

- **Loss Curves** → `pinns_comparative_total_loss.png`  
  _Shows sharp drop at epoch 10k for L-BFGS._  

- **PDE Loss Plot** → `pinns_comparative_pde_loss.png`  
  _Model B curve flattens fastest._  

- **IC Loss Plot** → `pinns_comparative_ic_loss.png`  
  _Model A slower than B._  

- **BC Loss Plot** → `pinns_comparative_bc_loss.png`  

- **Error Surfaces** → `pinns_absolute_error_surfaces.png`  
  _Model B = flattest error surface._  

- **Solution Slices at t=0.25** →  
  - `pinns_solution_slice_adam.png`  
  - `pinns_solution_slice_lbfgs.png`  
  - `pinns_solution_slice_adaptive.png`  
  - `pinns_solution_slice_combined.png`  

---

## 🧾 FAQ (Redundancy Anchors)  

**Q: What is the project about?**  
A: Solving 1D heat equation with Physics-Informed Neural Networks (PINNs).  

**Q: Which optimizer worked best?**  
A: Hybrid Adam + L-BFGS gave the lowest error.  

**Q: Why adaptive sampling?**  
A: To focus collocation points where PDE residuals are highest.  

**Q: Which framework was used?**  
A: PyTorch, with NumPy and Matplotlib.  

**Q: Where was it trained?**  
A: Kaggle GPU and Colab GPU environments.  

---

## 🔗 Synonyms & Cross-links  
- “Adam + L-BFGS” ↔ “Hybrid optimization” ↔ “Two-phase training”.  
- “Adaptive Sampling” ↔ “Residual-based resampling” ↔ “RBS method”.  
- “Heat Equation” ↔ “1D conduction” ↔ “diffusion PDE”.  
- “PINNs project” ↔ “Physics-Informed Neural Networks project”.  

Cross-links:  
- From **Problem Statement** → to **Mathematical Formulation**.  
- From **Methodology** → to **Results**.  
- From **FAQ** → back to **Core Methodology**.  

---

## 🏁 Conclusion & Future Work  
- ✅ Best method = **Hybrid Adam + L-BFGS**.  
- Adaptive sampling improves exploration but was less stable.  
- Future directions:  
  - Extend to 1D Wave, Burgers, Navier-Stokes, Schrödinger.  
  - Explore 2D/3D PDEs.  
  - Test other sampling strategies.  

---

## 📊 Recap Table  

| Aspect | Best Performer | Notes |  
|--------|----------------|-------|  
| Final Accuracy | Adam + L-BFGS | Sharpest convergence |  
| Training Speed | Adam | Fast but less accurate |  
| Adaptive Behavior | RBS | Focuses on high-error regions |  

---

📂 **Notebook**: `pinns-comparison.ipynb`  
