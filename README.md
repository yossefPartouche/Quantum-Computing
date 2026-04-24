# AI Quantum Observer: Trajectory Reconstruction via Online Learning

### The Challenge 

An Original research (Berkeley) successfully used LSTMs to reconstruct qunatum dynamics but relied on a massive static dataset **1.5 million traces**. While this provides a stable baseline, it's misleading. The **uniqueness of the noise realisation** in the Hilbert space, so the evolution of the (our) qubit's density matrix $\rho$ is governed by: 

$$d\rho_t = -i [H, \rho_t]dt + \gamma D[\sigma_z]\rho_tdt + \sqrt{\eta \gamma}\mathbf{H}[\sigma_z]\rho_td\mathbf{W}_t$$

Where: 

- **The Drift:** $-i [H, \rho_t]dt$ Rabi Oscillations
- **The Diffusion (Decoherence):** $\gamma D[\sigma_z]\rho_tdt$ Gradual Loss of quantum information 
- **Stochastic Kick:** $\sqrt{\eta \gamma} \mathbf{H}[\sigma_z]\rho_td\mathbf{W}_t$: 
  - Where $d\mathbf{W}_t$ represents an ever so small increment of a **Wiener Process**, which is drawn from a Guassian distribution with $\mu = 0$ and $\sigma^2 = dt$, thus the probability of obtaining the exact same sequence of $d\mathbf{W}_t$ values over a $4 \mu s$ window is effectively zero.


In real-world quantum hardware,  By pivoting to an online learning method, we elilminated the risk of the model 'memorizing' noise pattern and forced it to adapt to the live unpredictable nature of quantum evolution.

### Online Learning 
This project implements an **Online, RL-style Training Loop**

- **Dynamic Generation**: Measurement records ($V_{t}$) are generated during the training, preventing the model from "memorizing" specific noise patterns and forcing it to learn universal physics. 
  
- **Physics-First Optimization**: We discovered that **Cross-Entropy Loss** is a poor metric for quantum claribration due to inherent probabilistic entropy. We transitioned to using **Relatice Physical Error $(\epsilon)$** as the primary driver for early stopping and model selection. 

### Key Performance 

- **Accuracy:*** 80% (Projective measurement/the label contributing to the loss)
- **Calibration** Physical Error $\epsilon \approx 0.0005$, outperforming the paper's benchmark of 0.01
- **Efficiency** Achieved convergene in 270, 0000 traces, an 82% reduction in data volume compared to original research. 

---

**Credits & Collaboration**
I developed this project with architectural guidance, physics debugging and research narrative assistance provided by **Gemini**. This allowed me to dive deep in to the intersection of LSTMs and Stochastic Master Equation (SME), specifically optimizing for limited computation environments. 

