# Statistical Mechanics of Learning
## Physics of Data | UdeA 2026-1

This repository serves as a computational laboratory for the study of statistical learning. We approach artificial intelligence not as a collection of heuristic algorithms, but as a complex physical system governed by the principles of information theory, thermodynamics, and high-dimensional geometry.

### Theoretical Framework

In this project, we interpret the learning process through the following physical formalisms:

1. **Energy Landscapes**: The loss function $J(\theta)$ is treated as a Hamiltonian $\mathcal{H}(\theta)$, representing the energy of the system given a configuration of parameters (spins) $\theta$.
2. **Stochastic Dynamics**: Optimization is modeled as the navigation of this landscape. Stochastic Gradient Descent (SGD) is interpreted as a discretized Langevin equation:
   $$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{H}(\theta_t) + \sqrt{2\eta T} \xi_t$$
   where $\eta$ is the learning rate and $T$ represents the effective temperature of the system.
3. **Inference and Gibbs Distribution**: We seek the ground state of the system or, in Bayesian terms, to sample from the Gibbs distribution:
   $$P(\theta | \mathcal{D}) = \frac{1}{\mathcal{Z}} \exp\left(-\beta \mathcal{H}(\theta)\right)$$
   where $\beta = 1/T$ is the inverse temperature and $\mathcal{Z}$ is the partition function.

### Repository Architecture

The project adheres to a decoupled architecture to ensure scientific reproducibility and modularity:

- `src/`: **The Engine.** Contains the core algorithmic implementations and mathematical definitions (The "Laws of Motion").
- `notebooks/`: **The Logbook.** Experimental execution, data visualization, and interpretation of results.
- `data/`: **The Phase Space.** Immutable raw data and processed tensors.
- `artifacts/`: **State Files.** Verifiable outputs (JSON/PNG) generated during the experimental phases.

### Installation

To initialize the laboratory environment and install the local package in an editable state:

```bash
# Clone the repository
git clone https://github.com/SiririComun/Physics-of-Data.git
cd Physics-of-Data

# Install dependencies and local library
pip install -e .
```

### Course Context
- **Institution:** Universidad de Antioquia (UdeA)
- **Course:** Física Computacional 2
- **Topic:** Statistical Learning
- **Author:** Juan Pablo Sanchez

---
*“The goal of learning is to find the simplest description of the data that is consistent with the observations.”*
```