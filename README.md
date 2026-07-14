# Stochastic Simulation & Kinetic Theory Framework (Monte Carlo)

## Overview
This repository contains a robust computational framework deploying Monte Carlo simulations to model particle dynamics, evaluate variance reduction techniques, and simulate stochastic probability pathways. The codebase provides a rigorous algorithmic solution to tracking system states over time, where deterministic equations are computationally prohibitive.

## Mathematical Formulation
The framework simulates state transitions governed by stochastic processes. By utilising pseudo-random sampling from probability distributions, the model approximates multi-dimensional integrals representing expected system states:

$$E[X] \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i)$$

Where N is the number of simulation trials and $x_i$ are independent, identically distributed random variables.

### Algorithmic Architecture
* **Stochastic Path Generation**: Simulates highly liquid, unpredictable system states using precise probabilistic distributions.
* **Convergence & Variance Reduction**: Optimised tracking loops that measure convergence rates, ensuring statistical stability as the sample size N scales.
* **Kinetic Theory Application**: Modelled for multi-particle system simulations, analysing density variations and random collision pathways.

## Getting Started

### Prerequisites
* Python 3.8+
* SciPy / NumPy (For statistical probability distribution mapping)

### Quick Execution
```python
from monte_carlo_simulation import run_stochastic_simulation

# Initialise simulation hyper-parameters
trials = 100000
initial_state = 1.0

# Execute Monte Carlo simulation pipeline
simulation_results = run_stochastic_simulation(num_trials=trials, start_value=initial_state)
print(f"Approximated Expected System State: {simulation_results}")
```

## Academic Context
Developed as part of the core curriculum in **Kinetic Theory and Stochastic Simulation** during the MSc in Mathematical Engineering program at the **University of L'Aquila, Italy**.
