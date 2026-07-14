#!/usr/bin/env python3
"""
Monte Carlo Simulation of the 2D Ising Model
=============================================

This script implements a Monte Carlo simulation of the 2D Ising model
using the Metropolis algorithm. It was originally developed as a project
for the course "Kinetic Theory and Stochastic Simulation".

Features:
- Efficient implementation with O(1) energy updates
- Multiple initial conditions (all up, all down, random)
- Thermodynamic quantities: magnetization, energy, susceptibility, specific heat
- Snapshot saving for visualization
- Publication-quality plots

Author: Alex Hammah
Course: Kinetic Theory and Stochastic Simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import os

# =============================================================================
# PARAMETERS
# =============================================================================

L = 16                    # Lattice size (L x L) - reduced for fast execution
N = L * L                 # Total number of spins
J = 1.0                   # Coupling constant
RANDOM_SEED = 42          # For reproducibility

np.random.seed(RANDOM_SEED)

# =============================================================================
# INITIAL CONFIGURATIONS
# =============================================================================

def initialize_lattice(L: int, mode: str = "random") -> np.ndarray:
    """
    Initialize a 2D spin lattice.
    
    Parameters
    ----------
    L : int
        Lattice size (L x L)
    mode : str
        'up'    : all spins +1
        'down'  : all spins -1
        'random': random ±1 spins
        
    Returns
    -------
    spins : np.ndarray
        2D array of shape (L, L) with values ±1
    """
    if mode == "up":
        return np.ones((L, L), dtype=np.int8)
    elif mode == "down":
        return -np.ones((L, L), dtype=np.int8)
    elif mode == "random":
        return np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# ENERGY AND MAGNETIZATION
# =============================================================================

def compute_energy(spins: np.ndarray, J: float = 1.0) -> float:
    """
    Compute total energy of the Ising configuration.
    
    Uses vectorized operations for speed.
    
    Parameters
    ----------
    spins : np.ndarray
        2D spin configuration
    J : float
        Coupling constant
        
    Returns
    -------
    energy : float
        Total energy (sum over nearest-neighbor pairs)
    """
    L = spins.shape[0]
    
    # Right and down neighbors (periodic boundaries)
    right = np.roll(spins, -1, axis=1)
    down  = np.roll(spins, -1, axis=0)
    
    # Each bond counted once
    energy = -J * np.sum(spins * (right + down))
    return energy


def compute_magnetization(spins: np.ndarray) -> float:
    """Compute total magnetization (sum of all spins)."""
    return float(np.sum(spins))


# =============================================================================
# METROPOLIS ALGORITHM (Efficient implementation)
# =============================================================================

def metropolis_step(spins: np.ndarray, beta: float, J: float = 1.0) -> None:
    """
    Perform one Monte Carlo sweep using the Metropolis algorithm.
    
    Updates L*L spins. Energy is computed locally for efficiency.
    
    Parameters
    ----------
    spins : np.ndarray
        Current spin configuration (modified in-place)
    beta : float
        Inverse temperature (1/T)
    J : float
        Coupling constant
    """
    L = spins.shape[0]
    
    for _ in range(N):
        # Pick random site
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        
        S = spins[i, j]
        
        # Sum of four neighbors (periodic boundaries)
        neighbors = (
            spins[(i + 1) % L, j] +
            spins[(i - 1) % L, j] +
            spins[i, (j + 1) % L] +
            spins[i, (j - 1) % L]
        )
        
        # Energy change if we flip this spin
        dE = 2 * J * S * neighbors
        
        # Metropolis acceptance
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] *= -1


def metropolis_step_vectorized(spins: np.ndarray, beta: float, J: float = 1.0) -> None:
    """
    Vectorized Metropolis step (alternative implementation).
    
    Note: The single-site version above is usually faster for small L.
    """
    L = spins.shape[0]
    
    # Propose flipping all spins at once (Glauber dynamics style)
    # This is less efficient than single-site for small systems
    i = np.random.randint(0, L, size=N)
    j = np.random.randint(0, L, size=N)
    
    for idx in range(N):
        ii, jj = i[idx], j[idx]
        S = spins[ii, jj]
        neighbors = (spins[(ii+1)%L, jj] + spins[(ii-1)%L, jj] +
                     spins[ii, (jj+1)%L] + spins[ii, (jj-1)%L])
        dE = 2 * J * S * neighbors
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[ii, jj] *= -1


# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def run_simulation(
    T: float,
    steps: int,
    L: int = 32,
    initial_condition: str = "random",
    J: float = 1.0,
    snapshot_interval: int = 2000
) -> Tuple[List[float], List[float], Dict[int, np.ndarray]]:
    """
    Run a single Monte Carlo simulation at fixed temperature.
    
    Parameters
    ----------
    T : float
        Temperature
    steps : int
        Number of Monte Carlo sweeps
    L : int
        Lattice size
    initial_condition : str
        'up', 'down', or 'random'
    J : float
        Coupling constant
    snapshot_interval : int
        Take snapshots every this many steps
        
    Returns
    -------
    magnetization : list
        Magnetization per spin at each step
    energy : list
        Energy per spin at each step
    snapshots : dict
        {step: configuration} at selected times
    """
    beta = 1.0 / T
    spins = initialize_lattice(L, initial_condition)
    
    magnetization = []
    energy = []
    snapshots = {}
    
    snapshot_times = [snapshot_interval * k for k in range(1, 10)]
    
    for step in range(steps):
        metropolis_step(spins, beta, J)
        
        m = compute_magnetization(spins) / N
        e = compute_energy(spins, J) / N
        
        magnetization.append(m)
        energy.append(e)
        
        if step in snapshot_times:
            snapshots[step] = spins.copy()
    
    return magnetization, energy, snapshots


# =============================================================================
# THERMODYNAMIC QUANTITIES
# =============================================================================

def compute_thermodynamic_quantities(
    T_values: np.ndarray,
    n_thermal: int = 5000,
    n_measure: int = 5000,
    L: int = 32,
    J: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Compute thermodynamic quantities over a temperature range.
    
    Returns a dictionary with:
        'T', 'mean_m', 'mean_e', 'chi', 'C'
    """
    results = {
        'T': T_values,
        'mean_m': [],
        'mean_e': [],
        'chi': [],
        'C': []
    }
    
    for T in T_values:
        beta = 1.0 / T
        spins = initialize_lattice(L, "random")
        
        M_samples = []
        E_samples = []
        
        # Thermalization
        for _ in range(n_thermal):
            metropolis_step(spins, beta, J)
        
        # Measurements
        for _ in range(n_measure):
            metropolis_step(spins, beta, J)
            
            M = compute_magnetization(spins)
            E = compute_energy(spins, J)
            
            M_samples.append(M)
            E_samples.append(E)
        
        M_arr = np.array(M_samples)
        E_arr = np.array(E_samples)
        
        mean_m = np.mean(np.abs(M_arr)) / N
        mean_e = np.mean(E_arr) / N
        
        var_m = np.var(M_arr) / N
        var_e = np.var(E_arr) / N
        
        chi = beta * var_m
        C = beta**2 * var_e
        
        results['mean_m'].append(mean_m)
        results['mean_e'].append(mean_e)
        results['chi'].append(chi)
        results['C'].append(C)
    
    # Convert lists to arrays
    for key in results:
        results[key] = np.array(results[key])
    
    return results


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_magnetization_vs_time(
    T: float,
    steps: int = 20000,
    L: int = 32,
    save_path: Optional[str] = None
):
    """Plot magnetization evolution for different initial conditions."""
    initial_conditions = ["up", "down", "random"]
    
    plt.figure(figsize=(8, 5))
    
    for ic in initial_conditions:
        m, _, _ = run_simulation(T, steps, L, ic)
        plt.plot(m, label=ic, alpha=0.8)
    
    plt.xlabel("Monte Carlo sweeps")
    plt.ylabel("Magnetization per spin")
    plt.title(f"Magnetization vs Time (T = {T})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_thermodynamic_quantities(
    results: Dict[str, np.ndarray],
    save_dir: str = "figures"
):
    """Plot all thermodynamic quantities."""
    os.makedirs(save_dir, exist_ok=True)
    
    quantities = [
        ('mean_m', 'Mean |Magnetization|', 'mean_magnetization.png'),
        ('mean_e', 'Mean Energy per spin', 'mean_energy.png'),
        ('chi', 'Susceptibility', 'susceptibility.png'),
        ('C', 'Specific Heat', 'specific_heat.png')
    ]
    
    for key, ylabel, filename in quantities:
        plt.figure(figsize=(7, 5))
        plt.plot(results['T'], results[key], 'o-', markersize=4)
        plt.xlabel("Temperature")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close()


def plot_snapshots(
    snapshots: Dict[int, np.ndarray],
    save_path: Optional[str] = None
):
    """Plot spin configurations at different times."""
    n = len(snapshots)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
    for ax, (time, config) in zip(axes.flatten(), snapshots.items()):
        ax.imshow(config, cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f"t = {time}")
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("2D Ising Model Monte Carlo Simulation")
    print("=" * 50)
    print(f"Lattice size: {L} x {L}")
    print(f"Total spins: {N}")
    print()
    
    # Quick demo (reduced steps for fast execution)
    print("Running quick demo (Magnetization vs Time)...")
    plot_magnetization_vs_time(
        T=2.0,
        steps=3000,           # reduced for demo
        L=L,
        save_path="magnetization_demo.png"
    )
    
    print("\nDemo complete!")
    print("For full analysis, increase steps and use compute_thermodynamic_quantities()")
