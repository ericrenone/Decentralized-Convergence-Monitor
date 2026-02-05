# Decentralized Convergence Monitor

## Overview
The **Decentralized Convergence Monitor** is a **real-time visualization tool** for analyzing the **collective behavior of 1,000 autonomous nodes** in a decentralized network. It tracks:
- **Velocity distribution** of node movements.
- **Convergence trend** over time (log scale).

**Core Purpose:** Demonstrates **emergent convergence dynamics** in large-scale decentralized systems, with applications in **low-resource optimization**, **consensus protocols**, and **federated learning**.

---

## Key Features

### 1. **Real-Time Velocity Distribution**
- **Histogram** of node velocities, updated dynamically.
- **Density-based visualization** for intuitive understanding of system behavior.

### 2. **Convergence Trend Analysis**
- **Log-scale time series** of mean velocity.
- **Dynamic scaling** for real-time feel and clarity.

### 3. **Low-Resource Design**
- **Efficient updates** using vectorized operations.
- **Minimal dependencies** (NumPy, Matplotlib).

### 4. **Configurable Parameters**
- **Node count** (`N_NODES`).
- **Convergence rate** (`ALPHA`).
- **Noise level** (`NOISE_STD`).
- **Simulation duration** (`ROUNDS`).


**Why it matters:** Models **decentralized consensus** (e.g., federated averaging) with **controlled noise** for robustness.

---
