# maritime-tsp-qaoa-greedy
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.36%2B-brightgreen)](https://pennylane.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

# Maritime TSP — Quantum-Assisted Greedy Optimization

A hybrid **quantum–classical routing framework** for solving a **maritime Traveling Salesman Problem (TSP)** using **QAOA-assisted greedy decision making**.

This repository accompanies the research work:

**“An Evaluation of a Hybrid Quantum–Classical Approach for Maritime Route Optimization”**  
Author: **Aditya Singh**

The project investigates whether **variational quantum algorithms in the NISQ era** can meaningfully assist classical routing heuristics for realistic maritime logistics problems.

---

# Overview

The **Traveling Salesman Problem (TSP)** is a classical **NP-hard combinatorial optimization problem** in which the objective is to determine the shortest route that visits each location exactly once and returns to the starting point.

In maritime logistics, routing decisions must follow **navigable sea routes**, meaning that distances cannot be approximated using simple Euclidean geometry. Instead, realistic routing must consider **coastlines, sea corridors, and geographic constraints**.

This project implements a **hybrid quantum–classical optimization framework** where:

- The global TSP is decomposed into **sequential greedy routing decisions**
- A **Quantum Approximate Optimization Algorithm (QAOA)** circuit assists in selecting the next port
- **Classical heuristics** ensure feasibility and stability of the route

The goal is to **empirically evaluate the usefulness of shallow quantum circuits for practical logistics optimization problems.**

---

# Features

## Real Maritime Distances

Distances are computed using the **searoute library**, which calculates **real navigable sea paths** rather than straight-line Euclidean distances.

This produces a more realistic routing environment compared with typical academic TSP benchmarks.

---

## Hybrid Quantum–Classical Algorithm

The routing algorithm combines classical greedy construction with quantum-assisted decision making.

Key components include:

- **QAOA variational circuits**
- **XY mixer Hamiltonian**
- **One-hot encoded decision states**
- **Classical greedy fallback when qubit capacity is exceeded**

The global routing problem is therefore decomposed into a sequence of **small quantum decision problems**.

---

## Classical Baselines

The hybrid approach is benchmarked against several classical optimization methods:

- Random Search
- Hill Climbing
- Nearest Neighbor heuristic
- **LKH-3 solver** (state-of-the-art TSP heuristic)

These comparisons provide context for evaluating the effectiveness of quantum-assisted decisions.

---

## Experimental Studies

The repository includes experiments analyzing:

- **QAOA circuit depth (layer count)**
- **Look-ahead parameter (α)**
- **Quantum noise sensitivity**
- **Quantum decision capacity (qubit limits)**
- **Statistical performance across multiple random seeds**

These experiments explore how **NISQ-era limitations influence optimization performance**.

---

# Maritime Network

The dataset contains **20 international ports**, including:

Mumbai • Chennai • Kolkata • Kochi • Visakhapatnam • Paradip • Goa • Tuticorin • Haldia • Singapore • Colombo • Jebel Ali • Port Klang • Shanghai • Busan • Rotterdam • Fujairah • Port Hedland • Yokohama

Distances are computed using **real maritime routes** obtained through the `searoute` library.

---

# Algorithm Pipeline

The hybrid algorithm operates as follows:

1. Start from a selected port  
2. Maintain a set of **unvisited ports**
3. If the candidate set exceeds the **available qubit limit**, use classical greedy fallback
4. Otherwise:

   - Construct the **QAOA cost Hamiltonian**
   - Apply the **XY mixer Hamiltonian**
   - Optimize circuit parameters using **Nesterov momentum**

5. Sample the quantum output distribution to **select the next port**
6. Repeat until the full route is constructed

This strategy decomposes the **global TSP into a sequence of small quantum decision problems**.

---

# Requirements

Python **3.9+**

Install dependencies:

```bash
pip install pennylane pennylane-lightning numpy matplotlib searoute
```

Optional (recommended for benchmarking):

Download **LKH-3** and place the executable in the project directory.

---

# Running the Program

## Default execution

```bash
python maritime_tsp_quantum.py
```

## Custom configuration

```bash
python maritime_tsp_quantum.py --layers 3 --steps 300 --start_idx 0
```

### Parameters

| Parameter | Description |
|----------|-------------|
| `--start_idx` | Starting port index |
| `--layers` | QAOA circuit depth |
| `--steps` | Optimizer iterations |

---

# Notable Findings

Experimental results suggest several observations:

- Reducing **qubit cap from 8 to 4** improved mean cost  
  **92,544 km → 66,348 km**, indicating classical decisions dominate early routing quality.

- Noise levels **p = 0.001–0.050** showed minimal performance degradation, suggesting the algorithm is **not yet noise-limited**.

- **1-layer QAOA circuits** sometimes outperformed deeper circuits, implying that **optimizer budget rather than circuit depth** is the main bottleneck.

- Across **8 random seeds**, the algorithm exhibited a **17.1% coefficient of variation**, indicating sensitivity to stochastic initialization.

These results highlight current **limitations of shallow quantum circuits for large-scale logistics optimization**.

---

# License

This project is released under the **MIT License**.
