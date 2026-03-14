# maritime-tsp-qaoa-greedy
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.36%2B-brightgreen)](https://pennylane.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Maritime TSP — Quantum-Assisted Greedy Optimization

Hybrid quantum–classical routing framework for solving a maritime Traveling Salesman Problem (TSP) using QAOA-assisted greedy decision making.

## This repository accompanies the research work:

“An Evaluation of a Hybrid Quantum–Classical Approach for Maritime Route Optimization”
Author: Aditya Singh

The project explores how variational quantum algorithms in the NISQ era can assist classical routing heuristics for realistic maritime logistics problems.

## Overview

The Traveling Salesman Problem (TSP) is a classic NP-hard combinatorial optimization problem. In maritime logistics, routing decisions must respect real navigable sea routes, making the optimization more realistic and complex.

## This project implements a hybrid quantum–classical framework where:

The global TSP is decomposed into sequential greedy decisions

A Quantum Approximate Optimization Algorithm (QAOA) circuit guides the selection of the next port

Classical heuristics ensure feasibility and robustness

The goal is to empirically evaluate the effectiveness of shallow quantum circuits for real routing problems.

## Features
Real Maritime Distances

Distances are computed using the searoute library, which calculates navigable sea paths instead of Euclidean distance.

Hybrid Quantum–Classical Algorithm

A greedy construction strategy is enhanced with:

QAOA variational circuits

XY mixer Hamiltonian

One-hot encoded decision states

Classical Baselines

The implementation compares against several classical algorithms:

Random Search

Hill Climbing

Nearest Neighbor heuristic

LKH-3 solver (state-of-the-art TSP heuristic)

Experimental Studies

The repository includes experiments analyzing:

QAOA layer depth

Alpha parameter for look-ahead cost

Noise sensitivity in quantum circuits

Quantum decision capacity (qubit limits)

Statistical performance across random seeds

Maritime Network

## The dataset contains 20 international ports, including:

Mumbai

Chennai

Kolkata

Kochi

Visakhapatnam

Paradip

Goa

Tuticorin

Haldia

Singapore

Colombo

Jebel Ali

Port Klang

Shanghai

Busan

Rotterdam

Fujairah

Port Hedland

Yokohama

- Distances are calculated using realistic maritime routes.

## Algorithm Pipeline

The hybrid algorithm operates as follows:

- Start at a selected port

- Maintain a set of unvisited ports

- If the candidate set is larger than the maximum available qubits, use classical greedy fallback

Otherwise:

- Build a QAOA cost Hamiltonian

- Apply an XY mixer Hamiltonian

- Optimize circuit parameters using Nesterov momentum

- Sample the quantum output distribution to select the next port

- Repeat until the full route is constructed

- This approach reduces the global TSP into small quantum decision problems.

## Requirements

- Python 3.9+

Install dependencies:

- pip install pennylane pennylane-lightning numpy matplotlib searoute

Optional (for best classical baseline):

- Download LKH-3 solver and place the executable in the project folder.

- Running the Program

## Notable Findings

- Reducing qubit cap from 8 to 4 improved mean cost 
  from 92,544 km to 66,348 km — classical beats quantum 
  for early decisions
- Noise levels p=0.001 to p=0.050 showed no meaningful 
  degradation — algorithm is not yet noise-limited
- 1-layer QAOA outperformed 3–4 layer circuits — 
  optimizer budget is the bottleneck, not circuit depth
- CV = 17.1% across 8 seeds — high sensitivity to 
  stochastic initialization

## Default execution:

python maritime_tsp_quantum.py

Custom configuration:

python maritime_tsp_quantum.py --layers 3 --steps 300 --start_idx 0
