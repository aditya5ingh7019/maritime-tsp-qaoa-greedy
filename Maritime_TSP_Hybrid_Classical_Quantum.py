#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maritime TSP — Quantum-Assisted Greedy Optimization
====================================================
Research code accompanying the paper:
  "Hybrid QAOA-Assisted Greedy Routing for Maritime TSP"
  Author: Aditya Singh

Description:
    Implements a hybrid quantum-classical greedy routing framework
    for a 20-port maritime Traveling Salesman Problem using real
    navigable sea-route distances. QAOA with XY mixer guides local
    port selection within a greedy construction pipeline.

Functions:
    build_quantum_greedy_tour_ideal  — noiseless, lightning.qubit
    build_quantum_greedy_tour_noisy  — noisy, default.mixed
    build_quantum_greedy_tour_qcap   — qubit cap sensitivity study

Usage:
    python maritime_tsp_quantum.py
    python maritime_tsp_quantum.py --layers 1 --steps 300 --start_idx 0

Requirements:
    pennylane, pennylane-lightning, searoute, numpy, matplotlib
    LKH-3 executable at ./LKH-3.exe
"""

import argparse
import time
import numpy as np
import random
import searoute as sr
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt
import subprocess
import os
import tempfile
import sys

# =============================================================================
# PORT DATA AND DISTANCE MATRIX
# =============================================================================

PORTS = {
    "Mumbai": (18.94, 72.83),
    "Chennai": (13.08, 80.28),
    "Kolkata": (22.56, 88.34),
    "Kochi": (9.96, 76.27),
    "Visakhapatnam": (17.70, 83.30),
    "Paradip": (20.32, 86.61),
    "Goa": (15.40, 73.81),
    "Tuticorin": (8.77, 78.13),
    "Haldia": (22.03, 88.06),
    "Mormugao": (15.41, 73.81),
    "Singapore": (1.30, 103.77),
    "Colombo": (6.95, 79.85),
    "Jebel Ali": (25.00, 55.05),
    "Port Klang": (3.00, 101.40),
    "Shanghai": (31.23, 121.47),
    "Busan": (35.10, 129.04),
    "Rotterdam": (51.92, 4.48),
    "Fujairah": (25.12, 56.34),
    "Port Hedland": (-20.31, 118.57),
    "Yokohama": (35.44, 139.64),
}

PORT_LIST = list(PORTS.keys())
N_PORTS = len(PORT_LIST)


def maritime_distance(lat1, lon1, lat2, lon2):
    if abs(lat1 - lat2) < 1e-6 and abs(lon1 - lon2) < 1e-6:
        return 0.0
    try:
        route = sr.searoute((lon1, lat1), (lon2, lat2))
        return route['properties']['length']
    except Exception:
        return 999999.0


D = np.zeros((N_PORTS, N_PORTS))
for i in range(N_PORTS):
    for j in range(N_PORTS):
        if i != j:
            lat1, lon1 = PORTS[PORT_LIST[i]]
            lat2, lon2 = PORTS[PORT_LIST[j]]
            D[i, j] = maritime_distance(lat1, lon1, lat2, lon2)


def route_cost(route):
    total = sum(D[route[i], route[i+1]] for i in range(len(route)-1))
    total += D[route[-1], route[0]]
    return total


# =============================================================================
# CLASSICAL BASELINES
# =============================================================================

def random_search(n_trials=50000):
    best_cost = float('inf')
    best_route = None
    for _ in range(n_trials):
        route = list(range(N_PORTS))
        random.shuffle(route)
        cost = route_cost(route)
        if cost < best_cost:
            best_cost = cost
            best_route = route[:]
    return best_route, best_cost


def hill_climbing(start_route, n_steps=20000):
    current_route = start_route[:]
    current_cost = route_cost(current_route)
    for _ in range(n_steps):
        i, j = random.sample(range(N_PORTS), 2)
        new_route = current_route[:]
        new_route[i], new_route[j] = new_route[j], new_route[i]
        new_cost = route_cost(new_route)
        if new_cost < current_cost:
            current_route = new_route
            current_cost = new_cost
    return current_route, current_cost


def nearest_neighbor_tour(start_idx):
    unvisited = set(range(N_PORTS))
    unvisited.remove(start_idx)
    tour = [start_idx]
    current = start_idx
    while unvisited:
        next_port = min(unvisited, key=lambda x: D[current, x])
        tour.append(next_port)
        unvisited.remove(next_port)
        current = next_port
    return tour, route_cost(tour)


# =============================================================================
# LKH-3 INTEGRATION
# =============================================================================

LKH_EXE = r".\LKH-3.exe"


def save_tsplib_file(dist_matrix, port_names, filename):
    n = len(port_names)
    with open(filename, 'w') as f:
        f.write(f"NAME : MaritimeTSP\n")
        f.write(f"COMMENT : {n} ports\n")
        f.write(f"TYPE : TSP\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for i in range(n):
            f.write(" ".join(f"{int(round(dist_matrix[i,j]))}" for j in range(n)) + "\n")
        f.write("EOF\n")


def run_lkh(dist_matrix, port_names, max_trials=100, pop_size=1):
    with tempfile.TemporaryDirectory() as tmpdir:
        problem_file = os.path.join(tmpdir, "problem.tsp")
        par_file = os.path.join(tmpdir, "params.par")
        tour_file = os.path.join(tmpdir, "best.tour")
        output_file = os.path.join(tmpdir, "lkh.out")

        save_tsplib_file(dist_matrix, port_names, problem_file)
        with open(par_file, 'w') as f:
            f.write(f"PROBLEM_FILE = {problem_file}\n")
            f.write(f"OUTPUT_TOUR_FILE = {tour_file}\n")
            f.write(f"RUNS = {max_trials}\n")
            f.write("TRACE_LEVEL = 1\n")
            f.write("MOVE_TYPE = 5\n")

        if not os.path.exists(LKH_EXE):
            print(f"ERROR: LKH executable not found at: {LKH_EXE}")
            return None, float('inf')

        try:
            subprocess.run(
                [LKH_EXE, par_file],
                stdout=open(output_file, "w"),
                stderr=subprocess.STDOUT,
                check=True, timeout=300, shell=False
            )
        except Exception as e:
            print(f"LKH error: {e}")
            return None, float('inf')

        tour = []
        reading_tour = False
        try:
            with open(tour_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == "TOUR_SECTION":
                        reading_tour = True
                        continue
                    if reading_tour:
                        if line in ("-1", "EOF"):
                            break
                        try:
                            idx = int(line) - 1
                            if 0 <= idx < len(port_names):
                                tour.append(idx)
                        except ValueError:
                            pass
        except FileNotFoundError:
            print("Tour file not created")
            return None, float('inf')

        if not tour or len(tour) != len(port_names):
            print(f"Tour length mismatch: got {len(tour)}, expected {len(port_names)}")
            return None, float('inf')

        return tour, route_cost(tour)


# =============================================================================
# QUANTUM-ASSISTED GREEDY
# =============================================================================

def build_quantum_greedy_tour_ideal(
    start_idx=0, layers=3, steps_per_decision=300,
    alpha=0.5
):
    MAX_QUBITS = 8
    unvisited = set(range(N_PORTS))
    unvisited.remove(start_idx)
    tour = [start_idx]
    current = start_idx
    penalty = 10.0 * np.max(D)

    print(f"\nStarting from {PORT_LIST[start_idx]}")

    while unvisited:
        k = len(unvisited)

        if k > MAX_QUBITS:
            next_idx = min(unvisited, key=lambda x: D[current, x])
            print(f" {k:2d} ports left → chose {PORT_LIST[next_idx]} (classical fallback)")
            tour.append(next_idx)
            unvisited.remove(next_idx)
            current = next_idx
            continue

        rem = list(unvisited)
        effective_costs = []
        for j in rem:
            future = min(D[j, x] for x in unvisited if x != j) if len(unvisited) > 1 else 0
            effective_costs.append(D[current, j] + alpha * future)

        coeffs, ops = [], []
        for i, d in enumerate(effective_costs):
            coeffs += [d / 2, -d / 2]
            ops += [qml.Identity(i), qml.PauliZ(i)]
        for i in range(k):
            for j in range(i + 1, k):
                coeffs.append(penalty / 4)
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
        H_cost = qml.Hamiltonian(coeffs, ops)

        mixer_coeffs, mixer_ops = [], []
        for i in range(k):
            for j in range(i + 1, k):
                mixer_coeffs += [1.0, 1.0]
                mixer_ops += [
                    qml.PauliX(i) @ qml.PauliX(j),
                    qml.PauliY(i) @ qml.PauliY(j)
                ]
        H_mixer = qml.Hamiltonian(mixer_coeffs, mixer_ops)

        dev = qml.device("lightning.qubit", wires=k)

        @qml.qnode(dev)
        def energy_circuit(g, b):
            for i in range(k):
                qml.Hadamard(i)

            for gg, bb in zip(g, b):
                qml.qaoa.cost_layer(gg, H_cost)
                qml.qaoa.mixer_layer(bb, H_mixer)

            return qml.expval(H_cost)

        @qml.qnode(dev)
        def prob_circuit(g, b):
            for i in range(k):
                qml.Hadamard(i)

            for gg, bb in zip(g, b):
                qml.qaoa.cost_layer(gg, H_cost)
                qml.qaoa.mixer_layer(bb, H_mixer)

            return qml.probs(wires=range(k))

        g = pnp.random.uniform(0.05, 0.2, layers, requires_grad=True)
        b = pnp.random.uniform(1.0, 1.5, layers, requires_grad=True)
        opt = NesterovMomentumOptimizer(0.03)
        prev_energy = float("inf")
        stable = 0

        print(f" {k:2d} ports left → optimizing ({layers} layers, max {steps_per_decision} steps)... ",
              end="", flush=True)

        for step in range(steps_per_decision):
            g, b = opt.step(lambda x, y: energy_circuit(x, y), g, b)
            energy = energy_circuit(g, b)
            if abs(energy - prev_energy) < 1e-2:
                stable += 1
                if stable > 20:
                    print(f"(early stop at step {step})", end=" ", flush=True)
                    break
            else:
                stable = 0
            prev_energy = energy
            if step % 30 == 0 and step > 0:
                print(f"{step} ", end="", flush=True)

        print("done", flush=True)

        probs = prob_circuit(g, b)
        valid_states, weights = [], []
        for s in range(2**k):
            bs = format(s, f"0{k}b")
            if bs.count("1") == 1:
                valid_states.append(bs)
                weights.append(probs[s])

        if sum(weights) > 0:
            chosen = random.choices(valid_states, weights=weights)[0]
            next_idx = rem[chosen.index("1")]
            prob_str = f"quantum prob: {max(weights):.4f}"
        else:
            next_idx = min(unvisited, key=lambda x: D[current, x])
            prob_str = "fallback (nearest neighbor)"

        print(f" → chose {PORT_LIST[next_idx]} ({prob_str})", flush=True)
        tour.append(next_idx)
        unvisited.remove(next_idx)
        current = next_idx

    tour.append(start_idx)
    total_cost = route_cost(tour)
    tour_names = [PORT_LIST[i] for i in tour]
    path_indices = tour[:-1]
    return tour_names, total_cost, path_indices

def build_quantum_greedy_tour_noisy(
    start_idx=0, layers=3, steps_per_decision=150,
    alpha=0.5, noise_level=0.01        
):
    MAX_QUBITS = 8
    unvisited = set(range(N_PORTS))
    unvisited.remove(start_idx)
    tour = [start_idx]
    current = start_idx
    penalty = 10.0 * np.max(D)

    print(f"\nStarting from {PORT_LIST[start_idx]} (noisy simulation)")

    while unvisited:
        k = len(unvisited)

        if k > MAX_QUBITS:
            next_idx = min(unvisited, key=lambda x: D[current, x])
            print(f" {k:2d} ports left → chose {PORT_LIST[next_idx]} (classical fallback)")
            tour.append(next_idx)
            unvisited.remove(next_idx)
            current = next_idx
            continue

        rem = list(unvisited)
        effective_costs = []
        for j in rem:
            future = min(D[j, x] for x in unvisited if x != j) if len(unvisited) > 1 else 0
            effective_costs.append(D[current, j] + alpha * future)

        coeffs, ops = [], []
        for i, d in enumerate(effective_costs):
            coeffs += [d / 2, -d / 2]
            ops += [qml.Identity(i), qml.PauliZ(i)]
        for i in range(k):
            for j in range(i + 1, k):
                coeffs.append(penalty / 4)
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
        H_cost = qml.Hamiltonian(coeffs, ops)

        mixer_coeffs, mixer_ops = [], []
        for i in range(k):
            for j in range(i + 1, k):
                mixer_coeffs += [1.0, 1.0]
                mixer_ops += [
                    qml.PauliX(i) @ qml.PauliX(j),
                    qml.PauliY(i) @ qml.PauliY(j)
                ]
        H_mixer = qml.Hamiltonian(mixer_coeffs, mixer_ops)

        # DepolarizingChannel
        dev = qml.device("default.mixed", wires=k)

        @qml.qnode(dev)
        def energy_circuit(g, b):
            for i in range(k):
                qml.Hadamard(i)
            for gg, bb in zip(g, b):
                qml.qaoa.cost_layer(gg, H_cost)
                qml.qaoa.mixer_layer(bb, H_mixer)
            for wire in range(k):
                qml.DepolarizingChannel(noise_level, wires=wire) 
            return qml.expval(H_cost)

        @qml.qnode(dev)
        def prob_circuit(g, b):
            for i in range(k):
                qml.Hadamard(i)
            for gg, bb in zip(g, b):
                qml.qaoa.cost_layer(gg, H_cost)
                qml.qaoa.mixer_layer(bb, H_mixer)
            for wire in range(k):
                qml.DepolarizingChannel(noise_level, wires=wire)  
            return qml.probs(wires=range(k))

        g = pnp.random.uniform(0.05, 0.2, layers, requires_grad=True)
        b = pnp.random.uniform(1.0, 1.5, layers, requires_grad=True)
        opt = NesterovMomentumOptimizer(0.03)
        prev_energy = float("inf")
        stable = 0

        print(f" {k:2d} ports left → optimizing ({layers} layers, max {steps_per_decision} steps)... ",
        end="", flush=True)

        for step in range(steps_per_decision):
            g, b = opt.step(lambda x, y: energy_circuit(x, y), g, b)
            energy = energy_circuit(g, b)
            if step % 30 == 0 and step > 0:
                print(f"{step} ", end="", flush=True)
            if abs(energy - prev_energy) < 1e-2:
                stable += 1
                if stable > 20:
                    break
            else:
                stable = 0
            prev_energy = energy

        print("done", flush=True)

        probs = prob_circuit(g, b)
        valid_states, weights = [], []
        for s in range(2**k):
            bs = format(s, f"0{k}b")
            if bs.count("1") == 1:
                valid_states.append(bs)
                weights.append(probs[s])
        if sum(weights) > 0:
            chosen = random.choices(valid_states, weights=weights)[0]
            next_idx = rem[chosen.index("1")]
            prob_str = f"quantum prob: {max(weights):.4f}"
        else:
            next_idx = min(unvisited, key=lambda x: D[current, x])
            prob_str = "fallback (nearest neighbor)"
        print(f" → chose {PORT_LIST[next_idx]} ({prob_str})", flush=True)
        tour.append(next_idx)
        unvisited.remove(next_idx)
        current = next_idx

    tour.append(start_idx)
    total_cost = route_cost(tour)
    tour_names = [PORT_LIST[i] for i in tour]
    path_indices = tour[:-1]
    return tour_names, total_cost, path_indices

# =============================================================================
# PLOTTING
# =============================================================================

def plot_costs(q_cost, hc_cost, lkh_cost):

    labels = ["Quantum Greedy", "Hill Climbing", "LKH"]
    costs = [q_cost, hc_cost, lkh_cost]

    plt.figure(figsize=(6,4))
    plt.bar(labels, costs)

    plt.ylabel("Route Distance (km)")
    plt.title("Algorithm Cost Comparison")

    for i, v in enumerate(costs):
        plt.text(i, v + 500, f"{v:.0f}", ha='center')

    plt.tight_layout()
    plt.savefig("tsp_cost_comparison.png", dpi=300)
    print("Plot saved as tsp_cost_comparison.png")
    plt.close()

def plot_routes_academic(q_indices, q_cost, hc_route, hc_cost):

    plt.figure(figsize=(8,6))

    # Port coordinates
    lats = [PORTS[name][0] for name in PORT_LIST]
    lons = [PORTS[name][1] for name in PORT_LIST]

    # Plot ports
    plt.scatter(lons, lats, color='black', s=50, zorder=3)

    for i, name in enumerate(PORT_LIST):
        plt.text(lons[i]+0.2, lats[i]+0.2, name, fontsize=8)

    # Quantum route
    q_lons = [PORTS[PORT_LIST[i]][1] for i in q_indices + [q_indices[0]]]
    q_lats = [PORTS[PORT_LIST[i]][0] for i in q_indices + [q_indices[0]]]

    plt.plot(q_lons, q_lats, color='green', linewidth=2.5,
             label=f'Quantum Greedy ({q_cost:.0f} km)')

    # Hill climbing route
    hc_lons = [PORTS[PORT_LIST[i]][1] for i in hc_route + [hc_route[0]]]
    hc_lats = [PORTS[PORT_LIST[i]][0] for i in hc_route + [hc_route[0]]]

    plt.plot(hc_lons, hc_lats, color='blue', linestyle='--',
             linewidth=2, label=f'Hill Climbing ({hc_cost:.0f} km)')

    plt.title("Maritime TSP Route Comparison")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tsp_routes_academic.png", dpi=300)
    print("Plot saved as tsp_routes_academic.png")
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

def main(start_idx=0, layers=3, steps=300):

    random.seed(42)
    np.random.seed(42)

    # --- Classical Baselines ---
    print("Running classical baselines...")
    _, best_cost_rand = random_search()
    print(f"Best random cost: {best_cost_rand:.0f} km")

    current_route, current_cost = hill_climbing(list(range(N_PORTS)))
    print(f"Hill-climbing final cost: {current_cost:.0f} km\n")

    print("Running LKH-3...")
    lkh_route, lkh_cost = run_lkh(D, PORT_LIST, max_trials=100, pop_size=5)
    if lkh_route is not None:
        print(f"LKH-3 tour cost: {lkh_cost:,.0f} km")
        print("LKH tour: " + " → ".join([PORT_LIST[i] for i in lkh_route] + [PORT_LIST[lkh_route[0]]]))
    else:
        print("LKH-3 failed — skipping")

    # --- Main Quantum Run ---
    print("\nRunning quantum-assisted greedy...")
    q_tour, q_cost, q_indices = build_quantum_greedy_tour_ideal(
        start_idx=start_idx, layers=layers,
        steps_per_decision=steps
    )
    print("\nQuantum greedy tour:")
    print(" → ".join(q_tour))
    print(f"Quantum tour cost: {q_cost:.0f} km")
    print(f"Approximation ratio vs hill-climbing: {q_cost / current_cost:.3f}")
    plot_routes_academic(q_indices, q_cost, current_route, current_cost)
    plot_costs(q_cost, current_cost, lkh_cost)

    # --- Ablation Study ---
    print("\n=== Ablation Study: QAOA Layers & Alpha ===")
    ablation_results = []
    for l in [1, 2, 3, 4]:
        for alpha_val in [0.0, 0.3, 0.5, 0.7]:
            print(f"\nAblation → layers={l}, alpha={alpha_val}")
            _, q_cost_ab, _ = build_quantum_greedy_tour_ideal(
                start_idx=start_idx, layers=l,
                steps_per_decision=150, alpha=alpha_val
            )
            ablation_results.append((l, alpha_val, q_cost_ab))
            print(f"  → cost: {q_cost_ab:,.0f} km")

    print("\nAblation Summary:")
    print("layers | alpha | cost (km)")
    print("-" * 35)
    for l, a, c in sorted(ablation_results, key=lambda x: (x[0], x[1])):
        print(f"{l:6} | {a:5} | {c:,.0f}")

    # --- Statistical Evaluation ---
    NUM_EXPERIMENTS = 8
    seeds = [42 + i * 13 for i in range(NUM_EXPERIMENTS)]
    stats = {'qa_greedy_mumbai': [], 'nearest_neighbor_mumbai': [], 'qa_greedy_singapore': []}

    for exp_id, seed in enumerate(seeds):
        print(f"\n=== Experiment {exp_id+1}/{NUM_EXPERIMENTS} | Seed {seed} ===")
        random.seed(seed)
        np.random.seed(seed)
        start_time = time.time()

        _, q_cost_mum, _ = build_quantum_greedy_tour_ideal(
            start_idx=start_idx, layers=4,
            steps_per_decision=300
        )
        stats['qa_greedy_mumbai'].append(q_cost_mum)

        _, nn_cost = nearest_neighbor_tour(start_idx)
        stats['nearest_neighbor_mumbai'].append(nn_cost)

        print("Running QA-greedy Singapore...")
        _, q_cost_sin, _ = build_quantum_greedy_tour_ideal(
            start_idx=10, layers=6,
            steps_per_decision=300
        )
        stats['qa_greedy_singapore'].append(q_cost_sin)

        print(f"Experiment {exp_id+1} finished in {(time.time()-start_time)/60:.1f} minutes")

    print("\n=== STATISTICAL SUMMARY (8 runs) ===")
    for variant, costs in stats.items():
        if costs:
            print(f"{variant:25} → Mean {np.mean(costs):,.0f} ± {np.std(costs):,.0f} km "
                  f"(values: {[f'{c:,.0f}' for c in costs]})")

    # --- Noise Sensitivity Analysis ---
    print("\n=== Noise Sensitivity Analysis ===")
    noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05]
    noise_results = []
    noise_seeds = [42, 55, 77, 101]

    for noise_level in noise_levels:
        costs_at_noise = []
        for seed in noise_seeds:
            random.seed(seed)
            np.random.seed(seed)
            _, cost, _ = build_quantum_greedy_tour_noisy(
                start_idx=0, layers=3,
                steps_per_decision=150,
                alpha=0.5,
                noise_level=noise_level
            )
            costs_at_noise.append(cost)
        mean_cost = np.mean(costs_at_noise)
        std_cost = np.std(costs_at_noise)
        denom = lkh_cost if lkh_cost != float('inf') else current_cost
        ratio = mean_cost / denom if denom != 0 else float('nan')
        noise_results.append((noise_level, mean_cost, std_cost, ratio))
        print(f"  noise={noise_level:.3f} → mean {mean_cost:,.0f} ± {std_cost:,.0f} km | ratio={ratio:.3f}")

    print("\nNoise Sensitivity Summary:")
    print(f"{'Noise Level':<14} {'Mean Cost (km)':<18} {'Std Dev':<12} {'Approx Ratio'}")
    print("-" * 58)
    for nl, mc, sc, r in noise_results:
        print(f"{nl:<14.3f} {mc:<18,.0f} {sc:<12,.0f} {r:.3f}")

    # --- Qubit Cap Sensitivity Analysis ---
    print("\n=== Qubit Cap Sensitivity Analysis ===")
    qubit_caps = [4, 6, 8]
    qubit_results = []
    qubit_seeds = [42, 55, 77]

    def build_quantum_greedy_tour_qcap(
        start_idx=0, layers=3, steps_per_decision=80,
        alpha=0.5, max_qubits=8
    ):
        unvisited = set(range(N_PORTS))
        unvisited.remove(start_idx)
        tour = [start_idx]
        current = start_idx
        penalty = 10.0 * np.max(D)
        print(f"\nStarting from {PORT_LIST[start_idx]} (MAX_QUBITS={max_qubits})")
        quantum_decisions = 0
        classical_decisions = 0

        while unvisited:
            k = len(unvisited)

            if k > max_qubits:
                next_idx = min(unvisited, key=lambda x: D[current, x])
                print(f" {k:2d} ports left → chose {PORT_LIST[next_idx]} (classical fallback)")
                tour.append(next_idx)
                unvisited.remove(next_idx)
                current = next_idx
                classical_decisions += 1
                continue

            rem = list(unvisited)
            effective_costs = []
            for j in rem:
                future = min(D[j, x] for x in unvisited if x != j) if len(unvisited) > 1 else 0
                effective_costs.append(D[current, j] + alpha * future)

            coeffs, ops = [], []
            for i, d in enumerate(effective_costs):
                coeffs += [d / 2, -d / 2]
                ops += [qml.Identity(i), qml.PauliZ(i)]
            for i in range(k):
                for j in range(i + 1, k):
                    coeffs.append(penalty / 4)
                    ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
            H_cost = qml.Hamiltonian(coeffs, ops)

            mixer_coeffs, mixer_ops = [], []
            for i in range(k):
                for j in range(i + 1, k):
                    mixer_coeffs += [1.0, 1.0]
                    mixer_ops += [
                        qml.PauliX(i) @ qml.PauliX(j),
                        qml.PauliY(i) @ qml.PauliY(j)
                    ]
            H_mixer = qml.Hamiltonian(mixer_coeffs, mixer_ops)

            dev = qml.device("lightning.qubit", wires=k)

            @qml.qnode(dev)
            def energy_circuit(g, b):
                for i in range(k):
                    qml.Hadamard(i)
                for gg, bb in zip(g, b):
                    qml.qaoa.cost_layer(gg, H_cost)
                    qml.qaoa.mixer_layer(bb, H_mixer)

                return qml.expval(H_cost)

            @qml.qnode(dev)
            def prob_circuit(g, b):
                for i in range(k):
                    qml.Hadamard(i)
                for gg, bb in zip(g, b):
                    qml.qaoa.cost_layer(gg, H_cost)
                    qml.qaoa.mixer_layer(bb, H_mixer)

                return qml.probs(wires=range(k))

            g = pnp.random.uniform(0.05, 0.2, layers, requires_grad=True)
            b = pnp.random.uniform(1.0, 1.5, layers, requires_grad=True)
            opt = NesterovMomentumOptimizer(0.03)
            prev_energy = float("inf")
            stable = 0

            print(f" {k:2d} ports left → optimizing ({layers} layers, max {steps_per_decision} steps)... ", end="", flush=True)

            for step in range(steps_per_decision):
                g, b = opt.step(lambda x, y: energy_circuit(x, y), g, b)
                energy = energy_circuit(g, b)
                if step % 30 == 0 and step > 0:
                     print(f"{step} ", end="", flush=True)
                if abs(energy - prev_energy) < 1e-2:
                    stable += 1
                    if stable > 20:
                        break
                else:
                    stable = 0
                prev_energy = energy

            print("done", flush=True)

            probs = prob_circuit(g, b)
            valid_states, weights = [], []
            for s in range(2**k):
                bs = format(s, f"0{k}b")
                if bs.count("1") == 1:
                    valid_states.append(bs)
                    weights.append(probs[s])

            if sum(weights) > 0:
                chosen = random.choices(valid_states, weights=weights)[0]
                next_idx = rem[chosen.index("1")]
                prob_str = f"quantum prob: {max(weights):.4f}"
            else:
                next_idx = min(unvisited, key=lambda x: D[current, x])
                prob_str = "fallback (nearest neighbor)"
            print(f" → chose {PORT_LIST[next_idx]} ({prob_str})", flush=True)

            tour.append(next_idx)
            unvisited.remove(next_idx)
            current = next_idx
            quantum_decisions += 1

        tour.append(start_idx)
        total_decisions = quantum_decisions + classical_decisions
        quantum_ratio = quantum_decisions / total_decisions if total_decisions > 0 else 0

        return tour[:-1], route_cost(tour), quantum_decisions, classical_decisions, quantum_ratio

    for max_qubits in qubit_caps:
        costs_at_cap = []
        q_decisions_list = []
        c_decisions_list = []
        q_ratio_list = []

        print(f"\n  Testing MAX_QUBITS = {max_qubits}...")

        for seed in qubit_seeds:
            random.seed(seed)
            np.random.seed(seed)
            _, cost, q_dec, c_dec, q_ratio = build_quantum_greedy_tour_qcap(
                start_idx=0, layers=3,
                steps_per_decision=80,
                alpha=0.5,
                max_qubits=max_qubits
            )
            costs_at_cap.append(cost)
            q_decisions_list.append(q_dec)
            c_decisions_list.append(c_dec)
            q_ratio_list.append(q_ratio)
            print(f"    seed={seed} → cost={cost:,.0f} km | "
                  f"quantum decisions={q_dec} | classical decisions={c_dec} | "
                  f"quantum influence={q_ratio*100:.1f}%")

        mean_cost = np.mean(costs_at_cap)
        std_cost = np.std(costs_at_cap)
        denom = lkh_cost if lkh_cost != float('inf') else current_cost
        ratio = mean_cost / denom if denom != 0 else float('nan')
        mean_qdec = np.mean(q_decisions_list)
        mean_cdec = np.mean(c_decisions_list)
        mean_qratio = np.mean(q_ratio_list)

        qubit_results.append((
            max_qubits, mean_cost, std_cost, ratio,
            mean_qdec, mean_cdec, mean_qratio
        ))

    print("\nQubit Cap Sensitivity Summary:")
    print(f"{'MAX_QUBITS':<12} {'Mean Cost':<14} {'Std Dev':<10} "
          f"{'Ratio':<8} {'Q-Decisions':<13} {'C-Decisions':<13} {'Q-Influence%'}")
    print("-" * 80)
    for mq, mc, sc, r, qd, cd, qr in qubit_results:
        print(f"{mq:<12} {mc:<14,.0f} {sc:<10,.0f} "
              f"{r:<8.3f} {qd:<13.1f} {cd:<13.1f} {qr*100:.1f}%")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maritime TSP - Quantum-assisted Greedy")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)

    if "ipykernel" in sys.modules or "IPython" in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    main(args.start_idx, args.layers, args.steps)




