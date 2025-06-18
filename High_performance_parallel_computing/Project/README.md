# High-Performance Parallel Computing – CUDA Ising Model

This folder contains our final project for the *High-Performance Parallel Computing* course (March 2024), where we implement the 2D Ising model using the **Checkerboard Metropolis Algorithm** on an NVIDIA GPU with CUDA C.

---

## Problem Summary

The 2D Ising model simulates magnetic spin interactions on a lattice, where each spin takes values ±1 and interacts with its nearest neighbors. We implement the **Metropolis Monte Carlo method** using CUDA to parallelize updates efficiently, this time without the help of a sequential preexistent code to begin with.

Our implementation uses a **checkerboard decomposition**, allowing spin updates in parallel without conflicts. Each **CUDA thread is responsible for updating one spin**, maximizing the parallel use of GPU cores.

To validate the implementation, the simulation is run across a range of inverse temperature values (β = 1/T), with each β simulated for 3000 iterations. By saving the lattice state after each run, we visually confirm the expected phase transition from disordered to ordered states.

---

## Methods Used

- CUDA C implementation of the Metropolis algorithm
- Checkerboard update scheme with two separate sub-lattices
- GPU profiling using **Nsight Compute**
- Performance analysis using strong and weak scaling, and fits to **Ahmdahl’s** and **Gustafson’s laws**

---

## Files

- `kernel.cu`: CUDA implementation of the 2D Ising model with checkerboard pattern and temperature sweep
- `Makefile`: Instructions to compile the CUDA program
- `vizualize_pgms.py`: Python script to animate lattice snapshots saved as `.pgm` files
- `frames`: Folder containing the `.pgm` files
- `report.pdf`: Report with detailed explanation of the assignment and results
- `requirements.txt`: Python dependencies

---

## How to Run

### CUDA Simulation

Make sure you have a CUDA-capable GPU and NVIDIA’s CUDA Toolkit installed.

```bash
make run
```

The simulation runs the Ising model for multiple β values. For each value, it runs 3000 iterations and saves the final lattice configuration to a `.pgm` file.

### Visualize Output

To visualize the phase transition behavior:

```bash
python -m venv dummy_env
. ./dummy_venv/bin/activate
pip install -r requirements.txt
python vizualize_pgms.py
```

This will display a sequence of saved frames showing the system evolving from disordered to ordered states as β increases.

---

## Key Results

- One-thread-per-spin design enables high parallel efficiency, achieving ~1500 million lattice updates per second.
- GPU profiling confirms efficient usage of compute resources with room for further optimization (e.g., shared memory or more efficient memory congruency).
- Strong scaling shows optimal speedup up to 32 threads per block; weak scaling reveals overhead limitations.
- Visual inspection confirms expected phase transition behavior around critical temperature ($T_c \approx 2.38$), verifying implementation correctness.

---

## Authors

António Maschio,
Cyan,
Georgios Sevastakis,
Sebastian Dreizler

March 2024
