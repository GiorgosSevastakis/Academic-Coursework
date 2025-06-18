# High Performance Parallel Computing – Assignment 2

This folder contains my modified version of the original code for **Assignment 2** from the *High Performance Parallel Computing* course (February 2024). The task focuses on optimizing a **molecular dynamics simulation** of water molecules by restructuring data layouts and applying parallelism through vectorization and OpenMP.

---

## Problem Summary

We are given a simulation of a molecular system of **N water molecules**, each composed of 3 atoms (2 Hydrogen, 1 Oxygen). The atoms interact through:

- **Bond potentials**: Covalent O–H bonds
- **Angle potentials**: H–O–H angle deformation
- **Non-bonded potentials**: Lennard-Jones (LJ) and Coulombic interactions

The simulation is integrated over time using the **leap-frog integrator**.

---

## Tasks and Methods

### Task 1 – Struct-of-Arrays (SoA) Vectorization

- Rewrote the original Array-of-Structs (AoS) layout into a Struct-of-Arrays (SoA) format for better vectorization
- Updated all major functions (`UpdateBondForces`, `UpdateAngleForces`, `UpdateNonBondedForces`, `Evolve`) to operate over vectorized data
- Ensured identical output (including checksums) compared to the original implementation

### Task 2 – Profiling and Performance Analysis

- Used **gprof** to identify bottlenecks
- Found that `UpdateNonBondedForces` dominates runtime as molecule count increases (due to O(N²) complexity)
- Benchmarked performance for 2, 4, 16, and 128 molecules and explained performance scaling
- Compared SoA version with the original AoS: performance gains were limited due to cache behavior

### Task 3 – OpenMP SIMD Parallelization

- Inserted `#pragma omp simd` directives in key computational loops
- Used reductions and loop collapsing in `UpdateNonBondedForces` for efficient SIMD parallelism
- Verified correctness via checksum validation

---

## Files

- `Water_sequential.cpp`: Original sequential implementation using AoS layout
- `Water_vectorised.cpp`: Vectorized version using SoA layout
- `Water_vectorisedopenmp.cpp`: Final version with OpenMP SIMD pragmas
- `runscript.sh`: Automates compilation, execution, and profiling of all variants
- `report.pdf`: Detailed description of the assignment, implementation, results, and analysis
- `instructions.pdf`: Original assignment prompt from the course

---

## How to Run the Assignment

### Automated Benchmarking

Use the provided script to compile and run all code variants:

```bash
bash runscript.sh
```

This will:
- Compile all code versions with profiling flags
- Run simulations for multiple molecule counts
- Print and profile results with `gprof` for each variant (vectorized, sequential, OpenMP)

---

## Authors

### Original sequential code

Assistant Professor Weria Pezeshkian

### Vectorized code

António Maschio  
Dimitrios Anastasiou  
Georgios Sevastakis

February 2024
