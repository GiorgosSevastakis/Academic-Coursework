# High Performance Parallel Computing – Assignment 5

This assignment contains the submission for **Assignment 5** from the *High Performance Parallel Computing* course (March 2024). The task involved parallelizing a **shallow water simulation** using OpenACC starting from a provided sequential implementation, and analyzing the performance through profiling and scaling experiments.

---

## Problem Summary

We began with a provided sequential shallow water model simulating wave propagation using a finite-difference scheme. The model evolves water height (`e`) and velocity fields (`u`, `v`) over a 2D grid with periodic boundary conditions implemented via ghost cells.

The assignment required us to:
- Implement OpenACC parallelism
- Investigate performance bottlenecks
- Tune the number of gangs for optimal GPU utilization
- Analyze both strong and weak scaling performance
- Relate scaling results to Amdahl’s and Gustafson’s laws

---

## Methods Used

- OpenACC directives to parallelize core compute loops
- Integration of ghost cell boundary logic directly into the compute kernel
- Profiling with Nsight Compute CLI (`nv-nsight-cu-cli --clock-control none`)
- Execution time benchmarking for increasing numbers of gangs
- Strong and weak scaling experiments
- Interpretation of scaling trends using **Amdahl's Law** and **Gustafson's Law**
- Visualization of results in Jupyter using `plots.ipynb`

---

## Key Observations

- The final OpenACC version achieved better performance than the initial parallel attempt.
- Moving boundary condition logic into the main kernel reduced synchronization and memory latency.
- Profiling identified memory transfers and frequent kernel launches as key performance bottlenecks.
- Strong scaling improved consistently up to 512 gangs but tapered at 1024 due to overhead saturation.
- Weak scaling results showed that execution time per iteration remained stable up to 128 gangs.


---

## Files

- `sw_sequential.cpp`: Provided sequential code
- `sw_parallel_oldv.cpp`: Initial parallel version with boundary conditions outside compute loop
- `sw_parallel.cpp`: Final optimized OpenACC version
- `Makefile`: Makefile for compilation and profiling automation (strong scaling only)
- `Strong.csv`: Recorded execution times for powers-of-two `--numgangs` values
- `Weak.csv`: Recorded execution times for increased workload and number of gangs
- `plots.ipynb`: Jupyter notebook for speedup and efficiency plots
- `ExternalFunctions.py`: Helper functions for formatting and output
- `report.pdf`: Report with detailed explanation of the assignment and results
- `instructions.pdf`: Assignment prompt
- `requirements.txt`: Python dependencies

---

## How to Run

### Compile all
```bash
make -f Makefile
```

### Just profile the parallelized versions
```bash
make -f Makefile profile
```

### Create the Strong.csv 
```bash
make -f Makefile timecsv

```
### Python script to create the strong and weak scaling plots

```bash
pip install -r requirements.txt
jupyter notebook plots.ipynb
```

---

## Authors

# Original sequential code

Professor Markus Jochum

# Optimized code

Antonio Maschio
Dimitrios Anastasiou
Georgios Sevastakis

March 2024
