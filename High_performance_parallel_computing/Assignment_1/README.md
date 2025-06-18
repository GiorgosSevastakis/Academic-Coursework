# High Performance Parallel Computing – Assignment 1

This folder contains my submission for **Assignment 1** from the *High Performance Parallel Computing* course (February 2024). The task involves solving the **SIR epidemiological model** numerically and analyzing disease spread in a closed population using C++.

---

## Problem Summary

We study the classical **SIR model**, which describes the time evolution of three compartments:
- **S(t)**: Susceptible individuals
- **I(t)**: Infected individuals
- **R(t)**: Recovered individuals

The dynamics are governed by the following ODEs:

```
dS/dt = -β * I * S / N  
dI/dt = β * I * S / N - γ * I  
dR/dt = γ * I
```

where:
- `β = 1/5` days⁻¹: infection rate
- `γ = 1/10` days⁻¹: recovery rate
- `N = 1000`: total population  
- Initial condition: `S(0)=999`, `I(0)=1`, `R(0)=0`

We simulate the epidemic over **300 days** using the **forward Euler method** and analyze the evolution of the three compartments.

---

## Methods Used

- Numerical integration with forward Euler method
- Time discretization: `dt = 0.1` days
- Storage and output of S, I, R values into text file
- Plotting of time evolution using Python/Matplotlib
- Estimation of **critical vaccination threshold** to prevent outbreak
- Sensitivity analysis of time step, model parameters, and validation checks

---

## Key Observations

- Epidemic peaks around day 70 and ends by day 160
- Vaccinating at least **500 individuals** prevents an outbreak (`S(0) < 500`)
- Time step `dt = 0.1` balances accuracy and computational efficiency
- β and γ may vary in practice; their uncertainty impacts predictions
- Multiple checks were used to validate the model (e.g., total population conservation)

---

## Files

- `code.cpp`: C++ implementation of SIR model
- `plot_script.ipynb`: Python script to visualize the results
- `report.pdf`: Detailed description of the assignment, implementation, results, and analysis
- `instructions.pdf`: Original assignment prompt from the course

---

## How to Run

### Compile and Run the Simulation

```bash
g++ code.cpp -o sir_model
./sir_model 
```

This generates `sir_output.txt`.

### Plot the Results

Open the notebook in Jupyter:

```bash
python -m venv dummy_env
. ./dummy_venv/bin/activate
pip install -r requirements.txt
jupyter notebook plot_script.ipynb
```

---

## Authors

António Maschio  
Dimitrios Anastasiou  
Georgios Sevastakis

February 2024
