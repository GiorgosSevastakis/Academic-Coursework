# High Performance Parallel Computing - Assignment 1

## Overview

This assignment involves simulating the dynamics of the **SIR epidemiological model** using C++. The model tracks the progression of a disease through a population, categorizing individuals as **Susceptible (S)**, **Infected (I)**, or **Recovered (R)**.

We implement a **forward Euler method** to numerically integrate the governing differential equations over time, and analyze the outcomes through a Python plotting script.

## The SIR Model

The SIR model uses the following differential equations:

```
dS/dt = -β * I * S / N  
dI/dt = β * I * S / N - γ * I  
dR/dt = γ * I
```

Where:
- **S(t)**: Number of susceptible individuals  
- **I(t)**: Number of infected individuals  
- **R(t)**: Number of recovered individuals  
- **N**: Total population  
- **β (beta)**: Transmission rate (1/5 per day)  
- **γ (gamma)**: Recovery rate (1/10 per day)

Initial condition:  
- `S(0) = N - 1`, `I(0) = 1`, `R(0) = 0`  
- `N = 1000` individuals  
- Simulated over 300 days with a time step `dt = 0.1`

## Files

- `code.cpp`: C++ implementation of the SIR model using the forward Euler method. Outputs results to `sir_output.txt`.
- `plot_script.ipynb`: Python notebook to visualize the evolution of S, I, R over time.
- `report.pdf`: Detailed description of the assignment, implementation, results, and analysis.
- `instructions.pdf`: Original assignment prompt.

## How to Run

### Compile and Execute the C++ Code

```bash
g++ code.cpp -o sir_model
./sir_model
```

This generates the `sir_output.txt` file with simulation data.

### Plot the Results

Open and run the `plot_script.ipynb` Jupyter notebook. It will read `sir_output.txt` and generate plots for S, I, and R.

## Key Findings

- The infection peaks around day 70.
- The epidemic ends after ~160 days.
- To prevent an outbreak, at least **500 individuals need to be vaccinated** (i.e., `S(0) < 500`).

## Reflections

- **Time step**: `dt = 0.1` is chosen based on the inverse time scales of β and γ, ensuring numerical stability and accuracy.
- **Uncertainty in β and γ**: In reality, these vary over time. One could incorporate time-dependent or stochastic parameters in more advanced models.
- **Validation**: Results are tested through:
  1. Comparison with analytical or higher-order numerical methods
  2. Conservation of total population (`S + I + R = N`)
  3. Boundary tests (`I(0) = 0`, `β = 0`, `γ = 0`)
  4. Convergence checks by varying `dt`
  5. Logical consistency with expected epidemic dynamics

## Authors

- António Maschio  
- Dimitrios Anastasiou  
- Georgios Sevastakis

## License

This repository is part of a coursework for educational purposes.
