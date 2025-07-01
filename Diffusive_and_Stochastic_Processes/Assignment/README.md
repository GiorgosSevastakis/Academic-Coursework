# Diffusive and Stochastic Processes – Assignment

This folder contains the programming assignment for the *Diffusive and Stochastic Processes* course (June 2024). The project focuses on simulating stochastic processes such as Brownian motion and epidemic spread using both the Euler method and the Gillespie algorithm.

---

## Problem Summary

The assignment is structured in three parts:

1. **Langevin Dynamics Simulation**  
   Simulate a Brownian particle under a non-linear potential using the **Euler method**. The stochastic differential equation (SDE) includes Gaussian white noise.

2. **Stochastic Epidemic Model (SIR-like)**  
   Use the **Gillespie algorithm** to simulate an epidemic model where infected individuals can spontaneously recover or transmit the infection. 

3. **SIR Model with Immunity**  
   Extend the previous model to include an immune class with recovery and loss of immunity. Track both infected and immune populations over time using an updated set of propensity functions in the Gillespie method.

---

## Methods Used

- Euler integration of Langevin equations (Brownian motion)
- Gillespie stochastic simulation algorithm (SSA) for reaction-based dynamics
- Computation of ensemble averages and variances
- Handling of non-linear drift and time-dependent noise
- Statistical analysis of steady-state properties

---

## Files

- `Exercise_1.ipynb`: Euler simulation of a Brownian particle in a quartic potential
- `Exercise_2.ipynb`: Gillespie simulation of a basic flu epidemic model
- `Exercise_3.ipynb`: Gillespie simulation of an epidemic with immune dynamics
- `report.pdf`: Detailed analysis and plots for all three exercises
- `instructions.pdf`: Assignment description
- `requirements.txt`: Python dependencies

---

## How to Run

Make sure Python and Jupyter are installed. Then run:

```bash
python -m venv dummy_env
. ./dummy_venv/bin/activate
pip install -r requirements.txt
jupyter notebook Exercise_1.ipynb
```

Repeat for `Exercise_2.ipynb` and `Exercise_3.ipynb` as needed.

---

## Key Results

- The ensemble mean and variance of the Langevin trajectories align with physical expectations under the specified potential.
- The epidemic model reaches a steady state with reproducible statistics; time-averaged and ensemble-averaged quantities agree.
- Introducing immunity adds rich dynamical behavior, such as decay and resurgence patterns, under stochastic noise.

---

## Author

**Georgios Sevastakis**  
June 2024
