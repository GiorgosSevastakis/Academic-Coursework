# Inverse Problems – Assignment 2

This folder contains the second assignment for the *Inverse Problems* course (April 2024), focusing on the Bayesian inversion of magnetic field data using the Metropolis algorithm.

---

## Problem Summary

We aim to estimate the magnetization distribution \( m(x) \) of a magnetized plate from vertical magnetic field data measured 2 cm above its surface. Although the forward problem is linear, the prior is **non-Gaussian**, making traditional linear inversion techniques unsuitable. A **Bayesian approach** is used to sample the a posteriori distribution.

---

## Methods Used

- Discretization of the magnetized plate into 200 bands over 100 cm
- Formulation of the **forward model** using a Green's function-like kernel \( g_j(x) \)
- Construction of the matrix \( G \) and solution of \( d = Gm \)
- **Metropolis algorithm** to sample the a posteriori probability:
  - Perturbation of stripe magnetization
  - Addition/removal of stripe boundaries
  - Priors based on Gaussian (magnetization) and exponential (stripe width) distributions
- Posterior estimation of uncertainties from accepted samples

---

## Files

- `dataM.txt`: Observed data to solve the inverse problem
- `code.ipynb`: Code/solution containing the full implementation
- `ExternalFunctions.py`: Helper functions for formatting and output
- `report.pdf`: Report with detailed explanation of the assignment and results
- `instructions.pdf`: Assignment prompt
- `requirements.txt`: Python dependencies

---

## How to Run

```bash
python -m venv dummy_env
. ./dummy_venv/bin/activate
pip install -r requirements.txt
jupyter notebook code.ipynb
```

---

## Key Results

- The Metropolis algorithm successfully sampled the posterior magnetization distribution.
- Stripe width distribution followed the expected exponential prior.
- Posterior estimates closely matched observed data.
- Uncertainties were consistent with the prior standard deviation of 2.5 A/cm.

---

## Author

**Georgios Sevastakis**  

April 2024
