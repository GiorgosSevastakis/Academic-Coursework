# Inverse Problems – Assignment 3

This folder contains the third assignment for the *Inverse Problems* course (April 2024), which focuses on non-linear least squares inversion for spectral data analysis using Gaussian and Lorentzian models.

---

## Problem Summary

We are given a dataset representing Mössbauer spectroscopic measurements from Mars soil. The objective is to determine the **location (f)**, **area (A)**, and **width (c)** of multiple overlapping peaks in the spectrum by fitting both **Gaussian** and **Lorentzian** models using non-linear optimization.

---

## Methods Used

- Analytical derivation of model function and its derivatives for Gaussian and Lorentzian cases
- Construction of the Jacobian matrix \( G_k \) for each iteration
- Implementation of the **steepest descent algorithm** with adaptive step size \( ε_k \)
- Use of model covariance \( C_M \) and data covariance \( C_D \) matrices
- Comparison of model fits and convergence behavior

---

## Files

- `mars_soil.txt`: Data file with Mössbauer spectroscopy measurements
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

- The Lorentzian model provides a significantly better fit to the observed data than the Gaussian model.
- Gaussian peaks tend to underfit overlapping features, while Lorentzian peaks account better for long tails.
- The steepest descent method converged reliably with a well-chosen initial guess and adaptive step size.

---

## Author

**Georgios Sevastakis**  

April 2024
