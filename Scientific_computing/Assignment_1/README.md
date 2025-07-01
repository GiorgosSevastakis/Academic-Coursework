# Scientific Computing – Project 1

This folder contains the first project for the *Scientific Computing* course (September 2023). The assignment focuses on solving structured linear systems, analyzing numerical error and stability, and using least squares fitting to approximate the polarizability of a water molecule as a function of frequency.

---

## Problem Summary

The goal of this project is to compute the **frequency-dependent polarizability α(ω)** of a water molecule by solving a series of structured linear systems derived from quantum mechanical models. The full system is assembled from smaller matrices provided in `watermatrices.py`.

The project is divided into two main phases:

1. **Numerical Stability and LU Factorization**
   - Construct the matrices \(E\), \(S\), and \(z\), then solve \( (E - \omega S)x = z \)
   - Analyze the condition number and forward error bounds
   - Implement my own LU decomposition, forward and back substitution
   - Use these routines to calculate α(ω) = zᵗx for different frequencies

2. **Least Squares Approximation and Rational Fitting**
   - Implement QR decomposition using Householder transformations
   - Fit α(ω) with polynomial and rational approximations
   - Study numerical behavior near resonant frequencies and singularities
   - Extend approximation to broader frequency ranges including singular behavior

---

## Methods Used

- LU factorization and custom forward/back substitution
- Max-norm condition number analysis and error estimation
- QR decomposition with Householder reflections
- Linear least squares for polynomial and rational fitting
- Numerical investigation of resonant frequency behavior

---

## Files

- `code.ipynb`: Main notebook containing code and results for the entire project. It is structured as a report as well.
- `watermatrices.py`: Contains submatrices A, B, and vector y to construct the system
- `instructions.pdf`: Assignment prompt
- `requirements.txt`: Python dependencies

---

## How to Run

Make sure Python and Jupyter are installed. Then run:

```bash
python -m venv dummy_env
. ./dummy_venv/bin/activate
pip install -r requirements.txt
jupyter notebook code.ipynb
```

---

## Key Results

- Custom LU solver accurately recovers α(ω) and reveals error sensitivity around resonant frequencies
- Polynomial approximations of degree 4 and 6 achieve reasonable accuracy over bounded intervals
- Rational function fitting captures singularities and extends validity across wider frequency domains

---

## Author

Georgios Sevastakis

September 2023
