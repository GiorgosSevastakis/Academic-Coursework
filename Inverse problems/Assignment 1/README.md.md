# Inverse Problems – Assignment 1

This repository contains my submission for **Assignment 1** from the *Inverse Problems* course (October 2023). The task involves solving a **linear tomography problem** by analyzing the travel-time anomalies of acoustic waves passing through a subsurface medium.

---

## Problem Summary

We simulate the arrival times of waves from two sources detected by 12 receivers across a discretized 13×11 m^2 domain. A high-velocity inclusion in the medium causes deviations (anomalies) in travel time, which are used to infer the internal structure.

We solve both the **forward problem** (computing travel-time anomalies) and the **inverse problem** (reconstructing the slowness anomaly field).

---

## Methods Used

- Discretization of ray paths through a grid of 1x1 m^2 squares
- Analytical computation of travel-time anomalies
- Construction of a linear system `d = G·m`, where:
  - `d` is the data vector (arrival-time anomalies)
  - `G` is the path matrix (ray lengths per cell)
  - `m` is the slowness anomaly field
- **Tikhonov Regularization** to solve the ill-posed inverse problem under Gaussian noise
- Evaluation of resolution via delta function recovery

---

## Key Observations

- The forward model correctly captures the travel-time anomalies due to a known inclusion.
- The inverse solution using Tikhonov regularization can reconstruct the anomaly, but introduces artifacts along ray paths, especially where detector coverage is sparse.
- Adding more detectors or angular views would improve spatial resolution.

---

## Files

- `code`: The code/solution of he problem at hand
- `report.pdf`: Report describing the approach and findings
- `instructions.pdf`: Original assignment handout
- `requirements.txt`: Python dependencies

---

## How to Run

bash
pip install -r requirements.txt
jupyter notebook notebooks/Assignment_1_Inverse.ipynb

---

## Author

**Georgios Sevastakis**  
MSc Student, Computational Physics
October 2023
