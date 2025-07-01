# Scientific Computing – Project 4 (Part 2)

This folder contains the second part of Project 4 for the *Scientific Computing* course (November 2023). The assignment focuses on numerically solving a coupled reaction-diffusion system using the finite difference method and a forward Euler to analyze pattern formation under varying conditions.

---

## Problem Summary

This project investigates a **reaction-diffusion system** that models pattern formation in biological and chemical systems. The model consists of two coupled nonlinear partial differential equations representing concentrations \( p(x, y, t) \) and \( q(x, y, t) \) over a 2D spatial domain. The equations describe both diffusion and local chemical interaction dynamics:

\[
\frac{\partial p}{\partial t} = D_p \nabla^2 p + p^2 q + C - (K + 1)p
\]
\[
\frac{\partial q}{\partial t} = D_q \nabla^2 q - p^2 q + Kp
\]

Neumann (no-flux) boundary conditions are applied along the domain edges. Simulations are performed on a 40×40 spatial grid and evolved in time up to \( t = 2000 \), using the **Euler method** for time integration and finite differences for the Laplacian.

---

## Methods Used

- Finite difference approximation for space
- Forward Euler method for time integration
- Neumann boundary conditions implemented via mirroring
- Contour plots for steady-state spatial pattern visualization

---

## Files

- `code.ipynb`: Updated notebook containing full implementation and results
- `HelperFunctions.py`: Contains the simulation class and plotting utilities
- `report.pdf`: Detailed explanation of the model, methods, and results
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

- Successfully simulated and visualized pattern formation in 2D for various values of the interaction parameter \( K \)
- Demonstrated stability and accuracy of the finite difference + Euler method scheme
- Observed sensitive dependence of final spatial patterns on initial conditions and mesh resolution
- Implementation avoids for-loops by using NumPy’s `roll()` for efficient Laplacian calculation

---

## Author

Georgios Sevastakis 

November 2023
