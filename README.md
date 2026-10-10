# American Option Pricing with Randomized Neural Networks

Code accompanying Master's thesis in Applied Mathematics at Sapienza University of Rome (2025-2026).
The project studies Monte Carlo methods for multidimensional American and Bermudan options, with randomized neural networks (RaNNs) used as regression bases in primal and dual formulations of the optimal-stopping problem.

## Overview

A feasible exercise policy produces a lower bound for the option price. A suitable martingale produces an upper bound. The code compares classical polynomial or indicator bases with random features whose hidden weights and biases are sampled once and kept fixed. Fitting therefore remains a least-squares problem.

The repository implements:

| Method | Estimates | Basis variants |
|---|---|---|
| Longstaff-Schwartz Monte Carlo | lower bound | polynomial, RaNN |
| Haugh-Kogan | lower and upper bounds | polynomial, RaNN |
| Herrera et al. extension | lower and upper bounds | payoff-augmented RaNN |
| Alfonsi primal-dual | variance-reduced lower and dual upper bounds | local indicator, RaNN |


## Repository structure

```text
.
├── ALFONSI/               # Alfonsi primal-dual estimators
├── HK_and_Herrera/        # Haugh--Kogan and Herrera variants
├── LSMC/                  # Longstaff--Schwartz estimators
├── activations.py         # Activations for random features
├── basis.py               # Polynomial, indicator, and random bases
├── run_experiments.ipynb  # Run numerical experiments
├── payoff.py              # Multi-asset payoff functions
├── simulation_paths.py    # Independent multi-asset GBM simulation
├── standardization_ridge.py
└── requirements.txt       # Python dependencies
```

## Running experiments

Install the dependencies from the repository root:

```bash
python3 -m pip install -r requirements.txt
```

Open `run_experiments.ipynb` and select the parameters in the two configuration cells: one for LSMC, Haugh-Kogan, and Herrera, and one for Alfonsi.  
The notebook produces two detailed result tables and a compact combined comparison, each saved to CSV.


## References

- Longstaff, F. A. & Schwartz, E. S. (2001). *Valuing American options by simulation: a simple least-squares approach.* Review of Financial Studies.
- Rogers, L. C. G. (2002). *Monte Carlo valuation of American options.* Mathematical Finance.
- Haugh, M. B. & Kogan, L. (2004). *Pricing American options: a duality approach.* Operations Research.
- Herrera, C., Krach, F., Ruyssen, P. & Teichmann, J. (2024). *Optimal stopping via randomized neural networks.* Frontiers of Mathematical Finance.
- Alfonsi, A., Kebaier, A. & Lelong, J. (2025). *A pure dual approach for hedging Bermudan options.* Mathematical Finance.
- Alfonsi, A., Kebaier, A. & Lelong, J. (2026). *How can the dual martingale help solving the primal optimal stopping problem?* arXiv:2602.09950.

## Author

**Francesco Pontrelli** - M.S. in Applied Mathematics, Sapienza University of Rome.

Advisor: Prof.ssa Vittoria Silvestri. Co-advisors: Dr. Andrea Pallavicini and Prof.ssa Giulia Livieri.

