# Adaptive Graph-Enhanced Sparse Hedging (AGESH)

This repository contains the implementation and manuscript for the **Adaptive Graph-Enhanced Sparse Hedging (AGESH)** framework, a differentiable portfolio optimization method designed for high-dimensional financial markets.

## Project Overview

AGESH integrates graph theory with robust statistics to solve the "Markowitz Enigma" (instability in portfolio optimization). Key features include:
- **Topological Denoising:** Uses Kendall-Tau correlation with Benjamini-Hochberg FDR filtering to construct a robust market graph.
- **Spectral Smoothing:** Applies Simple Graph Convolutions (SGC) to smooth asset features along the market topology.
- **Masked Robust Regression:** Restricts hedging to topological neighbors using Huber loss and ElasticNet regularization.
- **Differentiable Optimization:** Uses a QP layer to optimize Sharpe Ratio end-to-end.

## File Structure

### Code
- **`agesh_experiment.py`**: The core implementation script. It contains:
  - `get_kendall_graph`: Constructs the market graph.
  - `sgc_features`: Computes spectral graph features.
  - `AGESH`: The main strategy class implementing the algorithm.
  - Benchmarks: `EqualWeight`, `MinVariance`, `Momentum`.
  - Backtesting engine to evaluate performance.
- **`inspect_data.py`**: A helper script to quickly inspect the raw CSV data (symbols, date range).
- **`test_simple.py`**: Simple test script for verification.

### Data
- **`sp500_raw_data.csv`**: Historical return data for 50 S&P 500 assets (2020-2025). Format: `Date, Symbol, Return`.

### Outputs
- **`experiment_results.csv`**: Performance metrics (Annual Return, Sharpe Ratio, Max Drawdown) from the latest run.
- **`equity_curves.png`**: Cumulative return plots comparing AGESH with benchmarks.
- **`experiment.log`**: Execution log capturing console output from `agesh_experiment.py`.

## Setup and Usage

### Prerequisites
The project requires Python 3.x and the following libraries:
```bash
pip install numpy pandas cvxpy scikit-learn matplotlib networkx scipy
```

### Running the Experiment
To run the full backtest and generate results:
```bash
python agesh_experiment.py
```
This will:
1. Load data from `sp500_raw_data.csv`.
2. Run backtests for AGESH and benchmark strategies.
3. Save performance metrics to `experiment_results.csv`.
4. Generate the equity curve plot `equity_curves.png`.
5. Log progress to `experiment.log`.

## Results
The latest experiment (configured with `gamma=5.0`, `alpha=0.1`) demonstrated that AGESH achieves a **Sharpe Ratio of ~0.89**, significantly outperforming the Minimum Variance benchmark (0.44) and Equal Weight strategy.

## Citation
If you use this code or framework, please refer to the manuscript `v4.tex` for citation details.
