import sys

# Redirect stdout/stderr to log file
log_file = open('experiment.log', 'w')
sys.stdout = log_file
sys.stderr = log_file

print("Starting experiment script...")

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy import stats
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import networkx as nx
import warnings
import traceback

warnings.filterwarnings("ignore")

print("Imports successful.")

# --- Constants ---
WINDOW_SIZE = 120  # Lookback window
REBALANCE_FREQ = 20  # Rebalance every 20 days
RISK_AVERSION = 1.0
TRANSACTION_COST = 0.001  # 10 bps

def load_data(filepath):
    print("Loading data...")
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Pivot to get (Date x Symbol) matrix of Returns
        returns = df.pivot(index='Date', columns='Symbol', values='Return')
        
        # Fill missing values (forward fill then backward fill)
        returns = returns.ffill().bfill()
        
        # Drop the first row if it's NaN (due to return calculation)
        returns = returns.dropna()
        
        print(f"Data loaded: {returns.shape[0]} days, {returns.shape[1]} assets")
        return returns
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        raise

# --- Helper Functions ---

def benjamini_hochberg(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg FDR correction.
    Returns boolean mask of significant hypotheses.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Critical values: (k/n) * alpha
    critical_vals = (np.arange(1, n + 1) / n) * alpha
    
    # Find largest k such that p_(k) <= (k/n)*alpha
    below_threshold = sorted_p <= critical_vals
    if not np.any(below_threshold):
        return np.zeros(n, dtype=bool)
        
    k_max = np.max(np.where(below_threshold)[0])
    significant_sorted = np.zeros(n, dtype=bool)
    significant_sorted[:k_max+1] = True
    
    # Map back to original indices
    significant = np.zeros(n, dtype=bool)
    significant[sorted_indices] = significant_sorted
    return significant

def get_kendall_graph(returns_window, alpha=0.05):
    """
    Compute Kendall-Tau correlation and apply FDR filtering.
    Returns Adjacency matrix (binary).
    """
    # Compute Kendall correlation matrix
    corr_matrix = returns_window.corr(method='kendall')
    
    # Calculate p-values. 
    T = returns_window.shape[0]
    tau = corr_matrix.values
    
    # Standard error for Kendall Tau under null hypothesis of independence
    sigma = np.sqrt(2 * (2 * T + 5) / (9 * T * (T - 1)))
    # Avoid division by zero if T is small (unlikely)
    if sigma == 0: sigma = 1e-6
    
    z_scores = tau / sigma
    
    # Two-sided p-values
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    
    # Flatten for FDR
    upper_tri_indices = np.triu_indices_from(p_values, k=1)
    p_vec = p_values[upper_tri_indices]
    
    # Apply BH
    sig_vec = benjamini_hochberg(p_vec, alpha)
    
    # Reconstruct Adjacency
    adj = np.eye(returns_window.shape[1]) # Self-loops
    adj[upper_tri_indices] = sig_vec.astype(int)
    adj = adj + adj.T - np.diag(np.diag(adj)) # Symmetrize
    
    # Ensure diagonal is 1
    np.fill_diagonal(adj, 1)
    
    return adj

def sgc_features(X, adj, K=2):
    """
    Compute Simple Graph Convolution features.
    H = (D^-0.5 A D^-0.5)^K X
    """
    # Normalize Adjacency
    D_vec = np.sum(adj, axis=1)
    D_vec[D_vec == 0] = 1 # Avoid div by zero
    D = np.diag(D_vec)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D_vec))
    
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    
    # Power K
    filter_op = np.linalg.matrix_power(norm_adj, K)
    
    # Apply to features X
    H = filter_op @ X
    return H

# --- Strategy Classes ---

class Strategy:
    def rebalance(self, returns_history):
        raise NotImplementedError

class EqualWeight(Strategy):
    def rebalance(self, returns_history):
        N = returns_history.shape[1]
        return np.ones(N) / N

class MinVariance(Strategy):
    def rebalance(self, returns_history):
        # Ledoit-Wolf shrinkage for robustness
        try:
            lw = LedoitWolf()
            lw.fit(returns_history)
            cov = lw.covariance_
            
            # Min Var Optimization
            N = cov.shape[0]
            w = cp.Variable(N)
            risk = cp.quad_form(w, cov)
            prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1, w >= 0])
            prob.solve()
            
            if w.value is None:
                return np.ones(N) / N
            return np.maximum(w.value, 0) / np.sum(np.maximum(w.value, 0))
        except Exception as e:
            print(f"MinVar Error: {e}")
            return np.ones(returns_history.shape[1]) / returns_history.shape[1]

class Momentum(Strategy):
    def rebalance(self, returns_history):
        # Simple momentum: return over last window
        moms = returns_history.mean(axis=0)
        # Top 30%
        N = len(moms)
        k = int(0.3 * N)
        if k < 1: k = 1
        
        top_indices = np.argsort(moms)[-k:]
        w = np.zeros(N)
        w[top_indices] = 1.0 / k
        return w

class AGESH(Strategy):
    def __init__(self, risk_aversion=1.0):
        self.gamma = risk_aversion
        
    def rebalance(self, returns_history):
        try:
            T, N = returns_history.shape
            
            # 1. Graph Construction
            adj = get_kendall_graph(returns_history)
            
            # 2. Features (Mean Return, Volatility)
            # Shape: (N, 2)
            mu = returns_history.mean(axis=0).values
            vol = returns_history.std(axis=0).values
            X = np.stack([mu, vol], axis=1)
            
            # 3. SGC Smoothing
            # Z: Smoothed features
            Z = sgc_features(X, adj, K=2)
            
            # 4. Masked Robust Regression
            betas = np.zeros((N, N))
            sigmas_sq = np.zeros(N)
            
            # Pre-standardize returns for regression stability
            scaler = StandardScaler()
            norm_returns = scaler.fit_transform(returns_history)
            
            for i in range(N):
                neighbors = np.where(adj[i] == 1)[0]
                neighbors = neighbors[neighbors != i] # Exclude self from regressors
                
                # Regress on neighbors
                if len(neighbors) > 0:
                    y = norm_returns[:, i]
                    X_neigh = norm_returns[:, neighbors]
                    
                    # Robust Regression with stronger regularization to prevent overfitting
                    # Increased alpha to 0.1, l1_ratio to 0.1 (mostly Ridge for stability)
                    model = SGDRegressor(loss='huber', penalty='elasticnet', alpha=0.1, l1_ratio=0.1, max_iter=2000, tol=1e-4, random_state=42)
                    model.fit(X_neigh, y)
                    
                    coefs = model.coef_
                    betas[i, neighbors] = coefs
                    
                    # Residual variance
                    preds = model.predict(X_neigh)
                    residuals = y - preds
                    # Add simple variance prior (shrinkage)
                    # sigma^2 = 0.5 * var(resid) + 0.5 * var(y)
                    # This prevents underestimation of risk
                    sigmas_sq[i] = 0.5 * np.var(residuals) + 0.5 * np.var(y)
                else:
                     sigmas_sq[i] = np.var(norm_returns[:, i])

                if sigmas_sq[i] < 1e-4: sigmas_sq[i] = 1e-4
            
            # 5. Construct Precision Matrix
            precision = np.zeros((N, N))
            for i in range(N):
                row = -betas[i]
                row[i] = 1.0
                precision[i] = row / sigmas_sq[i]
            
            # Re-scale precision matrix to match original scale
            # Since we standardized returns, this precision is for correlation/standardized data.
            # We need to scale back by 1/std_i * 1/std_j.
            # Cov_raw = D_std @ Cov_norm @ D_std
            # Prec_raw = D_std^-1 @ Prec_norm @ D_std^-1
            
            std_vec = returns_history.std(axis=0).values
            D_inv = np.diag(1.0 / (std_vec + 1e-6))
            precision = D_inv @ precision @ D_inv
                    
            # Symmetrize Precision
            precision = (precision + precision.T) / 2
            
            # Ensure PSD
            eigvals, eigvecs = np.linalg.eigh(precision)
            eigvals = np.maximum(eigvals, 1e-4)
            precision = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # 6. QP Optimization
            # Use Precision directly in objective? 
            # Minimize (gamma/2) * w.T @ Sigma @ w - mu.T @ w
            # If we have Precision Omega, Sigma = Omega^-1.
            # Calculating Sigma from Precision is better for QP solvers usually (standard form).
            sigma_est = np.linalg.inv(precision)
            
            # Expected Returns mu:
            # Blend Momentum and SGC
            mom = returns_history.mean(axis=0).values
            sgc_mu = Z[:, 0]
            # Normalize both
            mu_est = 0.5 * mom + 0.5 * sgc_mu
            
            w = cp.Variable(N)
            # Minimize (gamma/2) * w.T @ Sigma @ w - mu.T @ w
            obj = cp.Minimize((self.gamma / 2) * cp.quad_form(w, sigma_est) - mu_est @ w)
            constraints = [cp.sum(w) == 1, w >= 0]
            prob = cp.Problem(obj, constraints)
            # Let CVXPY choose the best available solver
            prob.solve() 
            
            if w.value is None or prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # print(f"QP Failed: {prob.status}")
                return np.ones(N) / N
            return np.maximum(w.value, 0) / np.sum(np.maximum(w.value, 0))
        except Exception as e:
            print(f"AGESH Error: {e}")
            traceback.print_exc()
            return np.ones(returns_history.shape[1]) / returns_history.shape[1]


# --- Backtest Engine ---

def run_backtest(returns, strategy):
    n_days, n_assets = returns.shape
    weights = np.zeros((n_days, n_assets))
    portfolio_returns = np.zeros(n_days)
    
    # Initial weights (Equal Weight)
    current_weights = np.ones(n_assets) / n_assets
    
    print(f"Running backtest for {strategy.__class__.__name__}...")
    
    # Limit backtest to last 500 days if it's too slow? 
    # Total days ~ 1250 (5 years). Should be fine.
    
    for t in range(WINDOW_SIZE, n_days):
        # Rebalance check
        if (t - WINDOW_SIZE) % REBALANCE_FREQ == 0:
            history = returns.iloc[t-WINDOW_SIZE:t]
            try:
                target_weights = strategy.rebalance(history)
                turnover = np.sum(np.abs(target_weights - current_weights))
                cost = turnover * TRANSACTION_COST
                current_weights = target_weights
            except Exception as e:
                print(f"Error at step {t}: {e}")
                cost = 0
        else:
            cost = 0
        
        # Calculate daily return
        day_return = np.sum(current_weights * returns.iloc[t])
        
        # If we rebalanced today, subtract cost
        if (t - WINDOW_SIZE) % REBALANCE_FREQ == 0:
             day_return -= cost
             
        portfolio_returns[t] = day_return
        weights[t] = current_weights
        
        if t % 100 == 0:
            print(f"Processed day {t}/{n_days}")
        
    return portfolio_returns[WINDOW_SIZE:]

def evaluate_performance(returns, name):
    ann_return = np.mean(returns) * 252
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    cum_returns = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    max_drawdown = np.min(drawdown)
    
    return {
        "Strategy": name,
        "Annual Return": ann_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }

# --- Main Execution ---

if __name__ == "__main__":
    try:
        data = load_data('sp500_raw_data.csv')
        
        # Use a smaller subset for debugging if needed, but 50 assets is ok.
        # data = data.iloc[:, :10]
        
        strategies = [
            EqualWeight(),
            MinVariance(),
            Momentum(),
            AGESH(risk_aversion=5.0) 
        ]
        
        results = []
        equity_curves = {}
        
        for strat in strategies:
            p_rets = run_backtest(data, strat)
            metrics = evaluate_performance(p_rets, strat.__class__.__name__)
            results.append(metrics)
            equity_curves[strat.__class__.__name__] = (1 + p_rets).cumprod()
            print(f"Finished {strat.__class__.__name__}: {metrics}")
            
        # Create Results DataFrame
        results_df = pd.DataFrame(results)
        print("\n--- Performance Results ---")
        print(results_df)
        results_df.to_csv('experiment_results.csv', index=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        for name, curve in equity_curves.items():
            plt.plot(curve, label=name)
        plt.title('Portfolio Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.savefig('equity_curves.png')
        print("Plot saved to equity_curves.png")
        
    except Exception as e:
        print(f"Fatal Error: {e}")
        traceback.print_exc()

print("Script completed.")
log_file.close()
