# Hierarchical-Risk-Parity-HRP-Port_Optimization
Implements the Hierarchical Risk Parity (HRP) algorithm, addressing the "Mean-Variance Curse" via clustering (unsupervised learning) to build a robust portfolio that does not require the inversion of a covariance matrix.

# Strategy
Unlike traditional Markowitz optimization, which is numerically unstable and highly sensitive to noise, this implementation follows a three-step process:
1. Tree Clustering: Grouping assets into a hierarchical structure based on correlation distance.
2. Quasi-Diagonalization: Reorganizing the covariance matrix so that similar assets are placed together.
3. Recursive Bisection: Allocating capital by splitting the portfolio into branches and assigning weights based on inverse variance.

# Key Performance Metrics
The model is evaluated on a diverse ETF universe (Equities, Bonds, Gold, and Crypto) using an out-of-sample (OOS) backtest (2025-2026).

* Annualized Volatility: Measures the realized risk of the portfolio. HRP seeks to minimize this by neutralizing cluster-specific correlations.
* Sharpe Ratio: The primary measure of risk-adjusted return.
* Maximum Drawdown (Max DD): Indicates the worst peak-to-trough decline, highlighting the portfolio's resilience during market stress.

# Technical Implementation
- Covariance Conditioning: Uses Ledoit-Wolf Shrinkage to improve the stability of the estimator.
- Clustering: Employs Single Linkage clustering to define the market topology.
- Data Source: Real-time market data via `yfinance`.

# Usage
Run the main script to generate the results.
