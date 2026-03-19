import yfinance as yf
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch 
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

# 1. Data 
tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'XLK', 'XBI', 'BTC-USD'] # Macro ETF's
data = yf.download(tickers, start='2022-01-01', end='2026-01-01')['Close']
all_returns = data.pct_change().dropna()

train_returns = all_returns.loc[:'2024-12-31']
test_returns = all_returns.loc['2025-01-01':]

# 2. Correlation and Distance Matrix
lw = LedoitWolf().fit(train_returns)
shrunk_cov = pd.DataFrame(lw.covariance_, index=tickers, columns=tickers)
shrunk_corr = train_returns.corr() 
dist = np.sqrt(0.5 * (1 - shrunk_corr))

print(f"\nCondition Number of Covariance Matrix: {np.linalg.cond(shrunk_cov):.2f}")

# 3. Single Linkage and Dendogram (Group Assets via the tree)
linkage = sch.linkage(dist, 'single')

plt.figure(figsize=(10, 6))
sch.dendrogram(linkage, labels=tickers)
plt.title('HRP Market Topology: Asset Clustering Tree')
plt.ylabel('Distance')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()

# 4. Quasi-Diagonalization (Flatten the tree)
def quasi_diag(link):
    return sch.to_tree(link, rd=False).pre_order(lambda x: x.id)

sort_ix = quasi_diag(linkage)
sorted_tickers = train_returns.columns[sort_ix].tolist()

# 5. Recursive Bisection Engine 
def cluster_var(cov, cluster_items):
    cluster_cov = cov.loc[cluster_items, cluster_items]
    ivp = 1.0 / (np.diag(cluster_cov) + 1e-9)
    weights = ivp / ivp.sum()
    return np.dot(np.dot(weights, cluster_cov), weights)

def rec_bisection(cov, sort_ix):
    weights = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]

    while len(clusters) > 0:
        clusters = [c[j:k] for c in clusters for j, k in ((0, len(c)//2), (len(c)//2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c_L, c_R = clusters[i], clusters[i+1]
            v_L = cluster_var(cov, c_L)
            v_R = cluster_var(cov, c_R) 
            alpha = 1 - v_L / (v_L + v_R)
            weights[c_L] *= alpha
            weights[c_R] *= (1 - alpha)
    return weights

# 6. Execution on training data
hrp_weights = rec_bisection(shrunk_cov * 252, sorted_tickers)

print("\n--- Final HRP Weights ---")
for ticker, weight in hrp_weights.sort_values(ascending=False).items():
    print(f"{ticker:8}: {weight:.2%}")

# 7. Return Calculation
hrp_returns = (test_returns * hrp_weights).sum(axis=1)
ew_returns = (test_returns * (1/len(tickers))).sum(axis=1)

# 8. Evaluation
def get_stats(ret, name="Portfolio"):
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / ann_vol
    cum_ret = (1 + ret).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    print(f"\n--- {name} (Out-of-Sample) ---")
    print(f"Annualized Vol: {ann_vol:.2%}")
    print(f"Sharpe Ratio:   {sharpe:.2f}")
    print(f"Max Drawdown:   {max_dd:.2%}")
    return cum_ret

# Comparison
hrp_cum = get_stats(hrp_returns, "HRP")
ew_cum = get_stats(ew_returns, "Equal Weight")

# Plots
plt.figure(figsize=(12,6))
plt.plot(hrp_cum, label='HRP Portfolio', color='royalblue', linewidth=2)
plt.plot(ew_cum, label='Equal Weight (1/N)', color='darkorange', linestyle='--')
plt.title("HRP vs Naive Diversification: Out-of-Sample Performance")
plt.ylabel("Cumulative Growth")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

hrp_weights.sort_values().plot(kind='barh', color='skyblue', edgecolor='black', title='Final HRP Allocation')
plt.xlabel('Portfolio Weight')
plt.tight_layout()
plt.show()
