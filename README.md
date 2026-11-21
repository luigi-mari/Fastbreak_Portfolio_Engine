# FastbreaK: Quantitative Portfolio  Engine

**FastbreaK** is an interactive, institutional-style quantitative portfolio backtesting and risk analytics web application built with **Python (Flask, NumPy, SciPy, scikit-learn, statsmodels, Plotly)**.

> **Live Demo:** [fastbreak-quant.onrender.com](https://fastbreak-quant.onrender.com)

---

## Key Quantitative Features

### 1. Global Out-of-Sample Walk-Forward Engine (Zero Look-Ahead)
Unlike standard retail backtesters that optimize weights on the full sample (introducing severe look-ahead bias), FastbreaK runs a **global Out-of-Sample Walk-Forward** engine (`3Y` or `5Y` rolling fit):
- At every rebalance date, each estimator fits **strictly on trailing historical returns** and holds (with daily drift) until the next rebalance.
- All KPIs, Quant Score, Rolling metrics, Tearsheet, Risk Attribution, Robustness tests, Head-to-Head comparisons, and Excel exports reflect the **exact same unbiased OOS series**.
- Actionable target weights shown in the allocation donuts correspond to the **latest trailing rolling window**.

### 2. Covariance Estimators & Allocation Schemes
Compares 6 long-only allocation schemes side-by-side (plus arbitrary custom weights):
- **Sample MLE (Markowitz)** — classical Gaussian mean-variance tangency portfolio
- **Global Minimum Variance (GMV)** — exact constrained long-only QP (`SLSQP`)
- **Hierarchical Risk Parity (HRP)** — López de Prado (2016) correlation-distance clustering, quasi-diagonalization & recursive bisection
- **Ledoit-Wolf Shrinkage** — optimal analytical shrinkage $\hat{\Sigma}_{\text{LW}} = (1-\hat{\delta})\hat{\Sigma} + \hat{\delta}F$
- **Marchenko-Pastur Denoising (RMT)** — spectral cleaning of the correlation matrix using the random-matrix upper edge $\lambda_+ = \sigma^2(1+\sqrt{q})^2$ with trace preservation
- **Elliptical MLE (Multivariate Student-$t$)** — IRLS fixed-point M-estimator weighting observations by squared Mahalanobis distance $d_t^2$:
  $$v(d_t^2) = \frac{\nu + N}{\nu + d_t^2}, \qquad \hat{\mu} = \frac{\sum_t v(d_t^2) r_t}{\sum_t v(d_t^2)}, \qquad \hat{\Sigma} = \frac{\nu}{\nu-2}\cdot\frac{1}{T}\sum_t v(d_t^2)(r_t-\hat{\mu})(r_t-\hat{\mu})^T$$

### 3. Risk Decomposition, PCA & Static Stress-Testing
- **Euler Risk Attribution**: Capital Weight vs Percentage Contribution to Risk (`PCR`), Marginal Contribution to Risk (`MCR`), Total Contribution to Risk (`TCR`, summing to $\sigma_p$), Portfolio Beta, and Diversification Ratio.
- **PCA Risk Directions**: Eigenvalue spectrum with Marchenko-Pastur noise edge $\lambda_+$, standardized Factor Loadings heatmap, Portfolio Variance Attribution across Principal Components, and **Effective Number of Bets (ENB)**.
- **Static Regime Stress-Test (Frozen Weights IS → OOS)**: Splits history into train/test blocks to evaluate regime resilience across estimators (`Max Sharpe` vs `GMV`) and track volatility forecast accuracy ($\sigma_{\text{OOS}}/\hat{\sigma}_{\text{IS}}$).

### 4. Multi-Engine Monte Carlo & Statistical Robustness
- **Forward Monte Carlo**: `Parametric Gaussian (Cholesky)`, `Circular Block Bootstrap (21d)`, and `Multivariate Student-t (ν=5)`.
- **Statistical Significance Validation**:
  - Classical & **Newey-West HAC** $t$-tests on excess returns and CAPM Jensen's Alpha
  - **Probabilistic Sharpe Ratio (PSR)**, **Deflated Sharpe Ratio (DSR)** (multiple-testing adjusted), and **Minimum Track Record Length (MinTRL)**
  - **Circular Block Bootstrap** (2,000 resamples) for 95% confidence intervals
  - **Random Long-Only Portfolio Null Test** (2,000 Dirichlet allocations) & expanding $t$-stat stability

### 5. Practical Research Tools
- **N+1 Asset Screener**: Evaluates multi-sector (`ticks.py`) or S&P 500 candidates on an exact apples-to-apples lookback window.
- **Head-to-Head Compare (A vs B)**: Benchmarks the active strategy against any custom portfolio with arbitrary assets and weights on common trading dates.
- **Multi-Sheet Excel Export (`.xlsx`)**: Exports Summary & KPIs, Composition & All-Scheme Weights, Daily Returns + Drawdowns + Individual Assets, and All-Schemes Metrics.

---

## Project Structure

```text
├── server.py              # Flask REST API, multi-user session cache, Light Plotly theme, Excel export
├── dashboard.html         # Light Minimal responsive frontend (Plotly.js)
├── quant_advanced.py      # Estimators (LW, MP, Elliptical MLE), HRP, PCA, OOS engine, Robustness tests
├── portfolio_last.py      # Core simulation, metrics & chart builders
├── portfolio_helpers.py   # Multi-currency FX conversion & N+1 screener
├── ticks.py               # Multi-sector equity universe
├── F.png                  # Brand logo & favicon
├── requirements.txt       # Python dependencies
├── render.yaml            # Render.com Blueprint configuration
├── Procfile               # Gunicorn process file
└── Dockerfile             # Optional container deployment
```

---

## Run Locally

```bash
pip install -r requirements.txt
python server.py
```

Open `http://127.0.0.1:5000` in your browser.

---

*Disclaimer: Market data is sourced via `yfinance` for educational and quantitative research purposes only. Not financial advice.*
