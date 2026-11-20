"""
FastbreaK — Advanced Quantitative Analytics Module
==================================================
Implements:
  1. Covariance & Mean Estimators:
     - Classical Gaussian MLE (sample)
     - Ledoit-Wolf Shrinkage
     - Marchenko-Pastur Random Matrix Theory Denoising
     - Elliptical MLE (Multivariate Student-t M-estimator via IRLS / Mahalanobis)
  2. Portfolio Optimizers (Long-only exact):
     - Global Minimum Variance (GMV)
     - Maximum Sharpe Ratio (Tangency)
     - Hierarchical Risk Parity (HRP, López de Prado 2016)
  3. Risk Decomposition & Factor Analysis:
     - Full Risk Attribution (Weight vs %TCTR, MCTR, TCTR/CCTR, Beta to Port, Diversification Ratio)
     - PCA Risk Directions (Scree + MP threshold, Factor Loadings Heatmap, Portfolio PC Exposure & ENB)
  4. In-Sample / Out-of-Sample Estimator Validation Engine
  5. Multi-Engine Monte Carlo Simulation (Gaussian Cholesky, Circular Block Bootstrap, Student-t)
  6. Statistical Robustness & Significance Analysis:
     - Classical & Newey-West HAC t-test on excess returns
     - CAPM Jensen's Alpha HAC t-test
     - Probabilistic Sharpe Ratio (PSR), Deflated Sharpe Ratio (DSR), Minimum Track Record Length (MinTRL)
     - Circular Block Bootstrap CIs & p-values
     - Random Long-Only Portfolio Null Test & Cumulative t-stat Stability
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── FastbreaK Design System ─────────────────────────────────────
C_BLUE   = "#0a84ff"
C_ORANGE = "#ff9f0a"
C_GREEN  = "#30d158"
C_RED    = "#ff453a"
C_PURPLE = "#bf5af2"
C_CYAN   = "#64d2ff"
C_YELLOW = "#ffd60a"
C_PINK   = "#ff375f"
C_TEAL   = "#5ac8fa"
C_DARK   = "#0a0a0f"
C_SURFACE= "#111118"
C_CARD   = "#13131c"
C_BORDER = "#1e1e2e"
C_TEXT   = "#e8e8f0"
C_MUTED  = "#555570"
TEMPLATE = "plotly_dark"


def _clean_psd(mat, eps=1e-10):
    """Symmetrize and project matrix to positive semi-definite."""
    m = 0.5 * (mat + mat.T)
    evals, evecs = np.linalg.eigh(m)
    evals = np.maximum(evals, eps)
    return (evecs * evals) @ evecs.T


def _cond_number(cov):
    evals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    evals = np.maximum(evals, 1e-15)
    return float(evals.max() / evals.min())


def _effective_n(w):
    w = np.asarray(w, dtype=float)
    s2 = float(np.sum(w ** 2))
    return float(1.0 / s2) if s2 > 0 else 0.0


# ═════════════════════════════════════════════════════════════════
# 1. COVARIANCE & MEAN ESTIMATORS
# ═════════════════════════════════════════════════════════════════

def estimate_sample_mle(returns_df, days=252):
    """Classical Gaussian MLE (sample mean & covariance)."""
    X = returns_df.values.astype(float)
    mu_d = np.mean(X, axis=0)
    cov_d = np.cov(X, rowvar=False, ddof=1)
    if cov_d.ndim == 0:
        cov_d = np.array([[float(cov_d)]])
    cov_d = _clean_psd(cov_d)
    return mu_d * days, cov_d * days, {
        "method": "Classical MLE",
        "cond_num": _cond_number(cov_d),
    }


def estimate_ledoit_wolf(returns_df, days=252):
    """Ledoit-Wolf analytical shrinkage covariance estimator."""
    X = returns_df.values.astype(float)
    mu_d = np.mean(X, axis=0)
    if X.shape[1] == 1:
        cov_d = np.array([[float(np.var(X, ddof=1))]])
        shrink = 0.0
    else:
        lw = LedoitWolf().fit(X)
        cov_d = _clean_psd(lw.covariance_)
        shrink = float(lw.shrinkage_)
    return mu_d * days, cov_d * days, {
        "method": "Ledoit-Wolf",
        "shrinkage": shrink,
        "cond_num": _cond_number(cov_d),
    }


def estimate_marchenko_pastur(returns_df, days=252):
    """Marchenko-Pastur Random Matrix Theory denoising of the correlation matrix.

    Steps:
      1. Compute sample correlation matrix C and its eigenvalues/eigenvectors.
      2. Compute aspect ratio q = N / T and iteratively fit noise bulk variance sigma^2:
           lambda_+ = sigma^2 * (1 + sqrt(q))^2
           sigma^2  = 1 - (1/N) * sum(lambda_i for lambda_i > lambda_+)
      3. Replace noise eigenvalues (lambda_i <= lambda_+) with their constant mean
         so trace(C_denoised) = N.
      4. Reconstruct C_denoised, rescale to unit diagonal, restore asset volatilities.
    """
    X = returns_df.values.astype(float)
    T, N = X.shape
    mu_d = np.mean(X, axis=0)
    std_d = np.std(X, axis=0, ddof=1)
    std_d = np.where(std_d <= 1e-12, 1e-12, std_d)

    if N == 1:
        cov_d = np.array([[float(std_d[0] ** 2)]])
        return mu_d * days, cov_d * days, {
            "method": "Marchenko-Pastur",
            "q": 1.0 / max(T, 1),
            "sigma2": 1.0,
            "lambda_plus": 1.0,
            "lambda_minus": 1.0,
            "n_signal": 1,
            "n_noise": 0,
            "raw_evals": [1.0],
            "denoised_evals": [1.0],
            "cond_num": 1.0,
        }

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    corr = 0.5 * (corr + corr.T)

    evals, evecs = np.linalg.eigh(corr)
    # Sort descending
    idx = np.argsort(evals)[::-1]
    evals = np.maximum(evals[idx], 1e-12)
    evecs = evecs[:, idx]

    q = float(N) / float(max(T, 1))
    # Iterative bulk noise variance fitting (López de Prado)
    sigma2 = 1.0
    for _ in range(15):
        lam_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2
        signal_mask = evals > lam_plus
        n_sig = int(signal_mask.sum())
        if n_sig == 0 or n_sig == N:
            break
        # Residual variance of noise eigenvalues
        new_sigma2 = float(np.sum(evals[~signal_mask]) / (N - n_sig))
        new_sigma2 = float(np.clip(new_sigma2, 0.05, 1.0))
        if abs(new_sigma2 - sigma2) < 1e-6:
            sigma2 = new_sigma2
            break
        sigma2 = new_sigma2

    lam_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2
    lam_minus = sigma2 * max(0.0, 1.0 - np.sqrt(q)) ** 2
    signal_mask = evals > lam_plus
    n_signal = int(signal_mask.sum())
    n_noise = N - n_signal

    denoised_evals = evals.copy()
    if n_noise > 0:
        avg_noise = float(np.mean(evals[~signal_mask]))
        denoised_evals[~signal_mask] = max(avg_noise, 1e-6)

    corr_den = (evecs * denoised_evals) @ evecs.T
    corr_den = 0.5 * (corr_den + corr_den.T)
    # Rescale to exact unit diagonal (valid correlation matrix)
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(corr_den), 1e-12))
    corr_den = corr_den * np.outer(d_inv, d_inv)
    np.fill_diagonal(corr_den, 1.0)
    corr_den = _clean_psd(corr_den, eps=1e-8)

    cov_d = corr_den * np.outer(std_d, std_d)
    cov_d = _clean_psd(cov_d)

    return mu_d * days, cov_d * days, {
        "method": "Marchenko-Pastur",
        "q": q,
        "sigma2": float(sigma2),
        "lambda_plus": float(lam_plus),
        "lambda_minus": float(lam_minus),
        "n_signal": n_signal,
        "n_noise": n_noise,
        "raw_evals": evals.tolist(),
        "denoised_evals": denoised_evals.tolist(),
        "cond_num": _cond_number(cov_d),
    }


def estimate_elliptical_mle(returns_df, days=252, nu=5.0, max_iter=200, tol=1e-7):
    """Multivariate Elliptical (Student-t) MLE via Iteratively Reweighted Least Squares.

    For density generator g(d^2) with Mahalanobis distance d_t^2 = (r_t - mu)^T Sigma^-1 (r_t - mu):
        v(x) = -2 g'(x) / g(x) = (nu + N) / (nu + x)
        mu_{k+1}    = sum_t v(d_t^2) r_t / sum_t v(d_t^2)
        Sigma_{k+1} = (1/T) sum_t v(d_t^2) (r_t - mu_{k+1})(r_t - mu_{k+1})^T

    Note: as nu -> infinity, v(x) -> 1 (Gaussian MLE).
    For finite nu > 2, true covariance = (nu / (nu - 2)) * Sigma_scatter.
    """
    X = returns_df.values.astype(float)
    T, N = X.shape

    # Auto-estimate nu from average excess kurtosis if requested
    if nu is None or (isinstance(nu, str) and nu.lower() == "auto"):
        ex_kurt = stats.kurtosis(X, axis=0, fisher=True, bias=False)
        avg_k = float(np.nanmedian(ex_kurt))
        if np.isfinite(avg_k) and avg_k > 0.25:
            nu_val = float(np.clip(4.0 + 6.0 / avg_k, 3.2, 30.0))
        else:
            nu_val = 10.0
    else:
        nu_val = float(nu)
        nu_val = max(nu_val, 2.5)

    # Initialize from sample MLE
    mu = np.mean(X, axis=0)
    sigma = np.cov(X, rowvar=False, ddof=1)
    if sigma.ndim == 0:
        sigma = np.array([[float(sigma)]])
    reg = 1e-8 * float(np.trace(sigma) / max(N, 1) + 1e-12)
    sigma = _clean_psd(sigma, eps=reg)

    converged = False
    iters = 0
    v = np.ones(T, dtype=float)

    for it in range(1, max_iter + 1):
        iters = it
        diff = X - mu
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            sigma_inv = np.linalg.pinv(sigma)

        # Squared Mahalanobis distances d_t^2
        d2 = np.einsum("ti,ij,tj->t", diff, sigma_inv, diff)
        d2 = np.maximum(d2, 0.0)

        # Weight function v(d_t^2) = -2 g'(d_t^2)/g(d_t^2) = (nu + N) / (nu + d_t^2)
        v = (nu_val + N) / (nu_val + d2)

        v_sum = float(np.sum(v))
        if v_sum <= 1e-15:
            break
        mu_new = (v[:, None] * X).sum(axis=0) / v_sum

        diff_new = X - mu_new
        w_diff = diff_new * np.sqrt(v)[:, None]
        sigma_new = (w_diff.T @ w_diff) / float(T)
        sigma_new = _clean_psd(sigma_new, eps=reg)

        mu_delta = float(np.linalg.norm(mu_new - mu) / (np.linalg.norm(mu) + 1e-12))
        sig_delta = float(np.linalg.norm(sigma_new - sigma, ord="fro") / (np.linalg.norm(sigma, ord="fro") + 1e-12))
        mu, sigma = mu_new, sigma_new

        if max(mu_delta, sig_delta) < tol:
            converged = True
            break

    # Convert Student-t scatter matrix to covariance matrix
    scale = (nu_val / (nu_val - 2.0)) if (2.05 < nu_val < 1e4) else 1.0
    cov_d = _clean_psd(sigma * scale)

    return mu * days, cov_d * days, {
        "method": "Elliptical MLE",
        "nu": round(nu_val, 2),
        "iterations": iters,
        "converged": converged,
        "avg_weight": float(np.mean(v)),
        "min_weight": float(np.min(v)),
        "max_weight": float(np.max(v)),
        "pct_downweighted": float(np.mean(v < 0.80)),
        "cond_num": _cond_number(cov_d),
    }


# ═════════════════════════════════════════════════════════════════
# 2. PORTFOLIO OPTIMIZERS (LONG-ONLY EXACT + HRP)
# ═════════════════════════════════════════════════════════════════

def optimize_gmv(cov_ann):
    """Exact long-only Global Minimum Variance portfolio."""
    cov = _clean_psd(np.asarray(cov_ann, dtype=float))
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def obj(w):
        return float(w @ cov @ w)

    def grad(w):
        return 2.0 * (cov @ w)

    res = optimize.minimize(obj, w0, jac=grad, method="SLSQP",
                            bounds=bounds, constraints=cons,
                            options={"ftol": 1e-12, "maxiter": 500})
    w = np.clip(res.x, 0.0, 1.0) if res.success else w0
    s = w.sum()
    return w / s if s > 0 else w0


def optimize_max_sharpe(mu_ann, cov_ann, rf=0.04):
    """Exact long-only Maximum Sharpe Ratio portfolio with multi-start."""
    mu = np.asarray(mu_ann, dtype=float)
    cov = _clean_psd(np.asarray(cov_ann, dtype=float))
    n = len(mu)
    if n == 1:
        return np.array([1.0])

    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def neg_sharpe(w):
        vol = np.sqrt(max(float(w @ cov @ w), 1e-14))
        ret = float(w @ mu)
        return -(ret - rf) / vol

    # Multi-start: equal weight, GMV, and best single-asset Sharpe
    starts = [np.ones(n) / n, optimize_gmv(cov)]
    vols = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    best_single = int(np.argmax((mu - rf) / vols))
    w_single = np.full(n, 0.02 / max(n - 1, 1))
    w_single[best_single] = 1.0 - 0.02
    starts.append(w_single / w_single.sum())

    best_w = starts[0]
    best_val = neg_sharpe(best_w)
    for w0 in starts:
        res = optimize.minimize(neg_sharpe, w0, method="SLSQP",
                                bounds=bounds, constraints=cons,
                                options={"ftol": 1e-11, "maxiter": 500})
        if res.success:
            wc = np.clip(res.x, 0.0, 1.0)
            if wc.sum() > 0:
                wc /= wc.sum()
                val = neg_sharpe(wc)
                if val < best_val:
                    best_val = val
                    best_w = wc

    return best_w


def optimize_hrp(returns_df, cov_ann=None, days=252, linkage_method="single"):
    """Hierarchical Risk Parity (López de Prado, 2016)."""
    X = returns_df.values.astype(float)
    n = X.shape[1]
    if n == 1:
        return np.array([1.0])

    if cov_ann is None:
        cov = np.cov(X, rowvar=False, ddof=1) * days
    else:
        cov = np.asarray(cov_ann, dtype=float)
    cov = _clean_psd(cov)

    if n == 2:
        ivp = 1.0 / np.maximum(np.diag(cov), 1e-12)
        return ivp / ivp.sum()

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = np.clip(0.5 * (corr + corr.T), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    # 1. Correlation distance matrix
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)
    dist_condensed = squareform(dist, checks=False)

    # 2. Hierarchical clustering & 3. Quasi-diagonalization
    link = linkage(dist_condensed, method=linkage_method)
    sort_ix = list(leaves_list(link))

    # 4. Recursive bisection
    def _cluster_var(cov_mat, items):
        sub = cov_mat[np.ix_(items, items)]
        diag = np.maximum(np.diag(sub), 1e-12)
        ivp = (1.0 / diag)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    weights = np.ones(n, dtype=float)
    clusters = [sort_ix]
    while clusters:
        new_clusters = []
        for c in clusters:
            if len(c) <= 1:
                continue
            split = len(c) // 2
            left, right = c[:split], c[split:]
            v_left = _cluster_var(cov, left)
            v_right = _cluster_var(cov, right)
            alpha = 1.0 - v_left / max(v_left + v_right, 1e-15)
            weights[left] *= alpha
            weights[right] *= (1.0 - alpha)
            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)
        clusters = new_clusters

    weights = np.clip(weights, 0.0, 1.0)
    return weights / weights.sum()


def compute_all_schemes(returns_df, days=252, rf=0.04, nu=5.0, objective="max_sharpe"):
    """Compute all advanced portfolio weight schemes & diagnostics.

    Returns dict of weights and diagnostics for:
      gmv, hrp, ledoit_wolf, marchenko_pastur, elliptical
    """
    mu_mle, cov_mle, diag_mle = estimate_sample_mle(returns_df, days)
    mu_lw,  cov_lw,  diag_lw  = estimate_ledoit_wolf(returns_df, days)
    mu_mp,  cov_mp,  diag_mp  = estimate_marchenko_pastur(returns_df, days)
    mu_ell, cov_ell, diag_ell = estimate_elliptical_mle(returns_df, days, nu=nu)

    w_gmv = optimize_gmv(cov_mle)
    w_hrp = optimize_hrp(returns_df, cov_ann=cov_mle, days=days)

    if objective == "gmv":
        w_lw  = optimize_gmv(cov_lw)
        w_mp  = optimize_gmv(cov_mp)
        w_ell = optimize_gmv(cov_ell)
    else:
        w_lw  = optimize_max_sharpe(mu_lw, cov_lw, rf)
        w_mp  = optimize_max_sharpe(mu_mp, cov_mp, rf)
        w_ell = optimize_max_sharpe(mu_ell, cov_ell, rf)

    return {
        "weights": {
            "gmv": w_gmv,
            "hrp": w_hrp,
            "ledoit_wolf": w_lw,
            "marchenko_pastur": w_mp,
            "elliptical": w_ell,
        },
        "diagnostics": {
            "mle": diag_mle,
            "ledoit_wolf": diag_lw,
            "marchenko_pastur": diag_mp,
            "elliptical": diag_ell,
        },
    }


# ═════════════════════════════════════════════════════════════════
# 3. FULL RISK ATTRIBUTION (MCTR, TCTR/CCTR, WEIGHT VS RISK)
# ═════════════════════════════════════════════════════════════════

def fig_risk_attribution_full(returns_df, weights, days=252):
    """Complete Euler Risk Decomposition:
       - Top-left:  Capital Weight % vs % Risk Contribution (%TCTR)
       - Top-right: MCTR (Marginal) & TCTR/CCTR (Total/Component, sums to sigma_p)
       - Bottom:    Full numerical breakdown table + Diversification Ratio
    """
    tickers = list(returns_df.columns)
    n = len(tickers)
    w = np.asarray(weights, dtype=float)
    if len(w) > n:
        w = w[:n]
    elif len(w) < n:
        w = np.append(w, np.zeros(n - len(w)))
    if w.sum() > 0:
        w = w / w.sum()

    cov = _clean_psd(returns_df.cov().values * days)
    port_var = float(w @ cov @ w)
    port_vol = float(np.sqrt(max(port_var, 1e-14)))

    # Euler decomposition of portfolio volatility
    mctr = (cov @ w) / port_vol                 # d(sigma_p)/d(w_i)
    tctr = w * mctr                             # Component / Total Contribution (sums to sigma_p)
    pct_tctr = (tctr / port_vol) * 100.0        # % of portfolio volatility (sums to 100%)
    asset_vols = np.sqrt(np.maximum(np.diag(cov), 1e-14))
    beta_to_port = mctr / port_vol              # asset beta w.r.t. portfolio
    div_ratio = float(np.dot(w, asset_vols) / port_vol) if port_vol > 0 else 1.0

    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.56, 0.44],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "table", "colspan": 2}, None]],
        subplot_titles=(
            "Capital Weight (%) vs Percentage Contribution to Risk (PCR)",
            "Marginal (MCR) & Total Contribution to Risk (TCR, ann. %)",
            "Euler Risk Decomposition Table (MCR · TCR · PCR)"
        ),
    )

    # Panel 1: Weight % vs PCR
    fig.add_trace(go.Bar(
        x=tickers, y=w * 100.0, name="Weight %",
        marker_color=C_BLUE, opacity=0.85,
        text=[f"{v:.1f}%" for v in w * 100.0], textposition="outside",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=tickers, y=pct_tctr, name="PCR (% Risk)",
        marker_color=[C_ORANGE if v >= 0 else C_CYAN for v in pct_tctr],
        text=[f"{v:.1f}%" for v in pct_tctr], textposition="outside",
    ), row=1, col=1)
    fig.add_hline(y=0, line_color="white", opacity=0.25, row=1, col=1)

    # Panel 2: MCR vs TCR (in annualised volatility %)
    fig.add_trace(go.Bar(
        x=tickers, y=mctr * 100.0, name="MCR (Marginal, ann. %)",
        marker_color=C_PURPLE, opacity=0.85,
        text=[f"{v:.1f}%" for v in mctr * 100.0], textposition="outside",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=tickers, y=tctr * 100.0, name="TCR (Total, sums to σ_p)",
        marker_color=C_GREEN,
        text=[f"{v:.2f}%" for v in tctr * 100.0], textposition="outside",
    ), row=1, col=2)
    fig.add_hline(y=0, line_color="white", opacity=0.25, row=1, col=2)

    # Panel 3: Summary Table
    order = np.argsort(pct_tctr)[::-1]
    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Asset</b>", "<b>Weight %</b>", "<b>Standalone σ</b>",
                    "<b>MCR</b>", "<b>TCR (Vol pts)</b>", "<b>PCR (%)</b>",
                    "<b>β to Portfolio</b>"],
            fill_color=C_BLUE, align="left",
            font=dict(color="white", size=11),
        ),
        cells=dict(
            values=[
                [tickers[i] for i in order] + ["<b>TOTAL / PORT</b>"],
                [f"{w[i]*100:.2f}%" for i in order] + [f"<b>{w.sum()*100:.1f}%</b>"],
                [f"{asset_vols[i]*100:.2f}%" for i in order] + [f"<b>{np.dot(w, asset_vols)*100:.2f}%</b> (wtd)"],
                [f"{mctr[i]*100:.2f}%" for i in order] + ["—"],
                [f"{tctr[i]*100:.2f}%" for i in order] + [f"<b>{tctr.sum()*100:.2f}%</b> (= σ_p)"],
                [f"{pct_tctr[i]:.2f}%" for i in order] + [f"<b>{pct_tctr.sum():.1f}%</b>"],
                [f"{beta_to_port[i]:.2f}" for i in order] + ["<b>1.00</b>"],
            ],
            align="left",
            fill_color=[[C_CARD, C_BORDER] * (len(order) // 2 + 2)],
            font=dict(color=C_TEXT, size=11),
            height=25,
        ),
    ), row=2, col=1)

    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=1, col=2)
    fig.update_layout(
        template=TEMPLATE,
        height=max(740, 450 + (n + 2) * 26),
        barmode="group",
        title=dict(
            text=(f"Risk Attribution (Euler Decomposition)  ·  Portfolio σ = {port_vol*100:.2f}%  ·  "
                  f"Diversification Ratio = {div_ratio:.2f}x  ·  Σ TCR = {tctr.sum()*100:.2f}%"),
            y=0.98, yanchor="top",
        ),
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        legend=dict(orientation="h", y=1.07, yanchor="bottom", x=0),
        margin=dict(t=125, b=30, l=50, r=30),
    )
    return fig


# ═════════════════════════════════════════════════════════════════
# 4. PCA — PORTFOLIO RISK DIRECTIONS & ENB
# ═════════════════════════════════════════════════════════════════

def compute_pca_risk(returns_df, weights, days=252):
    """Compute PCA on correlation (standardized factors + MP bound) and covariance (portfolio risk budget)."""
    tickers = list(returns_df.columns)
    X = returns_df.values.astype(float)
    T, N = X.shape
    w = np.asarray(weights, dtype=float)
    if len(w) > N:
        w = w[:N]
    elif len(w) < N:
        w = np.append(w, np.zeros(N - len(w)))
    if w.sum() > 0:
        w = w / w.sum()

    # 1. Correlation PCA (standardized factor structure + MP threshold)
    corr = np.corrcoef(X, rowvar=False) if N > 1 else np.array([[1.0]])
    corr = np.nan_to_num(corr, nan=0.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    evals_c, evecs_c = np.linalg.eigh(corr)
    idx_c = np.argsort(evals_c)[::-1]
    evals_c = np.maximum(evals_c[idx_c], 1e-12)
    evecs_c = evecs_c[:, idx_c]

    # Fix sign convention: largest absolute loading in each PC is positive
    for k in range(N):
        j = int(np.argmax(np.abs(evecs_c[:, k])))
        if evecs_c[j, k] < 0:
            evecs_c[:, k] *= -1.0

    # Standardized factor loadings = correlation between asset i and PC k
    loadings = evecs_c * np.sqrt(evals_c)[None, :]
    var_exp_corr = evals_c / np.sum(evals_c) * 100.0
    cum_var_corr = np.cumsum(var_exp_corr)

    # Marchenko-Pastur upper bound on correlation eigenvalues
    _, _, mp_diag = estimate_marchenko_pastur(returns_df, days)
    lam_plus = mp_diag["lambda_plus"]

    # 2. Covariance PCA (exact portfolio variance decomposition along orthogonal PCs)
    cov_ann = _clean_psd(returns_df.cov().values * days)
    evals_v, evecs_v = np.linalg.eigh(cov_ann)
    idx_v = np.argsort(evals_v)[::-1]
    evals_v = np.maximum(evals_v[idx_v], 1e-14)
    evecs_v = evecs_v[:, idx_v]

    port_var = float(w @ cov_ann @ w)
    w_tilde = evecs_v.T @ w                     # portfolio exposure to each orthogonal eigen-portfolio
    pc_var_contrib = (w_tilde ** 2) * evals_v   # sums exactly to port_var
    p_k = pc_var_contrib / max(port_var, 1e-14) # fractional risk contribution per PC

    # Effective Number of Bets (Meucci entropy ENB)
    p_pos = p_k[p_k > 1e-12]
    enb_entropy = float(np.exp(-np.sum(p_pos * np.log(p_pos))))
    enb_hhi = float(1.0 / np.sum(p_k ** 2)) if np.sum(p_k ** 2) > 0 else 1.0

    return {
        "tickers": tickers,
        "pc_labels": [f"PC{k+1}" for k in range(N)],
        "evals_corr": evals_c,
        "var_exp_corr": var_exp_corr,
        "cum_var_corr": cum_var_corr,
        "loadings": loadings,
        "lambda_plus": lam_plus,
        "n_signal": mp_diag["n_signal"],
        "pc_port_risk_pct": p_k * 100.0,
        "enb_entropy": enb_entropy,
        "enb_hhi": enb_hhi,
        "port_vol": float(np.sqrt(max(port_var, 0.0))),
    }


def fig_pca_risk_directions(returns_df, weights, days=252):
    """2-panel PCA Risk Directions chart:
       - Left:  Scree plot (Correlation Eigenvalues + Cumulative % + Marchenko-Pastur λ+ threshold)
       - Right: Active Portfolio Variance Attribution by Principal Component + ENB
    """
    pca = compute_pca_risk(returns_df, weights, days)
    pcs = pca["pc_labels"]
    n = len(pcs)
    lam_plus = pca["lambda_plus"]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": False}]],
        horizontal_spacing=0.10,
        subplot_titles=(
            f"Eigenvalue Spectrum & Marchenko-Pastur Bound (λ+ = {lam_plus:.2f})",
            f"Active Portfolio Variance by PC  ·  ENB = {pca['enb_entropy']:.2f} / {n} bets"
        ),
    )

    # Left: Eigenvalues bar (signal vs noise color) + Cumulative variance line
    bar_colors = [C_BLUE if ev > lam_plus else C_MUTED for ev in pca["evals_corr"]]
    fig.add_trace(go.Bar(
        x=pcs, y=pca["evals_corr"], name="Eigenvalue (λ_k)",
        marker_color=bar_colors,
        text=[f"{v:.2f}<br>({p:.0f}%)" for v, p in zip(pca["evals_corr"], pca["var_exp_corr"])],
        textposition="outside",
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=pcs, y=pca["cum_var_corr"], name="Cumulative Var %",
        mode="lines+markers", line=dict(color=C_ORANGE, width=2.5),
    ), row=1, col=1, secondary_y=True)

    fig.add_hline(
        y=lam_plus, line_dash="dash", line_color=C_RED,
        annotation_text=f"MP noise edge λ+={lam_plus:.2f} ({pca['n_signal']} signal PCs)",
        annotation_position="top right", annotation_font=dict(color=C_RED, size=10),
        row=1, col=1, secondary_y=False,
    )

    # Right: Portfolio Risk Attribution across PCs
    p_pct = pca["pc_port_risk_pct"]
    cum_p = np.cumsum(p_pct)
    fig.add_trace(go.Bar(
        x=pcs, y=p_pct, name="Portfolio Variance %",
        marker_color=[C_CYAN if k == 0 else C_PURPLE if k == 1 else C_GREEN for k in range(n)],
        text=[f"{v:.1f}%" for v in p_pct], textposition="outside",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=pcs, y=cum_p, name="Cum. Portfolio Risk %",
        mode="lines+markers", line=dict(color=C_YELLOW, width=2, dash="dot"),
    ), row=1, col=2)

    fig.update_yaxes(title_text="Eigenvalue (λ)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Variance (%)", range=[0, 108],
                     ticksuffix="%", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="% of Portfolio Variance", range=[0, max(108, p_pct.max() * 1.18)],
                     ticksuffix="%", row=1, col=2)

    fig.update_layout(
        template=TEMPLATE, height=480,
        title=(f"PCA Risk Directions  ·  PC1 explains {p_pct[0]:.1f}% of Portfolio Risk  ·  "
               f"Effective Number of Bets (ENB): {pca['enb_entropy']:.2f} (HHI: {pca['enb_hhi']:.2f})"),
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        legend=dict(orientation="h", y=-0.16, x=0),
        margin=dict(t=80, b=80),
    )
    return fig


def fig_pca_loadings(returns_df, weights, days=252):
    """Heatmap of standardized PCA factor loadings (Assets × Principal Components)."""
    pca = compute_pca_risk(returns_df, weights, days)
    tickers = pca["tickers"]
    # Show up to top 8 PCs for readability
    k_show = min(len(tickers), 8)
    pcs = [f"PC{k+1} ({pca['var_exp_corr'][k]:.1f}%)" for k in range(k_show)]
    Z = pca["loadings"][:, :k_show]

    fig = go.Figure(go.Heatmap(
        z=Z, x=pcs, y=tickers,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=np.round(Z, 2), texttemplate="%{text}",
        hovertemplate="Asset: %{y}<br>Factor: %{x}<br>Loading (corr): %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        template=TEMPLATE,
        height=max(380, len(tickers) * 34 + 120),
        title="PCA Factor Loadings Heatmap  ·  Asset Correlation with Each Risk Direction ([-1, +1])",
        xaxis_title="Principal Component (% of Asset Universe Variance)",
        yaxis_title="Asset",
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        margin=dict(t=60, b=50),
    )
    return fig


# ═════════════════════════════════════════════════════════════════
# 5. IN-SAMPLE / OUT-OF-SAMPLE ESTIMATOR VALIDATION ENGINE
# ═════════════════════════════════════════════════════════════════

def run_oos_validation(returns_df, bench_ret, days, rf, rebal_months,
                       run_sim_fn, compute_metrics_fn,
                       split_ratio=0.70, objective="max_sharpe", nu="5"):
    """Split backtest strictly into In-Sample (estimation) and Out-of-Sample (evaluation).

    Compares:
      1. Classical MLE (Gaussian baseline)
      2. Ledoit-Wolf Shrinkage
      3. Marchenko-Pastur Denoised
      4. Elliptical MLE (Student-t)
      5. HRP (Hierarchical Risk Parity)
    """
    n_total = len(returns_df)
    split_ratio = float(np.clip(split_ratio, 0.30, 0.85))
    split_idx = int(n_total * split_ratio)
    split_idx = max(120, min(split_idx, n_total - 60))

    ret_is = returns_df.iloc[:split_idx]
    ret_oos = returns_df.iloc[split_idx:]
    bench_oos = bench_ret.reindex(ret_oos.index).dropna()
    common_oos = ret_oos.index.intersection(bench_oos.index)
    ret_oos = ret_oos.loc[common_oos]
    bench_oos = bench_oos.loc[common_oos]

    # Fit estimators strictly on In-Sample window
    mu_mle, cov_mle, diag_mle = estimate_sample_mle(ret_is, days)
    mu_lw,  cov_lw,  diag_lw  = estimate_ledoit_wolf(ret_is, days)
    mu_mp,  cov_mp,  diag_mp  = estimate_marchenko_pastur(ret_is, days)
    mu_ell, cov_ell, diag_ell = estimate_elliptical_mle(ret_is, days, nu=nu)

    opt_fn = (lambda mu, cov: optimize_gmv(cov)) if objective == "gmv" else (lambda mu, cov: optimize_max_sharpe(mu, cov, rf))

    configs = [
        ("Classical MLE",    opt_fn(mu_mle, cov_mle), cov_mle, C_BLUE,   None),
        ("Ledoit-Wolf",      opt_fn(mu_lw,  cov_lw),  cov_lw,  C_YELLOW, "dash"),
        ("Marchenko-Pastur", opt_fn(mu_mp,  cov_mp),  cov_mp,  C_PINK,   "dot"),
        ("Elliptical MLE",   opt_fn(mu_ell, cov_ell), cov_ell, C_TEAL,   "longdash"),
        ("HRP",              optimize_hrp(ret_is, cov_ann=cov_mle, days=days), cov_mle, C_PURPLE, None),
    ]

    oos_series = {}
    is_metrics = {}
    oos_metrics = {}
    weights_map = {}
    pred_vols = {}

    bench_is = bench_ret.reindex(ret_is.index).dropna()

    for label, w, cov_est, _, _ in configs:
        weights_map[label] = w
        pred_vols[label] = float(np.sqrt(max(float(w @ cov_est @ w), 1e-14)))
        _, pr_is = run_sim_fn(ret_is, w, rebal_months)
        _, pr_oos = run_sim_fn(ret_oos, w, rebal_months)
        oos_series[label] = pr_oos
        is_metrics[label] = compute_metrics_fn(pr_is, bench_is, days, label, rf)
        oos_metrics[label] = compute_metrics_fn(pr_oos, bench_oos, days, label, rf)

    m_bench_oos = compute_metrics_fn(bench_oos, bench_oos, days, "Benchmark", rf)

    is_start = ret_is.index[0].strftime("%Y-%m-%d")
    is_end   = ret_is.index[-1].strftime("%Y-%m-%d")
    oos_start = ret_oos.index[0].strftime("%Y-%m-%d")
    oos_end   = ret_oos.index[-1].strftime("%Y-%m-%d")
    obj_lbl = "Min Variance (GMV)" if objective == "gmv" else "Max Sharpe (Tangency)"

    # ── Figure 1: OOS Cumulative Growth ──
    fig_eq = go.Figure()
    for label, _, _, color, dash in configs:
        cum = (1.0 + oos_series[label]).cumprod()
        fig_eq.add_trace(go.Scatter(
            x=cum.index, y=cum, name=f"{label} (OOS Sharpe {oos_metrics[label]['sharpe']:.2f})",
            line=dict(color=color, width=2.4, dash=dash),
        ))
    bcum = (1.0 + bench_oos).cumprod()
    fig_eq.add_trace(go.Scatter(
        x=bcum.index, y=bcum, name=f"Benchmark (OOS Sharpe {m_bench_oos['sharpe']:.2f})",
        line=dict(color=C_MUTED, width=1.6, dash="dot"),
    ))
    fig_eq.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig_eq.update_layout(
        template=TEMPLATE, height=480,
        title=(f"Out-of-Sample Cumulative Performance ({obj_lbl})  ·  "
               f"IS: {is_start}→{is_end} ({len(ret_is)}d)  |  OOS: {oos_start}→{oos_end} ({len(ret_oos)}d)"),
        yaxis_title="Growth of 1.0 (OOS)",
        hovermode="x unified",
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        legend=dict(orientation="h", y=-0.16, x=0),
        margin=dict(t=70, b=85),
    )

    # ── Figure 2: IS vs OOS Comparison Table ──
    rows_lbl, oos_cagr, oos_vol, oos_sh, oos_dd, oos_cvar = [], [], [], [], [], []
    is_sh_col, sh_deg, vol_ratio, eff_n_col = [], [], [], []

    for label, w, _, _, _ in configs:
        mo = oos_metrics[label]
        mi = is_metrics[label]
        vr = mo["ann_vol"] / pred_vols[label] if pred_vols[label] > 0 else 1.0
        rows_lbl.append(label)
        oos_cagr.append(f"{mo['ann_ret']:.2%}")
        oos_vol.append(f"{mo['ann_vol']:.2%}")
        oos_sh.append(f"{mo['sharpe']:.2f}")
        oos_dd.append(f"{mo['max_dd']:.2%}")
        oos_cvar.append(f"{mo['cvar_95']:.2%}")
        is_sh_col.append(f"{mi['sharpe']:.2f}")
        sh_deg.append(f"{mo['sharpe'] - mi['sharpe']:+.2f}")
        vol_ratio.append(f"{vr:.2f}x")
        eff_n_col.append(f"{_effective_n(w):.1f}")

    # Append benchmark row
    rows_lbl.append("Benchmark")
    oos_cagr.append(f"{m_bench_oos['ann_ret']:.2%}")
    oos_vol.append(f"{m_bench_oos['ann_vol']:.2%}")
    oos_sh.append(f"{m_bench_oos['sharpe']:.2f}")
    oos_dd.append(f"{m_bench_oos['max_dd']:.2%}")
    oos_cvar.append(f"{m_bench_oos['cvar_95']:.2%}")
    is_sh_col.append("—")
    sh_deg.append("—")
    vol_ratio.append("—")
    eff_n_col.append("—")

    # Highlight best OOS Sharpe among strategies
    sh_vals = [oos_metrics[c[0]]["sharpe"] for c in configs]
    best_idx = int(np.argmax(sh_vals))
    row_bg = [
        "rgba(48,209,88,0.18)" if i == best_idx else (C_CARD if i % 2 == 0 else C_BORDER)
        for i in range(len(rows_lbl))
    ]

    fig_tbl = go.Figure(go.Table(
        header=dict(
            values=["<b>Estimator</b>", "<b>OOS CAGR</b>", "<b>OOS Vol</b>",
                    "<b>OOS Sharpe</b>", "<b>OOS Max DD</b>", "<b>OOS CVaR 95%</b>",
                    "<b>IS Sharpe</b>", "<b>Sharpe Δ</b>",
                    "<b>σ_OOS / σ̂_IS</b>", "<b>Eff. N</b>"],
            fill_color=C_BLUE, align="left", font=dict(color="white", size=11),
        ),
        cells=dict(
            values=[rows_lbl, oos_cagr, oos_vol, oos_sh, oos_dd, oos_cvar,
                    is_sh_col, sh_deg, vol_ratio, eff_n_col],
            align="left",
            fill_color=[row_bg] * 10,
            font=dict(color=C_TEXT, size=11),
            height=30,
        ),
    ))
    fig_tbl.update_layout(
        template=TEMPLATE, height=270,
        title=f"In-Sample vs Out-of-Sample Estimator Scorecard  ·  Green = Best OOS Sharpe  ·  σ_OOS/σ̂_IS ≈ 1.00 = accurate risk forecast",
        margin=dict(l=0, r=0, t=42, b=0), paper_bgcolor=C_CARD,
    )

    # ── Figure 3: IS-Derived Weights Comparison ──
    tickers = list(returns_df.columns)
    fig_w = go.Figure()
    for label, w, _, color, _ in configs:
        fig_w.add_trace(go.Bar(
            name=label, x=tickers, y=w * 100.0, marker_color=color,
        ))
    fig_w.update_layout(
        template=TEMPLATE, height=380, barmode="group",
        title=f"In-Sample Estimated Weights Comparison ({obj_lbl})",
        yaxis=dict(title="Weight (%)", ticksuffix="%"),
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=55, b=70),
    )

    diagnostics = {
        "is_range": f"{is_start} → {is_end}",
        "oos_range": f"{oos_start} → {oos_end}",
        "is_days": len(ret_is),
        "oos_days": len(ret_oos),
        "objective": obj_lbl,
        "best_oos": configs[best_idx][0],
        "mle_cond": round(diag_mle["cond_num"], 1),
        "lw_shrinkage": round(diag_lw["shrinkage"], 3),
        "lw_cond": round(diag_lw["cond_num"], 1),
        "mp_signal": diag_mp["n_signal"],
        "mp_noise": diag_mp["n_noise"],
        "mp_lambda_plus": round(diag_mp["lambda_plus"], 2),
        "mp_cond": round(diag_mp["cond_num"], 1),
        "ell_nu": diag_ell["nu"],
        "ell_iters": diag_ell["iterations"],
        "ell_avg_w": round(diag_ell["avg_weight"], 3),
        "ell_min_w": round(diag_ell["min_weight"], 3),
        "ell_pct_down": round(diag_ell["pct_downweighted"] * 100.0, 1),
        "ell_cond": round(diag_ell["cond_num"], 1),
    }

    return {
        "fig_equity": fig_eq,
        "fig_table": fig_tbl,
        "fig_weights": fig_w,
        "diagnostics": diagnostics,
    }


# ═════════════════════════════════════════════════════════════════
# 6. MULTI-ENGINE MONTE CARLO SIMULATION
# ═════════════════════════════════════════════════════════════════

def fig_monte_carlo_v2(p_returns, returns_df, backtest_weights, days,
                       n_paths=1000, horizon_years=4, rf=0.04, engine="gaussian"):
    """Forward-looking Monte Carlo with 3 selectable engines:
       - 'gaussian'  : Parametric correlated Gaussian via Cholesky (L L^T = R)
       - 'bootstrap' : Circular Block Bootstrap (21-day blocks, preserves fat tails & vol clustering)
       - 'student_t' : Correlated Multivariate Student-t (nu=5, fitted via Elliptical MLE)
    """
    horizon_days = int(horizon_years * days)
    weights = np.asarray(backtest_weights, dtype=float)
    n_assets = returns_df.shape[1]
    if len(weights) > n_assets:
        weights = weights[:n_assets]
    elif len(weights) < n_assets:
        weights = np.append(weights, np.zeros(n_assets - len(weights)))
    if weights.sum() > 0:
        weights = weights / weights.sum()

    rng = np.random.default_rng(7)
    X = returns_df.values.astype(float)
    T = len(X)

    if engine == "bootstrap":
        # Circular Block Bootstrap on multivariate daily returns
        block = min(21, max(5, T // 10))
        n_blocks = int(np.ceil(horizon_days / block))
        starts = rng.integers(0, T, size=(n_paths, n_blocks))
        offsets = np.arange(block)[None, None, :]
        idx = (starts[:, :, None] + offsets) % T
        idx = idx.reshape(n_paths, n_blocks * block)[:, :horizon_days]
        # Compute portfolio daily return series directly from historical weighted returns
        hist_port = X @ weights
        port_rets = hist_port[idx]
        engine_label = f"Circular Block Bootstrap ({block}d blocks)"

    elif engine == "student_t":
        nu_sim = 5.0
        mu_ann, cov_ann, _ = estimate_elliptical_mle(returns_df, days, nu=nu_sim)
        mu_d = mu_ann / days
        # Scatter matrix = cov * (nu - 2) / nu
        scatter_d = _clean_psd((cov_ann / days) * ((nu_sim - 2.0) / nu_sim))
        try:
            L = np.linalg.cholesky(scatter_d)
        except np.linalg.LinAlgError:
            evals, evecs = np.linalg.eigh(scatter_d)
            L = evecs @ np.diag(np.sqrt(np.maximum(evals, 1e-12)))
        z = rng.standard_normal(size=(n_paths, horizon_days, n_assets))
        chi2 = rng.chisquare(df=nu_sim, size=(n_paths, horizon_days, 1))
        t_draws = (z @ L.T) / np.sqrt(chi2 / nu_sim)
        asset_rets = np.clip(t_draws + mu_d, -0.95, 5.0)
        port_rets = asset_rets @ weights
        engine_label = f"Multivariate Student-t (ν={nu_sim:.0f}, Elliptical MLE)"

    else:
        # Parametric Correlated Gaussian (Cholesky)
        mu_daily = returns_df.mean().values
        std_daily = returns_df.std().values
        corr = returns_df.corr().values
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        corr_pd = _clean_psd(corr, eps=1e-8)
        try:
            L = np.linalg.cholesky(corr_pd)
        except np.linalg.LinAlgError:
            L = np.eye(n_assets)
        z = rng.standard_normal(size=(n_paths, horizon_days, n_assets))
        z_corr = np.einsum("pti,ij->ptj", z, L.T)
        asset_rets = np.clip(z_corr * std_daily + mu_daily, -0.95, 5.0)
        port_rets = asset_rets @ weights
        engine_label = "Parametric Gaussian (Cholesky)"

    cum_growth = np.cumprod(1.0 + port_rets, axis=1)

    p5  = np.percentile(cum_growth, 5,  axis=0)
    p10 = np.percentile(cum_growth, 10, axis=0)
    p25 = np.percentile(cum_growth, 25, axis=0)
    p50 = np.percentile(cum_growth, 50, axis=0)
    p75 = np.percentile(cum_growth, 75, axis=0)
    p90 = np.percentile(cum_growth, 90, axis=0)
    p95 = np.percentile(cum_growth, 95, axis=0)
    mean = cum_growth.mean(axis=0)

    t = np.arange(1, horizon_days + 1)
    final = cum_growth[:, -1]
    var_95_val = float(np.percentile(final, 5))
    cvar_95_val = float(final[final <= var_95_val].mean()) if np.any(final <= var_95_val) else var_95_val
    ann_ret_mc = float(np.median(final)) ** (days / horizon_days) - 1.0
    p_loss = float(np.mean(final < 1.0))

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.66, 0.34],
        subplot_titles=(
            f"Monte Carlo Fan Chart · {engine_label} · {n_paths:,} paths · {horizon_years}Y",
            f"Terminal Wealth Distribution ({horizon_years}Y)"
        ),
    )

    for lo, hi, fill_c, lbl in [
        (p5,  p95, "rgba(10,132,255,0.10)", "5–95%"),
        (p10, p90, "rgba(10,132,255,0.16)", "10–90%"),
        (p25, p75, "rgba(10,132,255,0.24)", "25–75%"),
    ]:
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself", fillcolor=fill_c,
            line=dict(width=0), showlegend=True, name=lbl, hoverinfo="skip",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=p50, name="Median (50th)",
                             line=dict(color=C_BLUE, width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=mean, name="Mean",
                             line=dict(color=C_GREEN, width=1.6, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=p5, name="5th pct (VaR 95%)",
                             line=dict(color=C_RED, width=1.6, dash="dot")), row=1, col=1)

    idx_sample = rng.choice(n_paths, size=min(50, n_paths), replace=False)
    for i in idx_sample:
        fig.add_trace(go.Scatter(
            x=t, y=cum_growth[i],
            line=dict(color="rgba(150,180,255,0.11)", width=1),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.25)",
                  annotation_text="Break-even (1.0x)", row=1, col=1)

    # Cap extreme right-tail outliers only for histogram binning readability
    hi_clip = float(np.percentile(final, 99.5))
    fig.add_trace(go.Histogram(
        x=final[final <= hi_clip], nbinsx=60,
        marker_color=C_BLUE, opacity=0.72, name="Terminal wealth", showlegend=False,
    ), row=1, col=2)

    fig.add_vline(x=var_95_val, line_dash="dash", line_color=C_RED,
                  annotation_text=f"VaR 95%: {var_95_val:.2f}x",
                  annotation_font=dict(color=C_RED, size=10),
                  annotation_position="top left", row=1, col=2)
    fig.add_vline(x=cvar_95_val, line_dash="dot", line_color=C_ORANGE,
                  annotation_text=f"CVaR 95%: {cvar_95_val:.2f}x",
                  annotation_font=dict(color=C_ORANGE, size=10),
                  annotation_position="top right", row=1, col=2)
    fig.add_vline(x=1.0, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=1, col=2)

    fig.update_layout(
        template=TEMPLATE, height=590,
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18, yanchor="top", x=0),
        margin=dict(t=65, b=140),
        annotations=list(fig.layout.annotations) + [
            dict(
                text=(f"Median CAGR: <b>{ann_ret_mc:.1%}</b>  ·  "
                      f"5% VaR: <b>{var_95_val:.2f}x</b>  ·  "
                      f"CVaR 95%: <b>{cvar_95_val:.2f}x</b>  ·  "
                      f"P(loss): <b>{p_loss:.1%}</b>  ·  "
                      f"95th pct: <b>{float(np.percentile(final, 95)):.2f}x</b>"),
                xref="paper", yref="paper", x=0.0, y=-0.31, yanchor="top",
                showarrow=False, font=dict(size=11.5, color=C_CYAN), align="left",
            )
        ],
    )
    fig.update_xaxes(title_text="Trading days forward", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio value (×initial)", row=1, col=1)
    fig.update_xaxes(title_text="Terminal value (×initial)", row=1, col=2)
    fig.update_yaxes(title_text="Paths", row=1, col=2)
    return fig


# ═════════════════════════════════════════════════════════════════
# 7. STATISTICAL ROBUSTNESS & SIGNIFICANCE ANALYSIS
# ═════════════════════════════════════════════════════════════════

def _newey_west_se(x, max_lags=None):
    """Newey-West HAC standard error of the sample mean of 1D array x."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    if T < 5:
        return float(np.std(x, ddof=1) / np.sqrt(max(T, 1)))
    if max_lags is None:
        max_lags = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    max_lags = max(1, min(max_lags, T // 4))
    xc = x - np.mean(x)
    gamma0 = float(np.dot(xc, xc) / T)
    lrv = gamma0
    for lag in range(1, max_lags + 1):
        w = 1.0 - lag / (max_lags + 1.0)
        gamma_l = float(np.dot(xc[lag:], xc[:-lag]) / T)
        lrv += 2.0 * w * gamma_l
    lrv = max(lrv, 1e-18)
    return float(np.sqrt(lrv / T))


def compute_robustness_analysis(p_ret, bench_ret, returns_df, days=252, rf=0.04, n_trials=50, n_boot=2000):
    """Comprehensive statistical validation testing whether backtest returns are distinguishable from noise.

    Returns dict with:
      - summary metrics (t-tests, HAC t-test, CAPM Alpha t-test, PSR, DSR, MinTRL, Bootstrap CIs, Random Null p-value)
      - fig_bootstrap: Bootstrapped Sharpe & CAGR distributions with 95% CIs
      - fig_null:      Observed Sharpe vs Random Long-Only Portfolios Null + Cumulative t-stat stability
    """
    p = p_ret.dropna()
    b = bench_ret.reindex(p.index).dropna()
    common = p.index.intersection(b.index)
    p = p.loc[common]
    b = b.loc[common]

    r = p.values.astype(float)
    rb = b.values.astype(float)
    T = len(r)
    years = T / float(days)
    rf_d = (1.0 + rf) ** (1.0 / days) - 1.0
    ex = r - rf_d
    ex_b = rb - rf_d

    # 1. Classical & Newey-West HAC one-sample t-test (H0: E[r - rf] <= 0)
    mean_ex = float(np.mean(ex))
    se_std = float(np.std(ex, ddof=1) / np.sqrt(max(T, 1)))
    t_std = mean_ex / se_std if se_std > 0 else 0.0
    p_std = float(1.0 - stats.t.cdf(t_std, df=max(T - 1, 1)))

    se_hac = _newey_west_se(ex)
    t_hac = mean_ex / se_hac if se_hac > 0 else 0.0
    p_hac = float(1.0 - stats.norm.cdf(t_hac))

    # 2. CAPM Jensen's Alpha regression with HAC standard error
    var_b = float(np.var(ex_b, ddof=1))
    beta = float(np.cov(ex, ex_b, ddof=1)[0, 1] / var_b) if var_b > 1e-16 else 0.0
    alpha_d = float(np.mean(ex) - beta * np.mean(ex_b))
    resid = ex - (alpha_d + beta * ex_b)
    se_alpha_hac = _newey_west_se(resid)
    t_alpha = alpha_d / se_alpha_hac if se_alpha_hac > 0 else 0.0
    p_alpha = float(1.0 - stats.norm.cdf(t_alpha))
    alpha_ann = alpha_d * days

    # 3. Probabilistic Sharpe Ratio (PSR), Deflated Sharpe Ratio (DSR), MinTRL
    sr_daily = mean_ex / float(np.std(ex, ddof=1)) if np.std(ex, ddof=1) > 0 else 0.0
    sr_ann = sr_daily * np.sqrt(days)
    sr_bench_daily = float(np.mean(ex_b) / np.std(ex_b, ddof=1)) if np.std(ex_b, ddof=1) > 0 else 0.0
    sr_bench_ann = sr_bench_daily * np.sqrt(days)

    skew = float(stats.skew(ex, bias=False))
    kurt = float(stats.kurtosis(ex, fisher=False, bias=False))  # non-excess kurtosis (normal = 3)

    def _psr(sr_hat, sr_star, n_obs, g3, g4):
        denom = np.sqrt(max(1.0 - g3 * sr_hat + ((g4 - 1.0) / 4.0) * (sr_hat ** 2), 1e-12))
        z = (sr_hat - sr_star) * np.sqrt(max(n_obs - 1, 1)) / denom
        return float(stats.norm.cdf(z))

    psr_zero = _psr(sr_daily, 0.0, T, skew, kurt)
    psr_bench = _psr(sr_daily, sr_bench_daily, T, skew, kurt)

    # Deflated Sharpe Ratio (Bailey & López de Prado 2014): expected max SR under null for M effective trials
    m_eff = max(int(n_trials), 2)
    euler_gamma = 0.5772156649
    sr_std_trials = 1.0 / np.sqrt(max(T - 1, 1))
    z1 = stats.norm.ppf(1.0 - 1.0 / m_eff)
    z2 = stats.norm.ppf(1.0 - 1.0 / (m_eff * np.e))
    sr_star_dsr = sr_std_trials * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)
    dsr = _psr(sr_daily, sr_star_dsr, T, skew, kurt)

    # Minimum Track Record Length (MinTRL, 95% confidence for SR > 0)
    z_95 = stats.norm.ppf(0.95)
    if sr_daily > 1e-6:
        mintrl_days = 1.0 + (1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * (sr_daily ** 2)) * ((z_95 / sr_daily) ** 2)
        mintrl_years = float(mintrl_days / days)
    else:
        mintrl_years = float("inf")

    # 4. Circular Block Bootstrap (2,000 resamples, block=21d)
    rng = np.random.default_rng(42)
    block = min(21, max(5, T // 12))
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(n_boot, n_blocks))
    offsets = np.arange(block)[None, None, :]
    idx = (starts[:, :, None] + offsets) % T
    idx = idx.reshape(n_boot, n_blocks * block)[:, :T]
    boot_r = r[idx]
    boot_ex = boot_r - rf_d

    boot_mean = boot_ex.mean(axis=1)
    boot_std = np.maximum(boot_ex.std(axis=1, ddof=1), 1e-12)
    boot_sharpe = (boot_mean / boot_std) * np.sqrt(days)
    # Use log1p for fast numerical stability in CAGR
    boot_cagr = np.expm1(np.mean(np.log1p(np.clip(boot_r, -0.99, None)), axis=1) * days)

    sh_ci = (float(np.percentile(boot_sharpe, 2.5)), float(np.percentile(boot_sharpe, 97.5)))
    cagr_ci = (float(np.percentile(boot_cagr, 2.5)), float(np.percentile(boot_cagr, 97.5)))
    boot_p_zero = float(np.mean(boot_sharpe <= 0.0))
    boot_p_bench = float(np.mean(boot_sharpe <= sr_bench_ann))

    # 5. Random Long-Only Portfolio Null Test (2,000 Dirichlet random allocations on same assets)
    X_all = returns_df.reindex(common).dropna().values.astype(float)
    n_assets = X_all.shape[1]
    n_rand = 2000
    rand_w = rng.dirichlet(np.ones(n_assets), size=n_rand)  # (n_rand, N)
    rand_rets = rand_w @ X_all.T                            # (n_rand, T)
    rand_ex = rand_rets - rf_d
    rand_sharpe = (rand_ex.mean(axis=1) / np.maximum(rand_ex.std(axis=1, ddof=1), 1e-12)) * np.sqrt(days)
    null_p_val = float(np.mean(rand_sharpe >= sr_ann))
    null_pct_rank = float(np.mean(rand_sharpe < sr_ann) * 100.0)

    # 6. Expanding / Cumulative t-stat stability over time
    min_obs = min(max(63, days // 2), max(T // 3, 20))
    step = max(1, (T - min_obs) // 250)
    t_dates, t_series = [], []
    for k in range(min_obs, T + 1, step):
        sub = ex[:k]
        s_sub = float(np.std(sub, ddof=1))
        t_val = float(np.mean(sub) / (s_sub / np.sqrt(k))) if s_sub > 0 else 0.0
        t_dates.append(p.index[k - 1])
        t_series.append(t_val)

    # Verdict classification
    if p_hac < 0.05 and psr_zero >= 0.95 and boot_p_zero < 0.05:
        verdict = "STATISTICALLY SIGNIFICANT (95%+)"
        verdict_level = "strong"
    elif p_hac < 0.10 or psr_zero >= 0.90:
        verdict = "MODERATE EVIDENCE (90%)"
        verdict_level = "moderate"
    else:
        verdict = "NOT DISTINGUISHABLE FROM NOISE"
        verdict_level = "weak"

    # ── Figure A: Block Bootstrap Distributions (Sharpe & CAGR) ──
    fig_boot = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Block Bootstrap Sharpe ({n_boot:,} resamples, 95% CI: [{sh_ci[0]:.2f}, {sh_ci[1]:.2f}])",
            f"Block Bootstrap CAGR ({n_boot:,} resamples, 95% CI: [{cagr_ci[0]:.1%}, {cagr_ci[1]:.1%}])"
        ),
    )
    fig_boot.add_trace(go.Histogram(
        x=boot_sharpe, nbinsx=55, marker_color=C_BLUE, opacity=0.72, name="Boot Sharpe", showlegend=False
    ), row=1, col=1)
    fig_boot.add_vline(x=sr_ann, line_color=C_GREEN, line_width=2.5,
                       annotation_text=f"Observed {sr_ann:.2f}", row=1, col=1)
    fig_boot.add_vline(x=0.0, line_color=C_RED, line_dash="dash",
                       annotation_text="Zero (H0)", annotation_position="top left", row=1, col=1)
    fig_boot.add_vline(x=sr_bench_ann, line_color=C_ORANGE, line_dash="dot",
                       annotation_text=f"Bench {sr_bench_ann:.2f}", annotation_position="bottom right", row=1, col=1)
    fig_boot.add_vrect(x0=sh_ci[0], x1=sh_ci[1], fillcolor="rgba(48,209,88,0.08)", line_width=0, row=1, col=1)

    obs_cagr = float((1.0 + p).prod() ** (days / T) - 1.0)
    fig_boot.add_trace(go.Histogram(
        x=boot_cagr * 100.0, nbinsx=55, marker_color=C_PURPLE, opacity=0.72, name="Boot CAGR", showlegend=False
    ), row=1, col=2)
    fig_boot.add_vline(x=obs_cagr * 100.0, line_color=C_GREEN, line_width=2.5,
                       annotation_text=f"Observed {obs_cagr:.1%}", row=1, col=2)
    fig_boot.add_vline(x=0.0, line_color=C_RED, line_dash="dash", row=1, col=2)
    fig_boot.add_vrect(x0=cagr_ci[0] * 100.0, x1=cagr_ci[1] * 100.0,
                       fillcolor="rgba(191,90,242,0.08)", line_width=0, row=1, col=2)

    fig_boot.update_xaxes(title_text="Annualised Sharpe Ratio", row=1, col=1)
    fig_boot.update_xaxes(title_text="Annualised Return / CAGR (%)", ticksuffix="%", row=1, col=2)
    fig_boot.update_yaxes(title_text="Frequency", row=1, col=1)
    fig_boot.update_layout(
        template=TEMPLATE, height=430,
        title=f"Circular Block Bootstrap ({block}d blocks)  ·  P(Sharpe ≤ 0) = {boot_p_zero:.2%}  ·  P(Sharpe ≤ Bench) = {boot_p_bench:.2%}",
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD, margin=dict(t=65, b=50),
    )

    # ── Figure B: Random Portfolio Null Test & Expanding t-stat Stability ──
    fig_null = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Allocation Skill vs {n_rand:,} Random Portfolios (Rank: {null_pct_rank:.1f}th pct)",
            "Expanding Excess-Return t-Statistic Over Time"
        ),
    )
    fig_null.add_trace(go.Histogram(
        x=rand_sharpe, nbinsx=55, marker_color=C_MUTED, opacity=0.78, name="Random Portfolios", showlegend=False
    ), row=1, col=1)
    fig_null.add_vline(x=sr_ann, line_color=C_CYAN, line_width=2.5,
                       annotation_text=f"Active Port ({sr_ann:.2f})", row=1, col=1)
    fig_null.add_vline(x=float(np.median(rand_sharpe)), line_color=C_ORANGE, line_dash="dot",
                       annotation_text=f"Random Med ({np.median(rand_sharpe):.2f})",
                       annotation_position="bottom left", row=1, col=1)

    fig_null.add_trace(go.Scatter(
        x=t_dates, y=t_series, mode="lines", name="Expanding t-stat",
        line=dict(color=C_CYAN, width=2.2), fill="tozeroy",
        fillcolor="rgba(100,210,255,0.10)", showlegend=False,
    ), row=1, col=2)
    fig_null.add_hline(y=1.96, line_dash="dash", line_color=C_GREEN,
                       annotation_text="t = 1.96 (p=5%)", row=1, col=2)
    fig_null.add_hline(y=0.0, line_color="rgba(255,255,255,0.25)", row=1, col=2)

    fig_null.update_xaxes(title_text="Annualised Sharpe Ratio", row=1, col=1)
    fig_null.update_xaxes(title_text="Date", row=1, col=2)
    fig_null.update_yaxes(title_text="Count", row=1, col=1)
    fig_null.update_yaxes(title_text="Cumulative t-statistic", row=1, col=2)
    fig_null.update_layout(
        template=TEMPLATE, height=430,
        title=f"Randomness & Stability Check  ·  Random-Weights Null p-value = {null_p_val:.3f}  ·  HAC t-stat = {t_hac:.2f}",
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD, margin=dict(t=65, b=50),
    )

    summary = {
        "verdict": verdict,
        "verdict_level": verdict_level,
        "years": round(years, 1),
        "n_days": T,
        "sr_ann": round(sr_ann, 3),
        "sr_bench_ann": round(sr_bench_ann, 3),
        "t_std": round(t_std, 2),
        "p_std": round(p_std, 4),
        "t_hac": round(t_hac, 2),
        "p_hac": round(p_hac, 4),
        "alpha_ann": round(alpha_ann, 4),
        "t_alpha": round(t_alpha, 2),
        "p_alpha": round(p_alpha, 4),
        "skew": round(skew, 2),
        "excess_kurt": round(kurt - 3.0, 2),
        "psr_zero": round(psr_zero, 4),
        "psr_bench": round(psr_bench, 4),
        "dsr": round(dsr, 4),
        "mintrl_years": round(mintrl_years, 2) if np.isfinite(mintrl_years) else None,
        "boot_sh_ci": [round(sh_ci[0], 2), round(sh_ci[1], 2)],
        "boot_cagr_ci": [round(cagr_ci[0] * 100.0, 2), round(cagr_ci[1] * 100.0, 2)],
        "boot_p_zero": round(boot_p_zero, 4),
        "boot_p_bench": round(boot_p_bench, 4),
        "null_p_val": round(null_p_val, 4),
        "null_pct_rank": round(null_pct_rank, 1),
    }

    return {
        "summary": summary,
        "fig_bootstrap": fig_boot,
        "fig_null": fig_null,
    }


# ═════════════════════════════════════════════════════════════════
# 8. TRUE OUT-OF-SAMPLE ROLLING WALK-FORWARD FOR SCHEME COMPARISON
# ═════════════════════════════════════════════════════════════════

def run_rolling_oos_schemes(returns_df, bench_ret, days, rf, rebal_months,
                            compute_metrics_fn, nu=5.0, window_years=3):
    """True zero-look-ahead Out-of-Sample Walk-Forward comparison of all weighting schemes.

    At every rebalance date (in rebal_months) after the initial warm-up window,
    each estimator fits ONLY on trailing historical data [t - win : t], computes
    long-only target weights, and holds (with daily drift) until the next rebalance.
    """
    n_total = len(returns_df)
    win = min(int(window_years * days), max(120, n_total // 3))
    if n_total <= win + 30:
        win = max(60, n_total // 2)

    idx = returns_df.index
    # Identify first trading day of each rebalance month after warm-up
    rebal_set = set(rebal_months or [1, 5, 9])
    rebal_indices = []
    prev_ym = None
    for i in range(win, n_total):
        dt = idx[i]
        ym = (dt.year, dt.month)
        if ym != prev_ym:
            prev_ym = ym
            if dt.month in rebal_set:
                rebal_indices.append(i)

    # Always start at `win` so OOS series begins immediately after warm-up
    if not rebal_indices or rebal_indices[0] != win:
        rebal_indices = [win] + rebal_indices
    rebal_lookup = set(rebal_indices)

    schemes = [
        ("markowitz",        "Sample MLE",       C_BLUE,   None),
        ("gmv",              "Global Min Variance", C_CYAN, "dashdot"),
        ("hrp",              "HRP",              C_PURPLE, None),
        ("ledoit_wolf",      "Ledoit-Wolf",      C_YELLOW, "dash"),
        ("marchenko_pastur", "Marchenko-Pastur", C_PINK,   "dot"),
        ("elliptical",       "Elliptical MLE",   C_TEAL,   "longdash"),
    ]
    keys = [s[0] for s in schemes]
    label_map = {s[0]: s[1] for s in schemes}
    cols = list(returns_df.columns)
    X = returns_df.values.astype(float)
    n_assets = X.shape[1]

    cur_w = {k: np.ones(n_assets) / n_assets for k in keys}
    port_rets = {k: [] for k in keys}
    weights_hist_raw = {k: [] for k in keys}
    rebal_weights_raw = {k: [] for k in keys}
    rebal_dates = []
    oos_dates = idx[win:]

    for i in range(win, n_total):
        if i in rebal_lookup:
            hist = returns_df.iloc[max(0, i - win):i]
            mu_mle, cov_mle, _ = estimate_sample_mle(hist, days)
            mu_lw,  cov_lw,  _ = estimate_ledoit_wolf(hist, days)
            mu_mp,  cov_mp,  _ = estimate_marchenko_pastur(hist, days)
            mu_ell, cov_ell, _ = estimate_elliptical_mle(hist, days, nu=nu, max_iter=80)

            cur_w["markowitz"]        = optimize_max_sharpe(mu_mle, cov_mle, rf)
            cur_w["gmv"]              = optimize_gmv(cov_mle)
            cur_w["hrp"]              = optimize_hrp(hist, cov_ann=cov_mle, days=days)
            cur_w["ledoit_wolf"]      = optimize_max_sharpe(mu_lw, cov_lw, rf)
            cur_w["marchenko_pastur"] = optimize_max_sharpe(mu_mp, cov_mp, rf)
            cur_w["elliptical"]       = optimize_max_sharpe(mu_ell, cov_ell, rf)

            rebal_dates.append(idx[i])
            for k in keys:
                rebal_weights_raw[k].append(cur_w[k].copy())

        r_day = X[i]
        for k in keys:
            w = cur_w[k]
            weights_hist_raw[k].append(w.copy())
            pr = float(np.dot(r_day, w))
            port_rets[k].append(pr)
            denom = 1.0 + pr
            if denom > 1e-12:
                cur_w[k] = w * (1.0 + r_day) / denom

    oos_series_raw = {k: pd.Series(port_rets[k], index=oos_dates) for k in keys}
    bench_oos = bench_ret.reindex(oos_dates).dropna()
    common = oos_dates.intersection(bench_oos.index)
    bench_oos = bench_oos.loc[common]

    series_map = {k: oos_series_raw[k].loc[common] for k in keys}
    weights_hist_map = {
        k: pd.DataFrame(weights_hist_raw[k], index=oos_dates, columns=cols).loc[common]
        for k in keys
    }
    rebal_df_map = {
        k: pd.DataFrame(rebal_weights_raw[k], index=rebal_dates, columns=cols)
        for k in keys
    }

    # Per-rebalance realized OOS Sharpe for each scheme (for Walk-Forward tab)
    rebal_sharpes_map = {}
    for k in keys:
        s_k = series_map[k]
        sh_list = []
        for j in range(len(rebal_dates)):
            d0 = rebal_dates[j]
            d1 = rebal_dates[j + 1] if j + 1 < len(rebal_dates) else common[-1]
            seg = s_k[(s_k.index >= d0) & (s_k.index < d1)] if j + 1 < len(rebal_dates) else s_k[s_k.index >= d0]
            if len(seg) >= 10 and seg.std() > 1e-10:
                ann_r = float((1.0 + seg).prod() ** (days / len(seg)) - 1.0)
                ann_v = float(seg.std() * np.sqrt(days))
                sh_list.append(float((ann_r - rf) / ann_v))
            else:
                sh_list.append(0.0)
        rebal_sharpes_map[k] = sh_list

    metrics_map = {}
    metrics_list = []
    for k, lbl, _, _ in schemes:
        m_k = compute_metrics_fn(series_map[k], bench_oos, days, f"{lbl} [OOS]", rf)
        metrics_map[k] = m_k
        metrics_list.append(m_k)
    m_bench = compute_metrics_fn(bench_oos, bench_oos, days, "Benchmark", rf)
    metrics_map["benchmark"] = m_bench
    metrics_list.append(m_bench)

    # Compute actionable weights on the LATEST trailing rolling window [n_total - win : n_total]
    latest_hist = returns_df.iloc[max(0, n_total - win):]
    mu_mle_l, cov_mle_l, _ = estimate_sample_mle(latest_hist, days)
    mu_lw_l,  cov_lw_l,  _ = estimate_ledoit_wolf(latest_hist, days)
    mu_mp_l,  cov_mp_l,  _ = estimate_marchenko_pastur(latest_hist, days)
    mu_ell_l, cov_ell_l, _ = estimate_elliptical_mle(latest_hist, days, nu=nu, max_iter=100)

    latest_weights = {
        "markowitz":        optimize_max_sharpe(mu_mle_l, cov_mle_l, rf),
        "gmv":              optimize_gmv(cov_mle_l),
        "hrp":              optimize_hrp(latest_hist, cov_ann=cov_mle_l, days=days),
        "ledoit_wolf":      optimize_max_sharpe(mu_lw_l, cov_lw_l, rf),
        "marchenko_pastur": optimize_max_sharpe(mu_mp_l, cov_mp_l, rf),
        "elliptical":       optimize_max_sharpe(mu_ell_l, cov_ell_l, rf),
    }

    return {
        "series": series_map,
        "weights_hist": weights_hist_map,
        "rebal_df": rebal_df_map,
        "rebal_sharpes": rebal_sharpes_map,
        "bench_oos": bench_oos,
        "metrics": metrics_map,
        "metrics_list": metrics_list,
        "latest_weights": latest_weights,
        "latest_window_range": f"{latest_hist.index[0].strftime('%Y-%m-%d')} → {latest_hist.index[-1].strftime('%Y-%m-%d')}",
        "oos_start": common[0].strftime("%Y-%m-%d"),
        "oos_end": common[-1].strftime("%Y-%m-%d"),
        "n_rebalances": len(rebal_dates),
    }


# ═════════════════════════════════════════════════════════════════
# 9. PORTFOLIO A vs PORTFOLIO B COMPARISON CHARTS
# ═════════════════════════════════════════════════════════════════

def build_compare_ab_charts(pr_a, pr_b, bench_ret, label_a, label_b,
                            m_a, m_b, m_bench, tickers_a, w_a, tickers_b, w_b,
                            rf=0.04, cs="$"):
    """Build all comparison figures for Portfolio A (active) vs Portfolio B (custom)."""
    # 1. Cumulative Growth (log scale, rebased to 1.0 on common dates)
    cum_a = (1.0 + pr_a).cumprod()
    cum_b = (1.0 + pr_b).cumprod()
    cum_bm = (1.0 + bench_ret).cumprod()

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=cum_a.index, y=cum_a, name=f"Portfolio A ({label_a})",
                                line=dict(color=C_BLUE, width=2.8)))
    fig_eq.add_trace(go.Scatter(x=cum_b.index, y=cum_b, name=f"Portfolio B ({label_b})",
                                line=dict(color=C_GREEN, width=2.8)))
    fig_eq.add_trace(go.Scatter(x=cum_bm.index, y=cum_bm, name="Benchmark",
                                line=dict(color=C_MUTED, width=1.6, dash="dot")))
    corr_ab = float(pr_a.corr(pr_b))
    fig_eq.update_layout(
        template=TEMPLATE, height=460,
        title=(f"Portfolio A vs Portfolio B — Cumulative Growth ({cs}, log scale)  ·  "
               f"{pr_a.index[0].strftime('%Y-%m-%d')} → {pr_a.index[-1].strftime('%Y-%m-%d')}  ·  "
               f"Daily Corr(A,B) = {corr_ab:.2f}"),
        yaxis_type="log", yaxis_title="Growth of 1.0",
        hovermode="x unified",
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        legend=dict(orientation="h", y=-0.14, x=0),
        margin=dict(t=60, b=70),
    )

    # 2. Side-by-Side Metrics Table with Delta (A - B)
    rows = [
        ("Ann. Return (CAGR)", "ann_ret",      "{:.2%}", "{:+.2%}", True),
        ("Ann. Volatility",    "ann_vol",      "{:.2%}", "{:+.2%}", False),
        (f"Sharpe (rf={rf:.1%})", "sharpe",    "{:.2f}", "{:+.2f}", True),
        (f"Sortino (rf={rf:.1%})", "sortino",  "{:.2f}", "{:+.2f}", True),
        ("Max Drawdown",       "max_dd",       "{:.2%}", "{:+.2%}", True),
        ("Calmar Ratio",       "calmar",       "{:.2f}", "{:+.2f}", True),
        ("Beta (vs bench)",    "beta",         "{:.2f}", "{:+.2f}", False),
        ("Jensen's Alpha",     "alpha",        "{:.2%}", "{:+.2%}", True),
        ("Annual Win Rate",    "win_rate",     "{:.0%}", "{:+.0%}", True),
        ("VaR 95% (daily)",    "var_95",       "{:.2%}", "{:+.2%}", True),
        ("CVaR 95% (daily)",   "cvar_95",      "{:.2%}", "{:+.2%}", True),
        ("Avg Recovery (d)",   "avg_recovery", "{:.0f}", "{:+.0f}", False),
        ("Max Recovery (d)",   "max_recovery", "{:.0f}", "{:+.0f}", False),
    ]
    m_names, col_a, col_b, col_diff, col_bm, diff_colors = [], [], [], [], [], []
    for title, key, fmt, dfmt, higher_better in rows:
        va, vb, vbm = m_a[key], m_b[key], m_bench[key]
        diff = va - vb
        m_names.append(title)
        col_a.append(fmt.format(va))
        col_b.append(fmt.format(vb))
        col_diff.append(dfmt.format(diff))
        col_bm.append(fmt.format(vbm))
        a_wins = (diff > 0) if higher_better else (diff < 0)
        diff_colors.append("rgba(48,209,88,0.22)" if a_wins else "rgba(255,69,58,0.20)")

    fig_tbl = go.Figure(go.Table(
        header=dict(
            values=["<b>Metric</b>", f"<b>Portfolio A ({label_a})</b>",
                    f"<b>Portfolio B ({label_b})</b>", "<b>Δ (A − B)</b>", "<b>Benchmark</b>"],
            fill_color=C_BLUE, align="left", font=dict(color="white", size=12),
        ),
        cells=dict(
            values=[m_names, col_a, col_b, col_diff, col_bm],
            align="left",
            fill_color=[[C_CARD, C_BORDER] * 10, [C_CARD, C_BORDER] * 10,
                        [C_CARD, C_BORDER] * 10, diff_colors, [C_CARD, C_BORDER] * 10],
            font=dict(color=C_TEXT, size=11), height=28,
        ),
    ))
    fig_tbl.update_layout(template=TEMPLATE, height=420,
                          margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor=C_CARD)

    # 3. Drawdown & Annual Returns side-by-side
    def _dd(s):
        c = (1.0 + s).cumprod()
        return (c - c.cummax()) / c.cummax() * 100.0

    dd_a, dd_b = _dd(pr_a), _dd(pr_b)
    ann_a = pr_a.groupby(pr_a.index.year).apply(lambda x: (1.0 + x).prod() - 1.0) * 100.0
    ann_b = pr_b.groupby(pr_b.index.year).apply(lambda x: (1.0 + x).prod() - 1.0) * 100.0

    fig_dd_ann = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Underwater Drawdown (%) — A vs B", "Annual Returns (%) — A vs B"),
    )
    fig_dd_ann.add_trace(go.Scatter(x=dd_a.index, y=dd_a, name="Port A DD",
                                    fill="tozeroy", fillcolor="rgba(10,132,255,0.18)",
                                    line=dict(color=C_BLUE, width=1.8)), row=1, col=1)
    fig_dd_ann.add_trace(go.Scatter(x=dd_b.index, y=dd_b, name="Port B DD",
                                    fill="tozeroy", fillcolor="rgba(48,209,88,0.16)",
                                    line=dict(color=C_GREEN, width=1.8)), row=1, col=1)
    fig_dd_ann.add_trace(go.Bar(x=ann_a.index, y=ann_a, name="Port A Ann.",
                                marker_color=C_BLUE), row=1, col=2)
    fig_dd_ann.add_trace(go.Bar(x=ann_b.index, y=ann_b, name="Port B Ann.",
                                marker_color=C_GREEN), row=1, col=2)
    fig_dd_ann.update_yaxes(ticksuffix="%", row=1, col=1)
    fig_dd_ann.update_yaxes(ticksuffix="%", row=1, col=2)
    fig_dd_ann.update_layout(
        template=TEMPLATE, height=430, barmode="group",
        hovermode="x unified", paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        legend=dict(orientation="h", y=-0.16, x=0), margin=dict(t=55, b=70),
    )

    # 4. Side-by-side Allocation Donuts
    fig_alloc = make_subplots(
        rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=(f"Portfolio A — {label_a}", f"Portfolio B — {label_b}"),
    )
    fig_alloc.add_trace(go.Pie(
        labels=tickers_a, values=np.asarray(w_a) * 100.0, hole=0.48,
        textinfo="label+percent", textposition="inside", name="Port A",
    ), row=1, col=1)
    fig_alloc.add_trace(go.Pie(
        labels=tickers_b, values=np.asarray(w_b) * 100.0, hole=0.48,
        textinfo="label+percent", textposition="inside", name="Port B",
    ), row=1, col=2)
    fig_alloc.update_layout(
        template=TEMPLATE, height=360, paper_bgcolor=C_CARD, showlegend=False,
        margin=dict(t=50, b=20),
    )

    return {
        "fig_equity": fig_eq,
        "fig_table": fig_tbl,
        "fig_dd_ann": fig_dd_ann,
        "fig_alloc": fig_alloc,
        "corr_ab": round(corr_ab, 3),
    }
