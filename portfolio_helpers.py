"""
FastbreaK – Currency conversion & multi-portfolio helpers.
Imported by portfolio.py.
"""

import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

MAX_TICKERS = 20

DEFAULT_TICKERS = [
    ("AZO",  "USD"), ("MSFT", "USD"), ("COST", "USD"),
    ("JNJ",  "USD"), ("MCD",  "USD"), ("APH",  "USD"),
    ("CHD",  "USD"), ("TLT",  "USD"), ("GC=F", "USD"),
]

CURRENCY_OPTIONS = [
    {"label": "USD ($)", "value": "USD"},
    {"label": "EUR (€)", "value": "EUR"},
]

CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€"}

# ─────────────────────────────────────────────────────────────────
# FX DATA
# ─────────────────────────────────────────────────────────────────

def download_fx_rate(start_date, end_date):
    """Download EURUSD=X and return daily FX *price level* (ffill, no dropna).

    Returns a Series of EURUSD prices (1 EUR = X USD), indexed by trading date.
    Returns None on failure.
    NOTE: We return the price level (not pct_change) so callers can align to
    any index with ffill without introducing NaNs that drop rows.
    """
    try:
        fx = yf.download("EURUSD=X", start=start_date, end=end_date,
                         auto_adjust=True, progress=False)["Close"]
        if isinstance(fx, pd.DataFrame):
            fx = fx.squeeze()
        fx = fx.ffill().bfill()
        if fx.empty:
            return None
        return fx   # price level, NOT pct_change
    except Exception:
        return None


def convert_returns_currency(ret, src_ccy, tgt_ccy, fx_prices):
    """Convert a return Series from src_ccy to tgt_ccy.

    fx_prices : EURUSD price level Series (1 EUR = X USD).
                We compute the daily FX return *aligned to ret.index* using
                ffill so we never introduce new NaN rows.

    Returns a Series with EXACTLY the same index as ret.
    """
    if src_ccy == tgt_ccy or fx_prices is None:
        return ret

    # Align FX prices to the return series index using ffill (no new NaNs)
    fx_aligned = fx_prices.reindex(ret.index, method="ffill").ffill().bfill()

    # Compute FX daily returns from the price level
    fx_ret = fx_aligned.pct_change().fillna(0)

    if src_ccy == "EUR" and tgt_ccy == "USD":
        # EUR asset return → USD: multiply by (1 + FX daily move)
        return (1 + ret) * (1 + fx_ret) - 1
    elif src_ccy == "USD" and tgt_ccy == "EUR":
        # USD asset return → EUR: divide by (1 + FX daily move)
        return (1 + ret) / (1 + fx_ret) - 1
    return ret   # unknown pair — return unchanged


def convert_all_returns(returns_df, bench_ret, ticker_currencies,
                        base_currency, fx_prices):
    """Convert every column's returns to base_currency.

    fx_prices : EURUSD price level Series (from download_fx_rate).
    Returns (converted_returns_df, converted_bench_ret) with the SAME
    number of rows as the inputs — no rows are ever dropped here.
    """
    conv = returns_df.copy()
    for col in conv.columns:
        src = ticker_currencies.get(col, "USD")
        conv[col] = convert_returns_currency(conv[col], src, base_currency, fx_prices)

    # DO NOT dropna here — would silently empty the DataFrame when FX data
    # has gaps. NaN values (if any) are left for the caller to decide.
    if not isinstance(conv.index, pd.DatetimeIndex):
        conv.index = pd.DatetimeIndex(conv.index)

    bench_conv = convert_returns_currency(bench_ret, "USD", base_currency, fx_prices)
    if not isinstance(bench_conv.index, pd.DatetimeIndex):
        bench_conv.index = pd.DatetimeIndex(bench_conv.index)

    return conv, bench_conv


# ─────────────────────────────────────────────────────────────────
# MULTI-PORTFOLIO CUMULATIVE CHART
# ─────────────────────────────────────────────────────────────────

# Palette for multi-portfolio lines
_MP_COLORS  = {"Markowitz Optimal":        "#0a84ff",
               "Sample MLE":               "#0a84ff",
               "Sample MLE [OOS]":         "#0a84ff",
               "Global Min Variance":      "#64d2ff",
               "Global Min Variance [OOS]":"#64d2ff",
               "HRP":                      "#bf5af2",
               "HRP [OOS]":                "#bf5af2",
               "Ledoit-Wolf":              "#ffd60a",
               "Ledoit-Wolf [OOS]":        "#ffd60a",
               "Marchenko-Pastur":         "#ff375f",
               "Marchenko-Pastur [OOS]":   "#ff375f",
               "Elliptical MLE":           "#5ac8fa",
               "Elliptical MLE [OOS]":     "#5ac8fa",
               "Custom Weights":           "#30d158",
               "Benchmark":                "#555570"}
_MP_DASHES  = {"Markowitz Optimal":        None,
               "Sample MLE":               None,
               "Sample MLE [OOS]":         None,
               "Global Min Variance":      "dashdot",
               "Global Min Variance [OOS]":"dashdot",
               "HRP":                      None,
               "HRP [OOS]":                None,
               "Ledoit-Wolf":              "dash",
               "Ledoit-Wolf [OOS]":        "dash",
               "Marchenko-Pastur":         "dot",
               "Marchenko-Pastur [OOS]":   "dot",
               "Elliptical MLE":           "longdash",
               "Elliptical MLE [OOS]":     "longdash",
               "Custom Weights":           "dot",
               "Benchmark":                "dot"}


def fig_cumulative_multi(portfolios, bench_ret, cs="$"):
    """Cumulative-growth chart comparing multiple portfolios.

    portfolios : list of (label, returns_series)
    bench_ret  : benchmark return Series
    cs         : currency symbol for annotation
    """
    C_CARD = "#111118"
    TEMPLATE = "plotly_dark"

    fig = go.Figure()

    for label, p_ret in portfolios:
        cum = (1 + p_ret).cumprod()
        color = _MP_COLORS.get(label, "#17BECF")
        dash  = _MP_DASHES.get(label, None)
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum, name=label,
            line=dict(color=color, width=2.5, dash=dash),
        ))

    # Benchmark
    spy_cum = (1 + bench_ret).cumprod()
    fig.add_trace(go.Scatter(
        x=spy_cum.index, y=spy_cum, name="Benchmark",
        line=dict(color=_MP_COLORS["Benchmark"], width=1.5, dash="dot"),
    ))

    fig.update_layout(
        template=TEMPLATE, height=540,
        title=f"Cumulative Growth – Multi-Portfolio Comparison ({cs})  ·  log scale",
        yaxis_type="log", hovermode="x unified",
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        legend=dict(orientation="h", y=-0.12, x=0),
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# WEIGHT PARSING
# ─────────────────────────────────────────────────────────────────

def parse_custom_weights(raw_weights, n_tickers):
    """Parse user-entered weight values.  Returns np.ndarray or None.

    * If ALL weights are empty/zero -> returns None (no custom portfolio).
    * If any weight is provided but they don't ALL sum to ~100 -> returns None (ignored).
    * Only returns weights when ALL tickers have a weight and sum is ~100 (+-5).
    """
    parsed = []
    for w in raw_weights[:n_tickers]:
        try:
            v = float(w) if w not in (None, "", 0) else None
        except (ValueError, TypeError):
            v = None
        parsed.append(v)

    # All empty -> no custom portfolio
    if all(v is None for v in parsed):
        return None

    # Any missing -> ignore (user must fill ALL weights)
    if any(v is None for v in parsed):
        return None

    total = sum(parsed)

    # Sum must be close to 100  (tolerance +-5)
    if total <= 0 or abs(total - 100) > 5:
        return None

    # Normalise to sum = 1
    return np.array([v / total for v in parsed])


# ─────────────────────────────────────────────────────────────────
# S&P 500 TICKER LIST  (cached in-memory)
# ─────────────────────────────────────────────────────────────────

_SP500_CACHE = None

def get_sp500_tickers():
    """Fetch S&P 500 constituents from Wikipedia. Cached after first call."""
    global _SP500_CACHE
    if _SP500_CACHE is not None:
        return _SP500_CACHE
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        _SP500_CACHE = tickers
        return tickers
    except Exception:
        # Fallback: a small subset
        _SP500_CACHE = [
            "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","BRK-B",
            "UNH","JNJ","JPM","V","PG","XOM","HD","CVX","MA","ABBV",
            "MRK","KO","PEP","LLY","COST","TMO","AVGO","WMT","MCD",
            "CSCO","ACN","DHR","ABT","VZ","ADBE","CRM","NEE","TXN",
            "PM","RTX","HON","UNP","LOW","INTC","QCOM","INTU","AMAT",
        ]
        return _SP500_CACHE


# ─────────────────────────────────────────────────────────────────
# TICKS.PY UNIVERSE & SECTOR HELPERS
# ─────────────────────────────────────────────────────────────────

_TICKS_CACHE = None

def _clean_yf_ticker(t):
    """Filter out warrants/preferreds/non-ticker strings and format for yfinance."""
    if not isinstance(t, str):
        return None
    t = t.strip().upper()
    if not t or "^" in t or ".WS" in t or "-WS" in t or len(t) > 6 or not t[0].isalpha():
        return None
    # Skip obvious company-name strings accidentally present in ticks.py
    if t in {"IQVIA", "XYLEM", "EDGEWELL", "SPECTRUM", "ENERGIZER", "IDACORP", "GAMING"}:
        return None
    return t.replace(".", "-")


def get_ticks_sectors():
    """Load and sanitize sector ticker lists from ticks.py. Cached in memory."""
    global _TICKS_CACHE
    if _TICKS_CACHE is not None:
        return _TICKS_CACHE
    try:
        import ticks as _ticks_mod
        raw_sectors = getattr(_ticks_mod, "sectors", {})
        cleaned = {}
        sector_map = {}
        for sec, tlist in raw_sectors.items():
            if sec == "Other":
                continue
            seen = set()
            clean_list = []
            for raw_t in tlist:
                ct = _clean_yf_ticker(raw_t)
                if ct and ct not in seen:
                    seen.add(ct)
                    clean_list.append(ct)
                    sector_map.setdefault(ct, sec)
            if clean_list:
                cleaned[sec] = clean_list
        _TICKS_CACHE = (cleaned, sector_map)
    except Exception:
        _TICKS_CACHE = ({}, {})
    return _TICKS_CACHE


def get_n1_universe_options():
    """Return UI options for N+1 universe selector."""
    sectors, _ = get_ticks_sectors()
    opts = [
        {"value": "ticks_top", "label": "ticks.py — Top Liquid All Sectors (~250)"},
        {"value": "sp500",     "label": "S&P 500 Constituents"},
    ]
    for sec, tlist in sectors.items():
        opts.append({"value": f"sector:{sec}", "label": f"ticks.py — {sec} ({min(len(tlist), 120)})"})
    return opts


# ─────────────────────────────────────────────────────────────────
# N+1 SUGGESTION ENGINE  (covariance-based, apples-to-apples)
# ─────────────────────────────────────────────────────────────────

def run_suggestion_engine(returns, backtest_w, bench_ret, days,
                          rf=0.04, new_weight=0.10, top_n=15, lookback_years=3,
                          ranking="ir", universe="ticks_top",
                          base_currency="USD", fx_prices=None):
    """Evaluate candidate assets as the (N+1)th portfolio addition.

    Fixes vs legacy implementation:
      - Evaluates base portfolio & candidates on the EXACT same lookback window
        (apples-to-apples Vol_Change, Sharpe_Change, IR_Change).
      - Never calls global dropna() across all downloaded tickers (which previously
        wiped out the entire DataFrame whenever a single ticker failed or IPO'd late).
      - Ranks ALL valid candidates by the user's chosen preference BEFORE taking top_n.
      - Supports ticks.py multi-sector universes and S&P 500, plus FX conversion.
    """
    import warnings as _w
    import time as _time
    _w.filterwarnings("ignore")
    _t0 = _time.time()

    sectors, sector_map = get_ticks_sectors()
    existing = set(returns.columns.tolist())

    if universe == "sp500":
        pool = get_sp500_tickers()
    elif universe.startswith("sector:"):
        sec_name = universe.split("sector:", 1)[1]
        pool = sectors.get(sec_name, [])[:120]
    else:
        # ticks_top: first 25 flagship tickers from each sector in ticks.py
        pool = []
        seen_p = set()
        for sec, tlist in sectors.items():
            for t in tlist[:25]:
                if t not in seen_p:
                    seen_p.add(t)
                    pool.append(t)
        if not pool:
            pool = get_sp500_tickers()

    candidates = [t for t in pool if t not in existing]
    w_old = np.asarray(backtest_w, dtype=float)
    if len(w_old) > returns.shape[1]:
        w_old = w_old[:returns.shape[1]]
    if w_old.sum() > 0:
        w_old = w_old / w_old.sum()

    # Define exact lookback window for BOTH base portfolio and candidates
    end_date = returns.index[-1].strftime("%Y-%m-%d")
    if lookback_years and lookback_years > 0:
        window_start = returns.index[-1] - pd.DateOffset(years=lookback_years)
        start_ts = max(returns.index[0], window_start)
    else:
        start_ts = returns.index[0]
    start_date = start_ts.strftime("%Y-%m-%d")

    port_recent = returns[returns.index >= start_ts].dropna()
    bench_recent = bench_ret.reindex(port_recent.index).dropna()
    common_base = port_recent.index.intersection(bench_recent.index)
    port_recent = port_recent.loc[common_base]
    bench_recent = bench_recent.loc[common_base]

    if len(port_recent) < 60:
        return {"results": pd.DataFrame(), "base_vol": 0.0,
                "base_ret": 0.0, "base_sharpe": 0.0, "base_ir": 0.0}

    # Base portfolio metrics on the EXACT same lookback window
    mu_base = port_recent.mean().values * days
    cov_base = port_recent.cov().values * days
    base_ret = float(np.dot(w_old, mu_base))
    base_vol = float(np.sqrt(max(float(w_old @ cov_base @ w_old), 1e-14)))
    base_sharpe = (base_ret - rf) / base_vol if base_vol > 0 else 0.0

    port_daily_base = port_recent.values @ w_old
    active_base = port_daily_base - bench_recent.values
    te_base = float(np.std(active_base, ddof=1)) * np.sqrt(days)
    bench_ann_base = float(bench_recent.mean() * days)
    base_ir = (base_ret - bench_ann_base) / te_base if te_base > 1e-10 else 0.0

    print(f"[N+1] Downloading {len(candidates)} candidates ({universe}) from {start_date} to {end_date}...")

    try:
        dl = yf.download(candidates, start=start_date, end=end_date,
                         auto_adjust=True, progress=False, threads=True)
        if dl is None or dl.empty:
            return {"results": pd.DataFrame(), "base_vol": base_vol,
                    "base_ret": base_ret, "base_sharpe": base_sharpe, "base_ir": base_ir}
        raw = dl["Close"] if "Close" in dl else dl
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(name=candidates[0] if candidates else "CAND")
    except Exception:
        return {"results": pd.DataFrame(), "base_vol": base_vol,
                "base_ret": base_ret, "base_sharpe": base_sharpe, "base_ir": base_ir}

    # Compute returns WITHOUT global dropna() so one bad/late ticker never wipes out all others
    cand_returns = raw.ffill(limit=5).pct_change()
    if not isinstance(cand_returns.index, pd.DatetimeIndex):
        cand_returns.index = pd.DatetimeIndex(cand_returns.index)

    # Convert candidate returns from USD to base_currency if needed
    if base_currency != "USD" and fx_prices is not None:
        for col_name in cand_returns.columns:
            cand_returns[col_name] = convert_returns_currency(
                cand_returns[col_name], "USD", base_currency, fx_prices
            )

    scale = 1.0 - new_weight
    w_old_scaled = w_old * scale
    w_new = np.append(w_old_scaled, new_weight)
    min_obs = max(60, int(0.60 * len(port_recent)))

    results = []
    for ticker in cand_returns.columns:
        col = cand_returns[ticker].replace([np.inf, -np.inf], np.nan).dropna()
        if len(col) < min_obs:
            continue
        common_idx = port_recent.index.intersection(col.index)
        if len(common_idx) < min_obs:
            continue

        c_aligned = col.loc[common_idx]
        c_std = float(c_aligned.std(ddof=1))
        if not np.isfinite(c_std) or c_std <= 1e-8 or c_std > 0.25:
            continue

        p_sub = port_recent.loc[common_idx]
        b_sub = bench_recent.loc[common_idx]

        # Recompute base on exact common_idx so delta is 100% exact
        mu_p_sub = p_sub.mean().values * days
        cov_p_sub = p_sub.cov().values * days
        b_ret_i = float(np.dot(w_old, mu_p_sub))
        b_vol_i = float(np.sqrt(max(float(w_old @ cov_p_sub @ w_old), 1e-14)))
        b_sh_i = (b_ret_i - rf) / b_vol_i if b_vol_i > 0 else 0.0

        combined_vals = np.column_stack([p_sub.values, c_aligned.values])
        mu_comb = combined_vals.mean(axis=0) * days
        cov_comb = np.cov(combined_vals, rowvar=False, ddof=1) * days

        new_ret = float(np.dot(w_new, mu_comb))
        new_vol = float(np.sqrt(max(float(w_new @ cov_comb @ w_new), 1e-14)))
        new_sharpe = (new_ret - rf) / new_vol if new_vol > 0 else 0.0

        # Information Ratio on exact same dates
        port_daily_new = combined_vals @ w_new
        active_new = port_daily_new - b_sub.values
        te_new = float(np.std(active_new, ddof=1)) * np.sqrt(days)
        bench_ann_i = float(b_sub.mean() * days)
        ir = (new_ret - bench_ann_i) / te_new if te_new > 1e-10 else 0.0

        # Correlation with existing portfolio (actual portfolio series, not just avg asset)
        port_daily_old = p_sub.values @ w_old
        port_corr = float(np.corrcoef(port_daily_old, c_aligned.values)[0, 1])
        if not np.isfinite(port_corr):
            continue

        results.append({
            "Ticker": ticker,
            "Sector": sector_map.get(ticker, "Other"),
            "New_Vol": new_vol,
            "New_Ret": new_ret,
            "New_Sharpe": new_sharpe,
            "IR": ir,
            "Avg_Corr": port_corr,
            "Vol_Change": new_vol - b_vol_i,
            "Sharpe_Change": new_sharpe - b_sh_i,
        })

    if not results:
        return {"results": pd.DataFrame(), "base_vol": base_vol,
                "base_ret": base_ret, "base_sharpe": base_sharpe, "base_ir": base_ir,
                "elapsed_s": round(_time.time() - _t0, 1),
                "n_candidates_evaluated": 0, "start_date": start_date}

    df = pd.DataFrame(results)

    # Identify top-3 winners per criterion across ALL evaluated candidates
    top_min_vol = set(df.nsmallest(5, "New_Vol")["Ticker"].tolist())
    top_best_ir = set(df.nlargest(5, "IR")["Ticker"].tolist())
    top_best_sh = set(df.nlargest(5, "New_Sharpe")["Ticker"].tolist())
    top_min_corr = set(df.nsmallest(5, "Avg_Corr")["Ticker"].tolist())

    def assign_badge(row):
        badges = []
        t = row["Ticker"]
        if t in top_best_ir:  badges.append("↑ Best IR")
        if t in top_best_sh:  badges.append("↑ Best Sharpe")
        if t in top_min_vol:  badges.append("↓ Min Vol")
        if t in top_min_corr: badges.append("↓ Min Corr")
        return " | ".join(badges)

    df["Badge"] = df.apply(assign_badge, axis=1)

    # Sort ALL evaluated candidates strictly by user's selected ranking BEFORE truncating
    if ranking == "vol":
        df = df.sort_values("New_Vol", ascending=True)
    elif ranking == "diversify":
        df = df.sort_values("Avg_Corr", ascending=True)
    elif ranking == "sharpe":
        df = df.sort_values("New_Sharpe", ascending=False)
    else:  # "ir"
        df = df.sort_values("IR", ascending=False)

    result_df = df.head(top_n).reset_index(drop=True)
    elapsed = round(_time.time() - _t0, 1)
    print(f"[N+1] Completed in {elapsed}s — {len(result_df)} shown from {len(results)} evaluated (ranking={ranking})")
    return {
        "results": result_df,
        "base_vol": base_vol,
        "base_ret": base_ret,
        "base_sharpe": base_sharpe,
        "base_ir": base_ir,
        "elapsed_s": elapsed,
        "n_candidates_evaluated": len(results),
        "start_date": start_date,
        "lookback_years": lookback_years,
        "universe": universe,
    }
