"""
╔══════════════════════════════════════════════════════════════════╗
║        FastbreaK PORTFOLIO DASHBOARD                             ║
║        Interactive Dash Application                              ║
╚══════════════════════════════════════════════════════════════════╝

Run with:  python portfolio_dashboard.py
Then open: http://127.0.0.1:8050

Requirements:
    pip install dash dash-bootstrap-components yfinance plotly numpy pandas
"""

# ─────────────────────────────────────────────────────────────────
# DEPENDENCIES
# ─────────────────────────────────────────────────────────────────
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from portfolio_helpers import (MAX_TICKERS, DEFAULT_TICKERS, CURRENCY_OPTIONS,
                               CURRENCY_SYMBOLS, download_fx_rate,
                               convert_all_returns, fig_cumulative_multi,
                               parse_custom_weights, run_suggestion_engine)

# ─────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (style-guide.md)
# ─────────────────────────────────────────────────────────────────
C_BLUE   = "#0a84ff"    # --accent
C_ORANGE = "#ff9f0a"    # benchmark / secondary
C_GREEN  = "#30d158"    # --accent2 / positive PnL
C_RED    = "#ff453a"    # --warn / negative PnL
C_PURPLE = "#bf5af2"    # secondary accent
C_CYAN   = "#64d2ff"    # info/highlight
C_YELLOW = "#ffd60a"    # tertiary
C_DARK   = "#0a0a0f"    # --bg
C_SURFACE= "#111118"    # --surface (panels, sidebars)
C_CARD   = "#13131c"    # --card-bg
C_BORDER = "#1e1e2e"    # --border
C_TEXT   = "#e8e8f0"    # --text
C_MUTED  = "#555570"    # --muted

TEMPLATE = "plotly_dark"
RF_RATE  = 0.04   # Risk-free rate for Sharpe/Sortino

# ─────────────────────────────────────────────────────────────────
# CORE QUANT FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def download_data(tickers, benchmark, start_date, end_date):
    all_symbols = list(set(tickers + [benchmark]))
    data = yf.download(all_symbols, start=start_date, end=end_date,
                       auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    # Record first valid date per ticker BEFORE dropna (raw history length)
    ticker_start = {}
    for t in tickers:
        if t in data.columns:
            fv = data[t].first_valid_index()
            ticker_start[t] = fv
    returns = data.ffill().pct_change().dropna()
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.DatetimeIndex(returns.index)
    bench_series = returns[benchmark]
    asset_returns = returns.drop(columns=[benchmark])
    asset_returns = asset_returns[[c for c in tickers if c in asset_returns.columns]]
    return asset_returns, bench_series, ticker_start


def markowitz_optimize(returns, days, n_portfolios=8000, rf=None):
    if rf is None: rf = RF_RATE
    expected_returns = returns.mean() * days
    cov_matrix = returns.cov() * days
    num_assets = len(returns.columns)

    np.random.seed(42)
    results = np.zeros((3, n_portfolios))
    weights_record = []

    for i in range(n_portfolios):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        weights_record.append(w)
        p_ret = np.dot(w, expected_returns)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        results[0, i] = p_vol
        results[1, i] = p_ret
        results[2, i] = (p_ret - rf) / p_vol if p_vol > 0 else 0

    max_sharpe_idx = np.argmax(results[2, :])
    optimal_weights = weights_record[max_sharpe_idx]
    return optimal_weights, results, weights_record, max_sharpe_idx


def walk_forward_optimize(returns, days, window_5y=252*5, reopt_step=63, n_port=2000, rf=None):
    if rf is None: rf = RF_RATE
    rolling_results, rolling_dates, rolling_sharpes = [], [], []

    for i in range(window_5y, len(returns), reopt_step):
        window_data = returns.iloc[i - window_5y:i]
        exp_ret = window_data.mean() * days
        cov_mat = window_data.cov() * days

        best_sharpe, best_w = -np.inf, None
        for _ in range(n_port):
            w = np.random.random(len(returns.columns))
            w /= np.sum(w)
            p_ret = np.dot(w, exp_ret)
            p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
            sharpe = (p_ret - rf) / p_vol if p_vol > 0 else 0
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_w = w

        rolling_results.append(best_w)
        rolling_dates.append(returns.index[i])
        rolling_sharpes.append(best_sharpe)

    rolling_df = pd.DataFrame(rolling_results, index=rolling_dates, columns=returns.columns)
    return rolling_df, np.array(rolling_sharpes)


def run_simulation(returns, target_weights, rebalance_months):
    if returns.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    current_weights = target_weights.copy()
    weights_history, portfolio_returns = [], []

    first_days = (returns.groupby([returns.index.year, returns.index.month])
                  .apply(lambda x: x.index[0] if len(x) > 0 else None)
                  .dropna().values)

    for date, daily_ret in returns.iterrows():
        if date in first_days and date.month in rebalance_months:
            current_weights = target_weights.copy()
        weights_history.append(current_weights.copy())
        daily_port_ret = np.dot(daily_ret.values, current_weights)
        portfolio_returns.append(daily_port_ret)
        denom = 1 + daily_port_ret
        if denom != 0:
            current_weights = current_weights * (1 + daily_ret.values) / denom
        else:
            current_weights = target_weights.copy()

    wh = pd.DataFrame(weights_history, index=returns.index, columns=returns.columns)
    pr = pd.Series(portfolio_returns, index=returns.index)
    return wh, pr


def get_drawdown_series(returns_series):
    cum = (1 + returns_series).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak


def get_recovery_durations(returns_series):
    dd = get_drawdown_series(returns_series)
    is_uw = dd < 0
    groups = (is_uw != is_uw.shift()).cumsum()
    durations = is_uw.groupby(groups).sum()
    return durations[durations > 0].tolist()


def get_max_drawdown(series):
    if series.empty:
        return 0
    cum = (1 + series).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return dd.min()


def get_rolling_annualized(series, window_days, years):
    rolling_total = series.rolling(window=int(window_days)).apply(
        lambda x: (1 + x).prod() - 1, raw=False)
    return (1 + rolling_total) ** (1 / years) - 1


def compute_metrics(p_ret, bench_ret, days, label="Portfolio", rf=None):
    if rf is None:
        rf = RF_RATE

    # Use the full portfolio series for own metrics — no bench alignment needed
    p = p_ret.dropna()

    ann_ret = (1 + p).prod() ** (days / len(p)) - 1
    ann_vol = p.std() * np.sqrt(days)          # ddof=1 (Bessel), standard
    sharpe  = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0

    # Sortino: downside deviation below daily rf hurdle
    rf_daily = (1 + rf) ** (1 / days) - 1     # geometric daily equivalent
    excess   = p - rf_daily
    downside_sq = excess[excess < 0] ** 2
    down_std = np.sqrt(downside_sq.mean()) * np.sqrt(days) if len(downside_sq) > 0 else ann_vol
    sortino  = (ann_ret - rf) / down_std if down_std > 0 else 0

    max_dd  = get_max_drawdown(p)
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Beta / Jensen's Alpha — align both series on common dates
    common   = p.index.intersection(bench_ret.dropna().index)
    p_c      = p.loc[common]
    b_c      = bench_ret.loc[common]
    cov_mat  = np.cov(p_c.values, b_c.values)
    beta     = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 0 else 0
    bench_ann = (1 + b_c).prod() ** (days / len(b_c)) - 1
    alpha    = ann_ret - (rf + beta * (bench_ann - rf))   # Jensen's Alpha

    ann_ret_series = p.groupby(p.index.year).apply(lambda x: (1 + x).prod() - 1)
    win_rate = (ann_ret_series > 0).mean()

    var_95  = float(np.percentile(p.values, 5))
    cvar_95 = float(p[p <= var_95].mean())

    durations    = [d for d in get_recovery_durations(p) if d > 2]
    avg_recovery = np.mean(durations) if durations else 0
    max_recovery = max(durations) if durations else 0

    return dict(label=label, ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                sortino=sortino, max_dd=max_dd, beta=beta, alpha=alpha,
                calmar=calmar, win_rate=win_rate, var_95=var_95, cvar_95=cvar_95,
                avg_recovery=avg_recovery, max_recovery=max_recovery)


# ─────────────────────────────────────────────────────────────────
# SERIALISE / DESERIALISE  (preserves column order & DatetimeIndex)
# ─────────────────────────────────────────────────────────────────

def df_to_store(df):
    return {"data": df.to_json(date_format="iso"), "columns": list(df.columns)}

def df_from_store(d):
    df = pd.read_json(d["data"])
    df = df[d["columns"]]
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df

def series_to_store(s, name="value"):
    return {"data": s.to_json(date_format="iso"), "name": s.name or name}

def series_from_store(d):
    s = pd.read_json(d["data"], typ="series")
    s.index = pd.to_datetime(s.index)
    return s


# ─────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────

def fig_efficient_frontier(results, max_sharpe_idx, rf=RF_RATE,
                           extra_markers=None, weights_record=None,
                           expected_returns=None, cov_matrix=None):
    """
    Efficient frontier with:
    - Viridis scatter of simulated portfolios
    - Fitted parabolic frontier curve
    - Capital Market Line (CML)
    - Global Minimum Variance (GMV) portfolio marker
    - Max Sharpe marker
    - Optional extra markers (WF, Custom)
    - Tooltip allocation for key portfolios
    - Non-overlapping labels via textposition logic
    """
    vols   = results[0, :]
    rets   = results[1, :]
    sharpe = results[2, :]

    fig = go.Figure()

    # ── Scatter cloud ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=vols, y=rets, mode="markers",
        marker=dict(size=3, color=sharpe, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Sharpe", thickness=12),
                    opacity=0.6),
        text=[f"Sharpe: {s:.2f}" for s in sharpe],
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<br>%{text}<extra></extra>",
        name="Simulated portfolios", showlegend=True,
    ))

    # ── Fit a parabola to the upper frontier (Vol, Ret²) ───────
    try:
        from numpy.polynomial import polynomial as P
        sort_idx = np.argsort(vols)
        v_s, r_s = vols[sort_idx], rets[sort_idx]
        # Bin into deciles and take 90th percentile of ret in each bin
        n_bins = 30
        bins   = np.array_split(np.arange(len(v_s)), n_bins)
        v_env  = np.array([v_s[b].mean() for b in bins if len(b) > 0])
        r_env  = np.array([np.percentile(r_s[b], 90) for b in bins if len(b) > 0])
        # Fit degree-2 polynomial in vol space
        coeffs = np.polyfit(v_env, r_env, 2)
        v_fit  = np.linspace(v_s.min(), v_s.max(), 400)
        r_fit  = np.polyval(coeffs, v_fit)
        # Only show the upper part (positive slope region), exclude rightmost 5%
        slope  = np.gradient(r_fit)
        v_cutoff = np.percentile(v_s, 95)
        mask   = (slope > 0) & (v_fit <= v_cutoff)
        if mask.sum() > 5:
            fig.add_trace(go.Scatter(
                x=v_fit[mask], y=r_fit[mask], mode="lines",
                line=dict(color="rgba(255,255,255,0.40)", width=2, dash="dash"),
                name="Efficient Frontier", hoverinfo="skip",
            ))
    except Exception:
        pass   # if fitting fails, skip silently

    # ── Global Minimum Variance (GMV) ──────────────────────────
    gmv_idx = np.argmin(vols)
    fig.add_trace(go.Scatter(
        x=[vols[gmv_idx]], y=[rets[gmv_idx]], mode="markers+text",
        marker=dict(color=C_CYAN, size=14, symbol="circle",
                    line=dict(width=2, color="white")),
        text=["GMV"], textposition="top right",
        textfont=dict(color=C_CYAN, size=11),
        name="Global Min Variance",
        hovertemplate=(f"<b>Global Min Variance</b><br>"
                       f"Vol: {vols[gmv_idx]:.2%}<br>"
                       f"Ret: {rets[gmv_idx]:.2%}<extra></extra>"),
    ))

    # ── Capital Market Line (CML) ───────────────────────────────
    ms_vol = vols[max_sharpe_idx]
    ms_ret = rets[max_sharpe_idx]
    slope_cml = (ms_ret - rf) / ms_vol if ms_vol > 0 else 0
    v_max_cml  = vols.max() * 1.15
    cml_x = [0.0, v_max_cml]
    cml_y = [rf, rf + slope_cml * v_max_cml]
    fig.add_trace(go.Scatter(
        x=cml_x, y=cml_y, mode="lines",
        line=dict(color=C_YELLOW, width=1.8, dash="dot"),
        name=f"CML (slope={slope_cml:.2f})",
        hoverinfo="skip",
    ))
    # RF intercept annotation
    fig.add_annotation(
        x=0, y=rf, text=f"RF={rf:.1%}", showarrow=False,
        font=dict(color=C_YELLOW, size=10), xanchor="left", yanchor="bottom",
    )

    # ── Max Sharpe marker ───────────────────────────────────────
    # Build allocation tooltip
    alloc_tip = ""
    if weights_record is not None and expected_returns is not None:
        w = weights_record[max_sharpe_idx]
        top3 = sorted(enumerate(w), key=lambda x: -x[1])[:3]
        cols  = list(expected_returns.index)
        alloc_tip = "  |  ".join(f"{cols[j]}: {v*100:.0f}%" for j, v in top3)

    fig.add_trace(go.Scatter(
        x=[ms_vol], y=[ms_ret], mode="markers+text",
        marker=dict(color=C_RED, size=18, symbol="star",
                    line=dict(width=2, color="white")),
        text=["Max Sharpe"], textposition="top left",
        textfont=dict(color=C_RED, size=11),
        name="Max Sharpe",
        hovertemplate=(f"<b>Max Sharpe</b><br>"
                       f"Vol: {ms_vol:.2%}<br>Ret: {ms_ret:.2%}<br>"
                       f"Sharpe: {sharpe[max_sharpe_idx]:.2f}<br>"
                       + (f"Top3: {alloc_tip}" if alloc_tip else "")
                       + "<extra></extra>"),
    ))

    # ── Extra markers (WF, Custom) ──────────────────────────────
    if extra_markers:
        label_positions = ["top right", "bottom left", "top left", "bottom right"]
        for k, (label, ex_vol, ex_ret, color, symbol) in enumerate(extra_markers):
            pos = label_positions[k % len(label_positions)]
            fig.add_trace(go.Scatter(
                x=[ex_vol], y=[ex_ret], mode="markers+text",
                marker=dict(color=color, size=16, symbol=symbol,
                            line=dict(width=2, color="white")),
                text=[label], textposition=pos,
                textfont=dict(color=color, size=11),
                name=label,
                hovertemplate=(f"<b>{label}</b><br>"
                               f"Vol: {ex_vol:.2%}<br>Ret: {ex_ret:.2%}<extra></extra>"),
            ))

    fig.update_layout(
        template=TEMPLATE, height=620,
        title="Efficient Frontier · CML · Key Portfolios  ·  coordinates: μ·d vs σ·√d (same as MC cloud)",
        xaxis=dict(title="Annualised Volatility (σ·√252)", tickformat=".0%"),
        yaxis=dict(title="Annualised Return (μ·252)",      tickformat=".0%"),
        hovermode="closest",
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        legend=dict(
            orientation="h", x=0, y=-0.18, xanchor="left", yanchor="top",
            bgcolor="rgba(17,17,24,0.9)", bordercolor=C_BORDER, borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(t=60, b=160, r=40),
    )
    return fig


def fig_donut(weights, columns, title="Allocation"):
    df = pd.DataFrame({"Asset": columns, "Weight": weights * 100})
    df = df[df["Weight"] > 0.01].sort_values("Weight", ascending=False)
    fig = px.pie(df, values="Weight", names="Asset", hole=0.45,
                 color_discrete_sequence=px.colors.qualitative.Pastel, title=title)
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      marker=dict(line=dict(color=C_DARK, width=2)))
    fig.update_layout(template=TEMPLATE, height=460, paper_bgcolor=C_CARD,
                      annotations=[dict(text=f"{len(df)}<br>assets", x=0.5, y=0.5,
                                        font_size=16, showarrow=False, font_color=C_TEXT)])
    return fig


def fig_cumulative(p_returns, benchmark_returns, results, max_sharpe_idx):
    opt_cum = (1 + p_returns).cumprod()
    spy_cum = (1 + benchmark_returns.reindex(p_returns.index).dropna()).cumprod()

    start  = opt_cum.index[0]
    x_yrs  = (opt_cum.index - start).total_seconds() / (365.25 * 86400)
    growth_rate = results[1, max_sharpe_idx]
    power_fn    = (1 + growth_rate) ** x_yrs

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=opt_cum.index, y=opt_cum, name="Portfolio",
                             line=dict(color=C_BLUE, width=2.5)))
    fig.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum, name="S&P 500",
                             line=dict(color=C_ORANGE, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=opt_cum.index, y=power_fn,
                             name=f"Theoretical {growth_rate*100:.1f}%/yr",
                             line=dict(color=C_GREEN, width=1.5, dash="dash")))
    fig.update_layout(template=TEMPLATE, height=480,
                      title="Cumulative Growth (log scale)",
                      yaxis_type="log", hovermode="x unified",
                      paper_bgcolor=C_CARD, plot_bgcolor=C_CARD)
    return fig


def fig_pnl_vs_benchmark(p_returns, benchmark_returns, initial=10000, cs="$"):
    common = p_returns.index.intersection(benchmark_returns.index)
    p = p_returns.loc[common]
    b = benchmark_returns.loc[common]

    port_val  = (1 + p).cumprod() * initial
    bench_val = (1 + b).cumprod() * initial
    spread    = port_val - bench_val

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45],
                        vertical_spacing=0.04,
                        subplot_titles=(f"Portfolio Value ({cs})", f"Alpha Spread vs Benchmark ({cs})"))

    fig.add_trace(go.Scatter(x=port_val.index, y=port_val, name="Portfolio",
                             line=dict(color=C_BLUE, width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bench_val.index, y=bench_val, name="Benchmark",
                             line=dict(color=C_ORANGE, width=2, dash="dot")), row=1, col=1)

    # Outperformance fill
    pos = spread.copy(); pos[pos < 0] = 0
    neg = spread.copy(); neg[neg > 0] = 0
    fig.add_trace(go.Scatter(x=pos.index, y=pos.values, fill="tozeroy",
                             name="Outperformance", mode="none",
                             fillcolor="rgba(46,204,113,0.40)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=neg.index, y=neg.values, fill="tozeroy",
                             name="Underperformance", mode="none",
                             fillcolor="rgba(231,76,60,0.40)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=spread.index, y=spread.values, name="Spread",
                             mode="lines", showlegend=False,
                             line=dict(color="white", width=0.8, dash="dot")), row=2, col=1)

    fig.update_layout(template=TEMPLATE, height=690,
                      title=dict(text=f"P&L vs Benchmark  ·  {cs}{initial:,.0f} initial investment",
                                 y=0.98, yanchor="top"),
                      hovermode="x unified", paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
                      legend=dict(orientation="h", y=-0.08, yanchor="top", x=0),
                      margin=dict(t=90, b=75))
    fig.update_yaxes(tickprefix=cs, row=1, col=1)
    fig.update_yaxes(tickprefix=cs, row=2, col=1)
    return fig


def fig_weights_drift(weights_history_df):
    """Stacked area chart of actual drifting weights over time."""
    wh = weights_history_df.copy()
    wh.index = pd.to_datetime(wh.index)
    wh.index.name = "Date"
    melted = wh.reset_index().melt(id_vars="Date", var_name="Asset", value_name="Weight")
    fig = px.area(melted, x="Date", y="Weight", color="Asset",
                  title="Portfolio Composition Over Time  ·  Actual Weights (Drift & Rebalancing)",
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(template=TEMPLATE, height=540,
                      yaxis=dict(tickformat=".0%", range=[0, 1], title="Weight"),
                      hovermode="x unified", paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
                      legend=dict(orientation="h", y=-0.18, x=0),
                      margin=dict(b=100))
    return fig


def fig_individual_pnl(returns, initial=10000, cs="$"):
    """Cumulative P&L of each individual asset from same starting amount."""
    cum = (1 + returns).cumprod() * initial
    colors = px.colors.qualitative.Pastel
    fig = go.Figure()
    for i, col in enumerate(cum.columns):
        fig.add_trace(go.Scatter(x=cum.index, y=cum[col], name=col, mode="lines",
                                 line=dict(color=colors[i % len(colors)], width=2)))
    fig.update_layout(template=TEMPLATE, height=500,
                      title=f"Individual Asset P&L  ·  {cs}{initial:,.0f} per asset (log scale)",
                      yaxis_type="log", yaxis=dict(tickprefix=cs),
                      hovermode="x unified", paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
                      legend=dict(orientation="h", y=-0.15))
    return fig


def fig_drawdown(p_returns, benchmark_returns):
    dd_opt = get_drawdown_series(p_returns)
    dd_spy = get_drawdown_series(benchmark_returns.reindex(p_returns.index))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd_opt.index, y=dd_opt * 100, fill="tozeroy",
                             name="Portfolio", line=dict(color=C_BLUE, width=1),
                             fillcolor="rgba(65,105,225,0.25)"))
    fig.add_trace(go.Scatter(x=dd_spy.index, y=dd_spy * 100, name="S&P 500",
                             mode="lines", line=dict(color=C_ORANGE, width=1.5)))
    fig.update_layout(template=TEMPLATE, height=620,
                      title="Underwater Chart – Drawdown from Peak (%)",
                      yaxis=dict(ticksuffix="%"), hovermode="x unified",
                      paper_bgcolor=C_CARD, plot_bgcolor=C_CARD)
    return fig


def fig_annual_bars(p_returns, benchmark_returns):
    ann_opt = p_returns.groupby(p_returns.index.year).apply(
        lambda x: (1 + x).prod() - 1)
    ann_spy = benchmark_returns.groupby(benchmark_returns.index.year).apply(
        lambda x: (1 + x).prod() - 1)
    ann_opt, ann_spy = ann_opt.align(ann_spy, join="inner")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ann_opt.index, y=ann_opt * 100, name="Portfolio",
                         marker_color=C_BLUE))
    fig.add_trace(go.Bar(x=ann_spy.index, y=ann_spy * 100, name="S&P 500",
                         marker_color=C_ORANGE, opacity=0.75))
    fig.add_hline(y=ann_opt.mean() * 100, line_dash="dot", line_color=C_BLUE,
                  annotation_text=f"Avg Port {ann_opt.mean()*100:.1f}%")
    fig.add_hline(y=ann_spy.mean() * 100, line_dash="dot", line_color=C_ORANGE,
                  annotation_text=f"Avg Benchmark {ann_spy.mean()*100:.1f}%")
    fig.update_layout(template=TEMPLATE, height=540, barmode="group",
                      title="Annual Returns (%)", paper_bgcolor=C_CARD, plot_bgcolor=C_CARD)
    return fig



def fig_annual_max_drawdown(p_returns, benchmark_returns):
    """Annual max drawdown: worst drawdown from Jan 1st each calendar year."""
    def annual_mdd(series):
        result = {}
        for yr, grp in series.groupby(series.index.year):
            cum = (1 + grp).cumprod()
            peak = cum.cummax()
            dd = (cum - peak) / peak
            result[yr] = float(dd.min()) * 100  # negative %
        return pd.Series(result)

    mdd_port  = annual_mdd(p_returns)
    mdd_bench = annual_mdd(benchmark_returns.reindex(p_returns.index).dropna())
    mdd_port, mdd_bench = mdd_port.align(mdd_bench, join="inner")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mdd_port.index, y=mdd_port,
        name="Portfolio", marker_color=C_RED,
    ))
    fig.add_trace(go.Bar(
        x=mdd_bench.index, y=mdd_bench,
        name="Benchmark", marker_color="#8B0000", opacity=0.8,
    ))

    # --- THE FIX: Calculate means from the MDD series specifically ---
    avg_mdd_port = mdd_port.mean()
    avg_mdd_bench = mdd_bench.mean()

    fig.add_hline(y=avg_mdd_port, line_dash="dot", line_color=C_RED,
                  annotation_text=f"Avg Port MDD {avg_mdd_port:.1f}%")
    fig.add_hline(y=avg_mdd_bench, line_dash="dot", line_color="#8B0000",
                  annotation_text=f"Avg Bench MDD {avg_mdd_bench:.1f}%")

    fig.update_layout(
        template=TEMPLATE, height=460, barmode="group",
        title="Annual Max Drawdown (%) — worst intra-year drawdown from Jan 1st",
        yaxis=dict(ticksuffix="%"),
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
    )
    return fig


def fig_recovery_times(p_returns, benchmark_returns, benchmark_label="^GSPC"):
    """Two-panel recovery analysis: summary stats bar + ECDF of durations."""
    d_opt = sorted([d for d in get_recovery_durations(p_returns) if d > 2])
    d_spy = sorted([d for d in get_recovery_durations(
        benchmark_returns.reindex(p_returns.index)) if d > 2])

    def stats(d):
        if not d: return 0, 0, 0, 0
        return np.mean(d), np.median(d), np.percentile(d, 75), max(d)

    m_o, med_o, p75_o, mx_o = stats(d_opt)
    m_s, med_s, p75_s, mx_s = stats(d_spy)

    # ── Panel 1: Summary stats as grouped bars ──
    categories = ["Mean", "Median", "75th pct", "Max"]
    port_vals  = [m_o,  med_o,  p75_o,  mx_o]
    bench_vals = [m_s,  med_s,  p75_s,  mx_s]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Recovery Time Statistics (days)",
                                        "Empirical CDF of Recovery Durations"),
                        column_widths=[0.45, 0.55])

    fig.add_trace(go.Bar(y=categories, x=port_vals, orientation="h",
                         name="Portfolio", marker_color=C_BLUE,
                         text=[f"{v:.0f}d" for v in port_vals],
                         textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(y=categories, x=bench_vals, orientation="h",
                         name=f"Benchmark", marker_color=C_ORANGE,
                         text=[f"{v:.0f}d" for v in bench_vals],
                         textposition="outside"), row=1, col=1)

    # ── Panel 2: ECDF ──
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    if d_opt:
        x_o, y_o = ecdf(d_opt)
        fig.add_trace(go.Scatter(x=x_o, y=y_o * 100, name="Portfolio ECDF",
                                 mode="lines", line=dict(color=C_BLUE, width=2.5),
                                 showlegend=False), row=1, col=2)
    if d_spy:
        x_s, y_s = ecdf(d_spy)
        fig.add_trace(go.Scatter(x=x_s, y=y_s * 100, name="Benchmark ECDF",
                                 mode="lines", line=dict(color=C_ORANGE, width=2.5,
                                 dash="dot"), showlegend=False), row=1, col=2)

    # Reference lines on ECDF at 50% and 90%
    for pct, col in [(50, "rgba(255,255,255,0.3)"), (90, "rgba(255,255,255,0.2)")]:
        fig.add_hline(y=pct, line_dash="dash", line_color=col,
                      annotation_text=f"{pct}%", annotation_position="right",
                      row=1, col=2)

    x_max_bar = max(mx_o, mx_s) * 1.4 if max(mx_o, mx_s) > 0 else 300
    fig.update_xaxes(title_text="Trading days", range=[0, x_max_bar], row=1, col=1)
    fig.update_xaxes(title_text="Trading days to recover", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative % of drawdowns", row=1, col=2)

    fig.update_layout(
        template=TEMPLATE, height=460,
        title=dict(
            text=f"Recovery Analysis  ·  Portfolio avg <b>{m_o:.0f}d</b>  vs  Benchmark avg <b>{m_s:.0f}d</b>",
            y=0.98, yanchor="top",
        ),
        barmode="group", hovermode="x unified",
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        legend=dict(orientation="h", y=-0.20, yanchor="top", x=0),
        margin=dict(r=60, t=95, b=80),
    )
    return fig


def fig_var_chart(p_returns, benchmark_returns, days, confidence=0.95):
    window = days
    roll_var_p = p_returns.rolling(window).quantile(1 - confidence)
    roll_var_b = benchmark_returns.reindex(p_returns.index).rolling(window).quantile(1 - confidence)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roll_var_p.index, y=roll_var_p * 100,
                             name=f"Portfolio VaR {confidence:.0%}",
                             line=dict(color=C_BLUE, width=2)))
    fig.add_trace(go.Scatter(x=roll_var_b.index, y=roll_var_b * 100,
                             name=f"Benchmark VaR {confidence:.0%}",
                             line=dict(color=C_ORANGE, width=2, dash="dot")))
    fig.update_layout(template=TEMPLATE, height=520,
                      title=f"Rolling 1Y Historical VaR ({confidence:.0%} confidence)",
                      yaxis_title="Daily VaR (%)", yaxis=dict(ticksuffix="%"),
                      xaxis=dict(autorange=True),
                      hovermode="x unified", paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
                      margin=dict(l=60, r=20, t=50, b=40))
    return fig


def fig_rolling_returns(p_returns, benchmark_returns, days):
    periods = {"1Y": (days, 1), "3Y": (days*3, 3), "5Y": (days*5, 5),
               "10Y": (days*10, 10), "15Y": (days*15, 15), "20Y": (days*20, 20)}
    fig = go.Figure()
    for i, (name, (d_win, yrs)) in enumerate(periods.items()):
        r_opt = get_rolling_annualized(p_returns, d_win, yrs)
        r_spy = get_rolling_annualized(benchmark_returns.reindex(p_returns.index), d_win, yrs)
        fig.add_trace(go.Scatter(x=r_opt.index, y=r_opt*100, name=f"Portfolio ({name})",
                                 visible=(i == 0), line=dict(color=C_BLUE, width=2.5)))
        fig.add_trace(go.Scatter(x=r_spy.index, y=r_spy*100, name=f"Benchmark ({name})",
                                 visible=(i == 0), line=dict(color=C_ORANGE, width=2, dash="dot")))
    buttons = []
    for i, name in enumerate(periods.keys()):
        vis = [False] * (len(periods) * 2)
        vis[i*2] = True; vis[i*2+1] = True
        buttons.append(dict(label=name, method="update",
                            args=[{"visible": vis}, {"title": f"Rolling Annualised – {name}"}]))
    fig.update_layout(template=TEMPLATE, height=500,
                      updatemenus=[dict(active=0, buttons=buttons, x=1.0,
                                        xanchor="left", y=1.15, yanchor="top", showactive=True)],
                      title="Rolling Annualised Returns", yaxis_title="Annualised Return (%)",
                      hovermode="x unified", paper_bgcolor=C_CARD, plot_bgcolor=C_CARD)
    fig.add_hline(y=0, line_color="white", line_width=0.5, opacity=0.3)
    return fig


def fig_rolling_sharpe(p_returns, benchmark_returns, days, rf=RF_RATE):
    window = days
    def rolling_sharpe(ret):
        rr = ret.rolling(window).apply(lambda x: (1+x).prod()**(days/window)-1, raw=False)
        rv = ret.rolling(window).std() * np.sqrt(days)
        return (rr - rf) / rv
    rs_p = rolling_sharpe(p_returns)
    rs_b = rolling_sharpe(benchmark_returns.reindex(p_returns.index))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs_p.index, y=rs_p, name="Portfolio",
                             line=dict(color=C_BLUE, width=2)))
    fig.add_trace(go.Scatter(x=rs_b.index, y=rs_b, name="Benchmark",
                             line=dict(color=C_ORANGE, width=2, dash="dot")))
    fig.add_hline(y=0, line_color="white", opacity=0.2)
    fig.add_hline(y=1, line_color=C_GREEN, line_dash="dash", opacity=0.4,
                  annotation_text="Sharpe = 1")
    fig.update_layout(template=TEMPLATE, height=500, title="Rolling 1Y Sharpe Ratio",
                      yaxis_title="Sharpe Ratio", hovermode="x unified",
                      paper_bgcolor=C_CARD, plot_bgcolor=C_CARD)
    return fig


def fig_rolling_heatmap(rolling_weights_df):
    fig = px.imshow(rolling_weights_df.T,
                    labels=dict(x="Optimisation Date", y="Asset", color="Weight"),
                    color_continuous_scale="YlGnBu",
                    title="5Y Walk-Forward: Rolling Weights Regime Heatmap")
    fig.update_layout(template=TEMPLATE, height=560, paper_bgcolor=C_CARD)
    return fig


def fig_correlation_matrix(returns):
    corr = returns.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_display = corr.where(~mask)
    text_arr = np.where(mask, "", np.round(corr.values, 2))
    fig = go.Figure(go.Heatmap(
        z=corr_display.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=text_arr, texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"
    ))
    fig.update_layout(template=TEMPLATE, height=600, title="Asset Correlation Matrix",
                      paper_bgcolor=C_CARD, plot_bgcolor=C_CARD)
    return fig


def fig_return_distributions(p_returns, benchmark_returns, days):
    periods_list = [("1Y", days, 1), ("3Y", days*3, 3),
                    ("5Y", days*5, 5), ("10Y", days*10, 10)]
    fig = go.Figure()
    all_data = []
    for name, d_win, yrs in periods_list:
        r_opt = get_rolling_annualized(p_returns, d_win, yrs).dropna() * 100
        r_spy = get_rolling_annualized(
            benchmark_returns.reindex(p_returns.index), d_win, yrs).dropna() * 100
        if len(r_opt) > 10 and len(r_spy) > 10:
            try:
                tmp = ff.create_distplot([r_opt.values, r_spy.values],
                                         [f"Portfolio ({name})", f"Benchmark ({name})"],
                                         bin_size=0.5, curve_type="kde",
                                         show_hist=True, show_rug=False,
                                         colors=[C_BLUE, C_ORANGE])
                all_data.append(tmp.data)
            except Exception:
                all_data.append(None)
        else:
            all_data.append(None)

    traces_per_period = []
    idx = 0
    for period_data in all_data:
        if period_data:
            for trace in period_data:
                trace.visible = False
                fig.add_trace(trace)
            traces_per_period.append((idx, idx + len(period_data)))
            idx += len(period_data)
        else:
            traces_per_period.append(None)

    if traces_per_period and traces_per_period[0]:
        for k in range(*traces_per_period[0]):
            fig.data[k].visible = True

    buttons = []
    for i, (name, _, _) in enumerate(periods_list):
        vis = [False] * len(fig.data)
        if traces_per_period[i]:
            for k in range(*traces_per_period[i]):
                vis[k] = True
        buttons.append(dict(label=name, method="update",
                            args=[{"visible": vis}, {"title": f"Return Distribution – {name}"}]))
    fig.update_layout(template=TEMPLATE, height=600,
                      title="Return Distributions (Rolling)",
                      xaxis_title="Annualised Return (%)",
                      updatemenus=[dict(active=0, buttons=buttons, x=1.0, xanchor="left", y=1.1)],
                      paper_bgcolor=C_CARD, plot_bgcolor=C_CARD)
    fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.4)
    return fig


def fig_metrics_table(metrics_list, rf=RF_RATE):
    rows = [
        (f"Ann. Return (CAGR)",         "ann_ret",      "{:.2%}"),
        (f"Ann. Volatility",             "ann_vol",      "{:.2%}"),
        (f"Sharpe  (rf={rf:.1%})",       "sharpe",       "{:.2f}"),
        (f"Sortino (rf={rf:.1%})",       "sortino",      "{:.2f}"),
        ("Max Drawdown",                 "max_dd",       "{:.2%}"),
        ("Calmar Ratio",                 "calmar",       "{:.2f}"),
        ("Beta  (vs benchmark)",         "beta",         "{:.2f}"),
        ("Jensen's Alpha",               "alpha",        "{:.2%}"),
        ("Annual Win Rate",              "win_rate",     "{:.0%}"),
        ("VaR 95% (daily)",              "var_95",       "{:.2%}"),
        ("CVaR 95% (daily)",             "cvar_95",      "{:.2%}"),
        ("Avg Recovery (days)",          "avg_recovery", "{:.0f}"),
        ("Max Recovery (days)",          "max_recovery", "{:.0f}"),
    ]
    headers = ["Metric"] + [m["label"] for m in metrics_list]
    cells_list = [[r[0] for r in rows]]
    for m in metrics_list:
        cells_list.append([r[2].format(m[r[1]]) for r in rows])
    fig = go.Figure(go.Table(
        header=dict(values=[f"<b>{h}</b>" for h in headers],
                    fill_color=C_BLUE, align="left",
                    font=dict(color="white", size=13)),
        cells=dict(values=cells_list, align="left",
                   fill_color=[[C_CARD, C_BORDER] * 20],
                   font=dict(color=C_TEXT, size=12), height=28)
    ))
    fig.update_layout(template=TEMPLATE, height=420,
                      margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor=C_CARD)
    return fig


def fig_monthly_heatmap(p_returns):
    # Version-agnostic monthly compounding (works on pandas <2.2 and >=2.2)
    s = p_returns.dropna()
    if s.empty:
        fig = go.Figure()
        fig.update_layout(template=TEMPLATE, height=320, title="Monthly Returns Heatmap (%) — No Data",
                          paper_bgcolor=C_CARD, plot_bgcolor=C_CARD)
        return fig
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.DatetimeIndex(s.index)
    monthly = s.groupby([s.index.year, s.index.month]).apply(lambda x: (1 + x).prod() - 1)
    df_m = pd.DataFrame({
        "Year":  monthly.index.get_level_values(0),
        "Month": monthly.index.get_level_values(1),
        "Ret":   monthly.values * 100.0,
    })
    pivot = df_m.pivot(index="Year", columns="Month", values="Ret")
    # Ensure all 12 months appear in chronological order
    pivot = pivot.reindex(columns=list(range(1, 13)))
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = month_names
    z_vals = pivot.values
    text_vals = np.where(np.isnan(z_vals), "", np.vectorize(lambda v: f"{v:.1f}%")(np.nan_to_num(z_vals)))
    h = max(320, len(pivot) * 26 + 90)
    fig = go.Figure(go.Heatmap(
        z=z_vals, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="RdYlGn", zmid=0,
        text=text_vals, texttemplate="%{text}",
        hovertemplate="Year %{y} – %{x}: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(template=TEMPLATE, height=h,
                      title="Monthly Returns Heatmap (%)",
                      xaxis_title="Month", yaxis_title="Year",
                      paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
                      yaxis=dict(autorange="reversed"))
    return fig


def fig_risk_attribution(returns, weights, days=252):
    """Risk Attribution: marginal contribution to portfolio volatility per asset.

    Uses the standard formula:
        MCTR_i = (Σw)_i / σ_p   where (Σw)_i is the i-th element of Cov·w
        %RCTR_i = w_i · MCTR_i / σ_p  (percentage contribution to variance)

    Shows two panels: % contribution to vol and standalone asset vol.
    """
    # weights may come in original ticker order — align to returns.columns
    tickers = list(returns.columns)
    n = len(tickers)
    w = np.array(weights)
    # If weights vector is longer than columns (dropped tickers), trim to match
    if len(w) > n:
        w = w[:n]
    elif len(w) < n:
        w = np.append(w, np.zeros(n - len(w)))
    # Re-normalise to sum=1 in case of rounding after trimming
    if w.sum() > 0:
        w = w / w.sum()
    cov = returns.cov().values * days
    port_var = float(np.dot(w.T, np.dot(cov, w)))
    port_vol = np.sqrt(port_var) if port_var > 0 else 1e-8

    # Component contribution to variance (correct formula):
    # MCTR_i  = (Cov · w)_i / σ_p          (marginal contribution to vol)
    # CCTR_i  = w_i · MCTR_i               (component contribution to vol, sums to σ_p)
    # %CCTR_i = CCTR_i / σ_p * 100         (% of total vol, sums to 100%)
    # Negative values are CORRECT: negative covariance assets (e.g. bonds/gold)
    # actually reduce total portfolio volatility — they are diversifiers.
    mctr = np.dot(cov, w) / port_vol        # marginal contrib to vol
    cctr = w * mctr                          # component contrib (sums to port_vol)
    pct_contrib = cctr / port_vol * 100      # % of total vol (sums to 100%)

    asset_vols = np.sqrt(np.diag(cov)) * 100
    tickers = list(returns.columns)

    # Green = positive contributor, Cyan = diversifier (negative contrib = vol reducer)
    colors_contrib = [C_BLUE if v >= 0 else C_CYAN for v in pct_contrib]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "% Contribution to Portfolio Vol (negative = diversifier)",
            "Standalone Asset Volatility (ann.)"
        ),
        column_widths=[0.5, 0.5],
    )
    fig.add_trace(go.Bar(
        x=tickers, y=pct_contrib,
        marker_color=colors_contrib,
        text=[f"{v:.1f}%" for v in pct_contrib],
        textposition="outside",
        name="Vol Contribution %",
    ), row=1, col=1)
    fig.add_hline(y=0, line_color="white", opacity=0.3, row=1, col=1)
    fig.add_trace(go.Bar(
        x=tickers, y=asset_vols,
        marker_color=C_BLUE,
        text=[f"{v:.1f}%" for v in asset_vols],
        textposition="outside",
        name="Standalone Vol %",
    ), row=1, col=2)

    sum_check = pct_contrib.sum()
    fig.update_layout(
        template=TEMPLATE, height=500,
        title=(f"Risk Attribution  ·  Portfolio σ={port_vol*100:.2f}%  ·  "
               f"Sum of contributions={sum_check:.1f}%  ·  "
               f"Cyan bars = vol reducers (negative covariance)"),
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        showlegend=False,
        margin=dict(t=90, b=60),
    )
    fig.update_yaxes(ticksuffix="%")
    return fig


def fig_performance_attribution(returns, weights, bench_ret, days=252):
    """Performance Attribution (Brinson-style simplified).

    For each asset: contribution to portfolio return = w_i * (r_i_ann - r_bench_ann)
    Shows active return contribution (alpha source) per asset.
    """
    tickers = list(returns.columns)
    n = len(tickers)
    w = np.array(weights)
    if len(w) > n:
        w = w[:n]
    elif len(w) < n:
        w = np.append(w, np.zeros(n - len(w)))
    if w.sum() > 0:
        w = w / w.sum()
    bench_ann = float((1 + bench_ret).prod() ** (days / len(bench_ret)) - 1)

    asset_ann = []
    for t in tickers:
        r = returns[t].dropna()
        ann = float((1 + r).prod() ** (days / len(r)) - 1) if len(r) > 0 else 0
        asset_ann.append(ann)

    asset_ann = np.array(asset_ann)
    active_ret = asset_ann - bench_ann               # excess return per asset
    contribution = w * active_ret * 100              # weighted active return %
    total_active = float(np.dot(w, active_ret)) * 100

    colors = [C_GREEN if v >= 0 else C_RED for v in contribution]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Weighted Active Return Contribution (%)", "Asset Ann. Return vs Benchmark"),
        column_widths=[0.5, 0.5],
    )
    fig.add_trace(go.Bar(
        x=tickers, y=contribution,
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in contribution],
        textposition="outside",
        name="Active Contribution",
    ), row=1, col=1)
    fig.add_hline(y=0, line_color="white", opacity=0.3, row=1, col=1)

    # Panel 2: asset ann return vs bench line
    fig.add_trace(go.Bar(
        x=tickers, y=asset_ann * 100,
        marker_color=[C_GREEN if v >= bench_ann else C_RED for v in asset_ann],
        text=[f"{v*100:.1f}%" for v in asset_ann],
        textposition="outside",
        name="Asset Ann. Return",
    ), row=1, col=2)
    fig.add_hline(
        y=bench_ann * 100, line_color=C_ORANGE, line_dash="dot",
        annotation_text=f"Bench {bench_ann*100:.1f}%",
        annotation_position="right", row=1, col=2,
    )

    fig.update_layout(
        template=TEMPLATE, height=480,
        title=f"Performance Attribution  ·  Total Active Return: {total_active:+.2f}%  ·  Benchmark: {bench_ann*100:.2f}%",
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        showlegend=False,
        margin=dict(t=80, b=60),
    )
    fig.update_yaxes(ticksuffix="%")
    return fig


def fig_rolling_volatility(p_ret, bench_ret, days=252):
    """Rolling volatility chart: 21-day (short) and 252-day (long), annualised."""
    short_window, long_window = 21, 252
    ann_factor = np.sqrt(252)

    vol_short_p = p_ret.rolling(short_window).std() * ann_factor
    vol_long_p  = p_ret.rolling(long_window).std() * ann_factor

    bench_aligned = bench_ret.reindex(p_ret.index)
    vol_short_b = bench_aligned.rolling(short_window).std() * ann_factor
    vol_long_b  = bench_aligned.rolling(long_window).std() * ann_factor

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vol_short_p.index, y=vol_short_p * 100,
                             name="Portfolio 21d", line=dict(color=C_BLUE, width=1.5)))
    fig.add_trace(go.Scatter(x=vol_long_p.index, y=vol_long_p * 100,
                             name="Portfolio 252d", line=dict(color=C_BLUE, width=2.5)))
    fig.add_trace(go.Scatter(x=vol_short_b.index, y=vol_short_b * 100,
                             name="Benchmark 21d",
                             line=dict(color=C_ORANGE, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=vol_long_b.index, y=vol_long_b * 100,
                             name="Benchmark 252d",
                             line=dict(color=C_ORANGE, width=2.5, dash="dash")))
    fig.update_layout(
        template=TEMPLATE, height=460,
        title="Rolling Volatility  ·  21d (short-term) vs 252d (long-term)  ·  Annualised",
        yaxis_title="Annualised Volatility (%)", yaxis=dict(ticksuffix="%"),
        hovermode="x unified", paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        legend=dict(orientation="h", y=-0.15, x=0),
    )
    return fig


def fig_suggestion_table(sug_data):
    """Build a Plotly table + badge display from suggestion engine results."""
    df = sug_data["results"]
    if df.empty:
        return None

    # Colour-code cells (matching actual badge strings)
    badge_colors = []
    for _, row in df.iterrows():
        badge = row.get("Badge", "")
        if "Best IR" in badge or "Best Sharpe" in badge:
            badge_colors.append("rgba(10,132,255,0.35)")
        elif "Min Corr" in badge:
            badge_colors.append("rgba(48,209,88,0.30)")
        elif "Min Vol" in badge:
            badge_colors.append("rgba(255,214,10,0.28)")
        else:
            badge_colors.append(C_CARD)

    sectors_col = df["Sector"].tolist() if "Sector" in df.columns else ["—"] * len(df)

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Rank</b>", "<b>Ticker</b>", "<b>Sector</b>", "<b>New Vol</b>",
                    "<b>New Ret</b>", "<b>Info Ratio</b>", "<b>New Sharpe</b>",
                    "<b>Corr w/ Port</b>", "<b>Vol Chg</b>",
                    "<b>Sharpe Chg</b>", "<b>Badge</b>"],
            fill_color=C_BLUE, align="left",
            font=dict(color="white", size=11),
        ),
        cells=dict(
            values=[
                list(range(1, len(df) + 1)),
                df["Ticker"].tolist(),
                sectors_col,
                [f"{v:.2%}" for v in df["New_Vol"]],
                [f"{v:.2%}" for v in df["New_Ret"]],
                [f"{v:.2f}" for v in (df["IR"] if "IR" in df.columns else df["New_Sharpe"])],
                [f"{v:.2f}" for v in df["New_Sharpe"]],
                [f"{v:.2f}" for v in df["Avg_Corr"]],
                [f"{v:+.2%}" for v in df["Vol_Change"]],
                [f"{v:+.2f}" for v in df["Sharpe_Change"]],
                df["Badge"].tolist(),
            ],
            align="left",
            fill_color=[
                [C_CARD] * len(df),
                [C_CARD] * len(df),
                [C_CARD] * len(df),
                [C_CARD] * len(df),
                [C_CARD] * len(df),
                [C_CARD] * len(df),
                [C_CARD] * len(df),
                [C_CARD] * len(df),
                ["rgba(48,209,88,0.25)" if v < 0 else "rgba(255,69,58,0.22)" for v in df["Vol_Change"]],
                ["rgba(48,209,88,0.25)" if v > 0 else "rgba(255,69,58,0.22)" for v in df["Sharpe_Change"]],
                badge_colors,
            ],
            font=dict(color=C_TEXT, size=11),
            height=28,
        ),
    ))
    fig.update_layout(
        template=TEMPLATE, height=max(360, len(df) * 30 + 80),
        margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor=C_CARD,
    )
    return fig


def fig_monte_carlo(p_returns, returns_df, backtest_weights, days,
                    n_paths=1000, horizon_years=4, rf=RF_RATE):
    """
    Forward-looking Monte Carlo using correlated asset paths.
    - Fits mu/sigma from historical daily returns of each asset.
    - Applies Cholesky decomposition of the correlation matrix for realism.
    - Produces fan chart with confidence bands + final distribution.
    """
    horizon_days = int(horizon_years * days)
    weights = np.array(backtest_weights)
    n_assets = returns_df.shape[1]

    # ── Fit parameters from history ──
    mu_daily  = returns_df.mean().values          # daily mean per asset
    std_daily = returns_df.std().values           # daily std per asset
    corr      = returns_df.corr().values

    # Ensure positive-definite correlation matrix
    corr_pd = corr.copy()
    corr_pd += 1e-8 * np.eye(n_assets)
    try:
        L = np.linalg.cholesky(corr_pd)
    except np.linalg.LinAlgError:
        L = np.eye(n_assets)

    # ── Simulate ──
    np.random.seed(7)
    z          = np.random.normal(size=(n_paths, horizon_days, n_assets))
    z_corr     = np.einsum("pti,ij->ptj", z, L.T)
    asset_rets = z_corr * std_daily + mu_daily         # (paths, days, assets)
    port_rets  = asset_rets @ weights                  # (paths, days)

    # Cumulative growth from 1.0
    cum_growth = np.cumprod(1 + port_rets, axis=1)    # (paths, days)

    # ── Percentile bands ──
    p5   = np.percentile(cum_growth, 5,  axis=0)
    p10  = np.percentile(cum_growth, 10, axis=0)
    p25  = np.percentile(cum_growth, 25, axis=0)
    p50  = np.percentile(cum_growth, 50, axis=0)
    p75  = np.percentile(cum_growth, 75, axis=0)
    p90  = np.percentile(cum_growth, 90, axis=0)
    p95  = np.percentile(cum_growth, 95, axis=0)
    mean = cum_growth.mean(axis=0)

    # Historical actual path (base = 1.0)
    hist_cum = (1 + p_returns).cumprod()
    hist_cum = hist_cum / hist_cum.iloc[0]

    t = np.arange(1, horizon_days + 1)

    # ── Final distribution metrics ──
    final = cum_growth[:, -1]
    var_95_val  = np.percentile(final, 5)
    cvar_95_val = final[final <= var_95_val].mean()
    ann_ret_mc  = float(np.median(final)) ** (days / horizon_days) - 1

    # ── Build figure ──
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=(
            f"Monte Carlo Fan Chart  ·  {n_paths:,} paths  ·  {horizon_years}Y horizon",
            f"Final Value Distribution  ·  {horizon_years}Y"
        )
    )

    # Shaded confidence bands (fan)
    for lo, hi, alpha_val, label in [
        (p5,  p95, "rgba(65,105,225,0.10)", "5–95%"),
        (p10, p90, "rgba(65,105,225,0.15)", "10–90%"),
        (p25, p75, "rgba(65,105,225,0.22)", "25–75%"),
    ]:
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself", fillcolor=alpha_val,
            line=dict(width=0), showlegend=True, name=label,
            hoverinfo="skip"
        ), row=1, col=1)

    # Median and mean lines
    fig.add_trace(go.Scatter(x=t, y=p50, name="Median",
                             line=dict(color=C_BLUE, width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=mean, name="Mean",
                             line=dict(color=C_GREEN, width=1.5, dash="dash")), row=1, col=1)

    # VaR floor line
    fig.add_trace(go.Scatter(x=t, y=p5, name="5th pct (VaR floor)",
                             line=dict(color=C_RED, width=1.5, dash="dot")), row=1, col=1)

    # Sample paths (50 random)
    idx_sample = np.random.choice(n_paths, size=min(60, n_paths), replace=False)
    for i in idx_sample:
        fig.add_trace(go.Scatter(
            x=t, y=cum_growth[i],
            line=dict(color="rgba(150,180,255,0.12)", width=1),
            showlegend=False, hoverinfo="skip"
        ), row=1, col=1)

    # Reference line at 1.0 (no gain/loss)
    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.25)",
                  annotation_text="Break-even", row=1, col=1)

    # ── Right panel: final value histogram ──
    fig.add_trace(go.Histogram(
        x=final, nbinsx=60,
        marker_color=C_BLUE, opacity=0.7,
        name="Final value dist.", showlegend=False
    ), row=1, col=2)

    # VaR and CVaR verticals
    fig.add_vline(x=var_95_val,  line_dash="dash",  line_color=C_RED,
                  annotation_text=f"VaR 95%<br>{var_95_val:.2f}x",
                  annotation_font=dict(color=C_RED, size=11),
                  annotation_position="top left", row=1, col=2)
    fig.add_vline(x=cvar_95_val, line_dash="dot", line_color=C_ORANGE,
                  annotation_text=f"CVaR 95%<br>{cvar_95_val:.2f}x",
                  annotation_font=dict(color=C_ORANGE, size=11),
                  annotation_position="top left", row=1, col=2)
    fig.add_vline(x=1.0, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  row=1, col=2)

    fig.update_layout(
        template=TEMPLATE, height=540,
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12, x=0),
        margin=dict(t=70, b=60),
        annotations=[
            dict(
                text=(f"Median CAGR: <b>{ann_ret_mc:.1%}</b>  ·  "
                      f"5% VaR: <b>{var_95_val:.2f}x</b>  ·  "
                      f"CVaR: <b>{cvar_95_val:.2f}x</b>  ·  "
                      f"P(loss): <b>{(final < 1).mean():.1%}</b>"),
                xref="paper", yref="paper", x=0.0, y=-0.06,
                showarrow=False, font=dict(size=12, color=C_MUTED),
                align="left"
            )
        ]
    )
    fig.update_xaxes(title_text="Trading days forward", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio value (×initial)", row=1, col=1)
    fig.update_xaxes(title_text="Portfolio value (×initial)", row=1, col=2)
    fig.update_yaxes(title_text="Paths", row=1, col=2)
    return fig


