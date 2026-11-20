"""
FastbreaK API Server
Serves dashboard.html and provides REST API for portfolio backtesting.
Run:  python server.py   →  http://127.0.0.1:5000
"""
import os, io, json, time, traceback
from collections import OrderedDict
import numpy as np, pandas as pd
from datetime import datetime
from flask import Flask, send_file, request, jsonify
import plotly.io as pio, plotly.graph_objects as go

# Import calculation + chart functions from existing modules
import portfolio_last as P
import portfolio_helpers as H
import quant_advanced as Q

app = Flask(__name__)

# ── Multi-User Session Store (LRU + TTL) & Download Cache ───────
_SESSIONS = OrderedDict()
_MAX_SESSIONS = 25
_SESSION_TTL = 7200  # 2 hours

_DL_CACHE = OrderedDict()
_DL_TTL = 1200       # 20 minutes
_MAX_DL_CACHE = 30
_NAME_CACHE = {}     # ticker -> full company/ETF/commodity name

def _fetch_one_name(ticker):
    if ticker in _NAME_CACHE:
        return ticker, _NAME_CACHE[ticker]
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        name = info.get('longName') or info.get('shortName') or ticker
    except Exception:
        name = ticker
    _NAME_CACHE[ticker] = name
    return ticker, name

def _get_ticker_names(tickers):
    from concurrent.futures import ThreadPoolExecutor
    missing = [t for t in tickers if t not in _NAME_CACHE]
    if missing:
        with ThreadPoolExecutor(max_workers=min(8, len(missing))) as ex:
            for t, nm in ex.map(_fetch_one_name, missing):
                _NAME_CACHE[t] = nm
    return {t: _NAME_CACHE.get(t, t) for t in tickers}

def _get_sid():
    return (request.headers.get('X-Session-ID')
            or request.args.get('sid')
            or 'default')

def _get_state():
    sid = _get_sid()
    now = time.time()
    entry = _SESSIONS.get(sid)
    if not entry:
        return None
    if now - entry['ts'] > _SESSION_TTL:
        _SESSIONS.pop(sid, None)
        return None
    _SESSIONS.move_to_end(sid)
    entry['ts'] = now
    return entry['data']

def _set_state(data):
    sid = _get_sid()
    _SESSIONS[sid] = {'ts': time.time(), 'data': data}
    _SESSIONS.move_to_end(sid)
    while len(_SESSIONS) > _MAX_SESSIONS:
        _SESSIONS.popitem(last=False)

def _cached_download(tickers, benchmark, start, end):
    key = (tuple(sorted(tickers)), benchmark, start, end)
    now = time.time()
    hit = _DL_CACHE.get(key)
    if hit and (now - hit['ts'] < _DL_TTL):
        _DL_CACHE.move_to_end(key)
        ret, bm, tstart = hit['val']
        cols = [c for c in tickers if c in ret.columns]
        return ret[cols].copy(), bm.copy(), dict(tstart)
    for attempt in range(2):
        try:
            res = P.download_data(tickers, benchmark, start, end)
            break
        except Exception:
            if attempt == 0:
                time.sleep(1)
            else:
                raise
    _DL_CACHE[key] = {'ts': now, 'val': res}
    _DL_CACHE.move_to_end(key)
    while len(_DL_CACHE) > _MAX_DL_CACHE:
        _DL_CACHE.popitem(last=False)
    ret, bm, tstart = res
    return ret.copy(), bm.copy(), dict(tstart)

# ── Quant Score (0-100) ─────────────────────────────────────────
def compute_quant_score(m, bm):
    """Composite portfolio quality score — normalised, rf-independent.

    Factors & weights (total = 100):
      IR = Ann.Ret / Vol                      weight 20  (rf-free alpha efficiency)
      Real Return Factor (Ret-2.5%) / Ret     weight 15  (inflation-adjusted quality)
      Risk Efficiency = |MaxDD| / Calmar      weight 10  (lower = better)
      Vol Efficiency vs Benchmark             weight  5
      CVaR 95% daily                          weight 10
      Avg Recovery Time                       weight 20
      Max Recovery Time                       weight 10
    Remaining 10 pts from win rate kept for consistency.
    """
    s = 0
    ann_ret    = m.get('ann_ret', 0)
    ann_vol    = m.get('ann_vol', 1) or 1
    bench_vol  = bm.get('ann_vol', 1) or 1
    INFLATION  = 0.025

    # 1. IR = Ann.Ret / Vol  (20pts) — ref: IR≥1.0 = full
    ir = ann_ret / ann_vol if ann_vol > 0 else 0
    s += min(max(ir / 1.0, 0), 1) * 20

    # 2. Real Return Factor = (Ret - Inflation) / |Ret|  (15pts)
    #    Positive only when ret > inflation and ret > 0
    if ann_ret > 0:
        rrf = (ann_ret - INFLATION) / ann_ret
        s += min(max(rrf, 0), 1) * 15
    # else 0

    # 3. Risk Efficiency = |MaxDD| / Calmar  (10pts, lower is better)
    dd     = abs(m.get('max_dd', 0)) or 0.01
    calmar = m.get('calmar', 0)
    # Calmar = ann_ret / |MaxDD|, so |MaxDD|/Calmar = |MaxDD|^2/ann_ret
    # Normalise: ratio < 0.10 = full, ratio > 0.50 = 0
    if calmar > 0:
        risk_eff = dd / calmar   # lower = better: high calmar vs low dd
        s += max(1 - risk_eff / 0.5, 0) * 10
    # if calmar <= 0: 0pts

    # 4. Vol Efficiency vs Benchmark  (5pts)
    vol_ratio = ann_vol / bench_vol if bench_vol > 0 else 1
    s += max(1 - vol_ratio, 0) * 5

    # 5. CVaR 95% daily  (10pts, ref: CVaR < 2% daily = full)
    cvar = abs(m.get('cvar_95', 0))
    s += max(1 - cvar / 0.03, 0) * 10

    # 6. Avg Recovery Time  (20pts, ref: <200 days avg = full)
    avg_rec = m.get('avg_recovery', 0)
    s += max(1 - avg_rec / 400, 0) * 20

    # 7. Max Recovery Time  (10pts, ref: <500 days max = full)
    max_rec = m.get('max_recovery', 0)
    s += max(1 - max_rec / 800, 0) * 10

    # 8. Win rate bonus  (10pts, kept for annual consistency)
    s += m.get('win_rate', 0) * 10

    return round(min(max(s, 0), 100), 1)

def compute_gmv_weights(returns):
    """Global Minimum Variance portfolio weights (analytical)."""
    cov = returns.cov().values
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    ones = np.ones(len(cov))
    w = cov_inv @ ones / (ones @ cov_inv @ ones)
    return np.maximum(w, 0) / np.maximum(w, 0).sum()  # long-only, normalised

# ── JSON helpers ────────────────────────────────────────────────
class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, float)): return float(o)
        if isinstance(o, (np.integer, int)): return int(o)
        if isinstance(o, pd.Timestamp): return o.isoformat()
        return super().default(o)

# ── Light Minimal Plotly Theme Transformer ──────────────────────
_DARK_BG_SET = {'#13131c', '#111118', '#0a0a0f', '#1e1e2e', 'rgb(19, 19, 28)'}

def _map_cell_color(c, idx=0):
    if not isinstance(c, str):
        return c
    cl = c.strip().lower()
    if cl in ('#13131c', '#111118', '#0a0a0f'):
        return '#ffffff'
    if cl == '#1e1e2e':
        return '#f8fafc'
    return c

def _apply_light_theme(fig_dict):
    """Harmonise every Plotly figure to the Light Minimal Institutional palette."""
    layout = fig_dict.setdefault('layout', {})
    # Strip dark template
    layout.pop('template', None)
    layout['paper_bgcolor'] = '#ffffff'
    layout['plot_bgcolor'] = '#ffffff'
    layout['font'] = {'family': 'JetBrains Mono, monospace', 'color': '#0f172a', 'size': 11}
    if 'title' in layout and isinstance(layout['title'], dict):
        layout['title']['font'] = {'family': 'Plus Jakarta Sans, sans-serif', 'size': 14, 'color': '#0f172a'}
    elif 'title' in layout and isinstance(layout['title'], str):
        layout['title'] = {'text': layout['title'], 'font': {'family': 'Plus Jakarta Sans, sans-serif', 'size': 14, 'color': '#0f172a'}}

    # Crisp light gridlines on all axes
    for k, v in list(layout.items()):
        if (k.startswith('xaxis') or k.startswith('yaxis')) and isinstance(v, dict):
            v['gridcolor'] = '#e2e8f0'
            v['zerolinecolor'] = '#cbd5e1'
            v['linecolor'] = '#e2e8f0'
            v.setdefault('tickfont', {})['color'] = '#475569'
            if isinstance(v.get('title'), dict):
                v['title'].setdefault('font', {})['color'] = '#334155'

    # Legend styling
    if isinstance(layout.get('legend'), dict):
        layout['legend']['bgcolor'] = 'rgba(255,255,255,0.92)'
        layout['legend']['bordercolor'] = '#e2e8f0'
        layout['legend']['font'] = {'family': 'JetBrains Mono, monospace', 'size': 10, 'color': '#334155'}

    # Reference lines (hlines/vlines): swap white lines to subtle slate
    for sh in layout.get('shapes', []) or []:
        if isinstance(sh, dict) and isinstance(sh.get('line'), dict):
            lc = str(sh['line'].get('color', '')).lower()
            if 'white' in lc or '255,255,255' in lc or '255, 255, 255' in lc:
                sh['line']['color'] = 'rgba(15,23,42,0.22)'

    # Annotations: swap light-on-dark text to crisp ink
    for ann in layout.get('annotations', []) or []:
        if isinstance(ann, dict):
            fn = ann.get('font')
            if isinstance(fn, dict):
                fc = str(fn.get('color', '')).lower()
                if fc in ('#e8e8f0', 'white', '#ffffff', '#64d2ff'):
                    fn['color'] = '#0f172a' if fc != '#64d2ff' else '#0284c7'
                elif fc == '#555570':
                    fn['color'] = '#475569'
            else:
                ann['font'] = {'color': '#0f172a', 'size': 11}

    # Trace-specific light adjustments
    for tr in fig_dict.get('data', []) or []:
        ttype = tr.get('type', '')
        if ttype == 'table':
            hdr = tr.get('header', {})
            if isinstance(hdr, dict):
                hdr['fill'] = {'color': '#0f172a'}
                hdr['line'] = {'color': '#1e293b', 'width': 1}
                hdr['font'] = {'color': '#ffffff', 'size': 11, 'family': 'Plus Jakarta Sans, sans-serif'}
            cells = tr.get('cells', {})
            if isinstance(cells, dict):
                cells['line'] = {'color': '#e2e8f0', 'width': 1}
                cells['font'] = {'color': '#0f172a', 'size': 11, 'family': 'JetBrains Mono, monospace'}
                fc = cells.get('fill', {}).get('color')
                if isinstance(fc, list):
                    cells['fill']['color'] = [
                        [_map_cell_color(x, i) for i, x in enumerate(col)] if isinstance(col, list)
                        else _map_cell_color(col)
                        for col in fc
                    ]
                elif isinstance(fc, str):
                    cells['fill']['color'] = _map_cell_color(fc)
        elif ttype == 'pie':
            mk = tr.setdefault('marker', {})
            mk['line'] = {'color': '#ffffff', 'width': 2}
        elif ttype == 'scatter':
            ln = tr.get('line', {})
            if isinstance(ln, dict):
                lc = str(ln.get('color', '')).lower()
                if 'white' in lc or '255,255,255' in lc:
                    ln['color'] = 'rgba(15,23,42,0.45)'
    return fig_dict

def fig_json(fig):
    raw = json.loads(pio.to_json(fig))
    return _apply_light_theme(raw)

def jresp(obj, status=200):
    return json.dumps(obj, cls=NpEncoder), status, {'Content-Type':'application/json'}

# ── Routes ──────────────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify(status='ok', sessions=len(_SESSIONS), cached_downloads=len(_DL_CACHE))

@app.route('/')
def index():
    return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard.html'))

@app.route('/F.png')
def logo_png():
    return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'F.png'), mimetype='image/png')

@app.route('/api/defaults')
def api_defaults():
    return jsonify(tickers=[t[0] for t in H.DEFAULT_TICKERS],
                   currencies=[t[1] for t in H.DEFAULT_TICKERS],
                   max_tickers=H.MAX_TICKERS,
                   n1_universes=H.get_n1_universe_options())

# ── Backtest endpoint ───────────────────────────────────────────
@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    try:
        d = request.json
        # Parse inputs
        td = []
        for i, t in enumerate(d.get('tickers', [])):
            t = (t or '').strip().upper()
            c = d['currencies'][i] if i < len(d.get('currencies',[])) else 'USD'
            w = d['weights'][i] if i < len(d.get('weights',[])) else None
            if t: td.append((t, c, w))
        if not td:
            return jsonify(status='error', message='No tickers'), 400

        tickers = [x[0] for x in td]
        tcurr = {x[0]: x[1] for x in td}
        raw_w = [x[2] for x in td]
        base = d.get('base_currency','USD')
        cs = H.CURRENCY_SYMBOLS.get(base, '$')
        rebal = [int(m.strip()) for m in d.get('rebal_months','1,5,9').split(',') if m.strip()]
        n_port = int(d.get('n_port', 8000))
        rf = float(d.get('rf', 4.0)) / 100.0
        wmode = d.get('weight_mode', 'markowitz')
        if wmode == 'walkforward':
            wmode = 'markowitz'
        est_mode = d.get('estimation_mode', 'oos3')  # 'oos3', 'oos5', 'insample'
        start = d.get('start_date', '1988-10-01')
        end = datetime.now().strftime('%Y-%m-%d')

        # Download with in-memory TTL cache + retry
        returns_full, bench_full, ticker_start = _cached_download(
            tickers, d.get('benchmark','^GSPC'), start, end
        )
        tickers = list(returns_full.columns)
        benchmark = d.get('benchmark','^GSPC')

        # FX conversion
        needs_fx = base != 'USD' or any(tcurr.get(t,'USD') != 'USD' for t in tickers)
        if needs_fx:
            fx = H.download_fx_rate(start, end)
            returns_full, bench_full = H.convert_all_returns(returns_full, bench_full, tcurr, base, fx)
            returns_full = returns_full.dropna(how='all'); bench_full = bench_full.dropna()
            ci = returns_full.index.intersection(bench_full.index)
            returns_full, bench_full = returns_full.loc[ci], bench_full.loc[ci]
        if returns_full.empty:
            return jsonify(status='error', message='No data after processing'), 400

        days = 365 if any('-USD' in x for x in tickers) else 252
        nu_val = d.get('elliptical_nu', 5.0)
        cw = H.parse_custom_weights(raw_w, len(tickers))

        # Efficient frontier Monte Carlo cloud (on full history for reference)
        _, mc_res, _, msi = P.markowitz_optimize(returns_full, days, n_port, rf)

        is_oos = est_mode in ('oos3', 'oos5')
        win_y = 5 if est_mode == 'oos5' else 3

        if is_oos:
            # ── GLOBAL ZERO-LOOK-AHEAD OUT-OF-SAMPLE WALK-FORWARD ──
            oos = Q.run_rolling_oos_schemes(
                returns_full, bench_full, days, rf, rebal,
                compute_metrics_fn=P.compute_metrics, nu=nu_val, window_years=win_y,
            )
            bench_ret = oos['bench_oos']
            returns = returns_full.loc[bench_ret.index]

            # Actionable weights = latest trailing rolling window (ready for today)
            opt_w = oos['latest_weights']['markowitz']
            gmv_w = oos['latest_weights']['gmv']
            hrp_w = oos['latest_weights']['hrp']
            lw_w  = oos['latest_weights']['ledoit_wolf']
            mp_w  = oos['latest_weights']['marchenko_pastur']
            ell_w = oos['latest_weights']['elliptical']

            pr_mko = oos['series']['markowitz']
            pr_gmv = oos['series']['gmv']
            pr_hrp = oos['series']['hrp']
            pr_lw  = oos['series']['ledoit_wolf']
            pr_mp  = oos['series']['marchenko_pastur']
            pr_ell = oos['series']['elliptical']

            m_mko = oos['metrics']['markowitz']
            m_gmv = oos['metrics']['gmv']
            m_hrp = oos['metrics']['hrp']
            m_lw  = oos['metrics']['ledoit_wolf']
            m_mp  = oos['metrics']['marchenko_pastur']
            m_ell = oos['metrics']['elliptical']
            mb    = P.compute_metrics(bench_ret, bench_ret, days, benchmark, rf)

            pr_cust, m_cust, wh_cust = None, None, None
            if cw is not None:
                wh_cust, pr_cust = P.run_simulation(returns, cw, rebal)
                m_cust = P.compute_metrics(pr_cust, bench_ret, days, 'Custom Weights', rf)

            mode_map = {
                'markowitz':        (opt_w, f'Sample MLE (OOS {win_y}Y)',        m_mko, pr_mko, oos['weights_hist']['markowitz']),
                'gmv':              (gmv_w, f'Global Min Variance (OOS {win_y}Y)', m_gmv, pr_gmv, oos['weights_hist']['gmv']),
                'hrp':              (hrp_w, f'HRP (OOS {win_y}Y)',               m_hrp, pr_hrp, oos['weights_hist']['hrp']),
                'ledoit_wolf':      (lw_w,  f'Ledoit-Wolf (OOS {win_y}Y)',       m_lw,  pr_lw,  oos['weights_hist']['ledoit_wolf']),
                'marchenko_pastur': (mp_w,  f'Marchenko-Pastur (OOS {win_y}Y)',  m_mp,  pr_mp,  oos['weights_hist']['marchenko_pastur']),
                'elliptical':       (ell_w, f'Elliptical MLE (OOS {win_y}Y)',    m_ell, pr_ell, oos['weights_hist']['elliptical']),
            }
            active_key = wmode if wmode in mode_map else 'markowitz'
            if wmode == 'custom' and cw is not None:
                bw, wl, mp, pr, wh = cw, 'Custom Weights', m_cust, pr_cust, wh_cust
            else:
                bw, wl, mp, pr, wh = mode_map[active_key]

            roll_df = oos['rebal_df'][active_key]
            roll_sh = oos['rebal_sharpes'][active_key]
            est_mode_label = f'Out-of-Sample Walk-Forward ({win_y}Y Rolling · Zero Look-Ahead)'
            weights_window_range = oos['latest_window_range']
            mps = [
                ('Sample MLE [OOS]',          pr_mko),
                ('Global Min Variance [OOS]', pr_gmv),
                ('HRP [OOS]',                 pr_hrp),
                ('Ledoit-Wolf [OOS]',         pr_lw),
                ('Marchenko-Pastur [OOS]',    pr_mp),
                ('Elliptical MLE [OOS]',      pr_ell),
            ]
            if pr_cust is not None:
                mps.append(('Custom Weights', pr_cust))
            est_diag = {}
        else:
            # ── IN-SAMPLE FULL HISTORY (STATIC REFERENCE) ──
            returns, bench_ret = returns_full, bench_full
            mu_mle_f, cov_mle_f, _ = Q.estimate_sample_mle(returns, days)
            opt_w = Q.optimize_max_sharpe(mu_mle_f, cov_mle_f, rf)
            adv = Q.compute_all_schemes(returns, days=days, rf=rf, nu=nu_val, objective='max_sharpe')
            gmv_w = adv['weights']['gmv']
            hrp_w = adv['weights']['hrp']
            lw_w  = adv['weights']['ledoit_wolf']
            mp_w  = adv['weights']['marchenko_pastur']
            ell_w = adv['weights']['elliptical']
            est_diag = adv['diagnostics']

            wh_mko, pr_mko = P.run_simulation(returns, opt_w, rebal)
            m_mko = P.compute_metrics(pr_mko, bench_ret, days, 'Sample MLE', rf)
            wh_gmv, pr_gmv = P.run_simulation(returns, gmv_w, rebal)
            m_gmv = P.compute_metrics(pr_gmv, bench_ret, days, 'Global Min Variance', rf)
            wh_hrp, pr_hrp = P.run_simulation(returns, hrp_w, rebal)
            m_hrp = P.compute_metrics(pr_hrp, bench_ret, days, 'HRP', rf)
            wh_lw,  pr_lw  = P.run_simulation(returns, lw_w, rebal)
            m_lw  = P.compute_metrics(pr_lw, bench_ret, days, 'Ledoit-Wolf', rf)
            wh_mp,  pr_mp  = P.run_simulation(returns, mp_w, rebal)
            m_mp  = P.compute_metrics(pr_mp, bench_ret, days, 'Marchenko-Pastur', rf)
            wh_ell, pr_ell = P.run_simulation(returns, ell_w, rebal)
            m_ell = P.compute_metrics(pr_ell, bench_ret, days, 'Elliptical MLE', rf)
            mb    = P.compute_metrics(bench_ret, bench_ret, days, benchmark, rf)

            pr_cust, m_cust, wh_cust = None, None, None
            if cw is not None:
                wh_cust, pr_cust = P.run_simulation(returns, cw, rebal)
                m_cust = P.compute_metrics(pr_cust, bench_ret, days, 'Custom Weights', rf)

            mode_map = {
                'markowitz':        (opt_w, 'Sample MLE (In-Sample)',          m_mko, pr_mko, wh_mko),
                'gmv':              (gmv_w, 'Global Min Variance (In-Sample)', m_gmv, pr_gmv, wh_gmv),
                'hrp':              (hrp_w, 'HRP (In-Sample)',                 m_hrp, pr_hrp, wh_hrp),
                'ledoit_wolf':      (lw_w,  'Ledoit-Wolf (In-Sample)',         m_lw,  pr_lw,  wh_lw),
                'marchenko_pastur': (mp_w,  'Marchenko-Pastur (In-Sample)',    m_mp,  pr_mp,  wh_mp),
                'elliptical':       (ell_w, 'Elliptical MLE (In-Sample)',      m_ell, pr_ell, wh_ell),
            }
            active_key = wmode if wmode in mode_map else 'markowitz'
            if wmode == 'custom' and cw is not None:
                bw, wl, mp, pr, wh = cw, 'Custom Weights', m_cust, pr_cust, wh_cust
            else:
                bw, wl, mp, pr, wh = mode_map[active_key]

            roll_df, roll_sh = None, []
            est_mode_label = 'In-Sample Full History (Static Reference)'
            weights_window_range = f"{returns.index[0].strftime('%Y-%m-%d')} → {returns.index[-1].strftime('%Y-%m-%d')}"
            mps = [
                ('Sample MLE',          pr_mko),
                ('Global Min Variance', pr_gmv),
                ('HRP',                 pr_hrp),
                ('Ledoit-Wolf',         pr_lw),
                ('Marchenko-Pastur',    pr_mp),
                ('Elliptical MLE',      pr_ell),
            ]
            if pr_cust is not None:
                mps.append(('Custom Weights', pr_cust))

        # Quant Score
        qs = compute_quant_score(mp, mb)

        # Recent perf
        tr = float(pr.iloc[-1]) if len(pr)>0 else 0
        wr = float((1+pr.iloc[-5:]).prod()-1) if len(pr)>=5 else 0
        mm = pr.index >= pr.index[-1].replace(day=1)
        mr = float((1+pr[mm]).prod()-1) if mm.any() else 0
        # Last month (previous calendar month)
        prev_m_end = pr.index[-1].replace(day=1) - pd.Timedelta(days=1)
        prev_m_start = prev_m_end.replace(day=1)
        lm_mask = (pr.index >= prev_m_start) & (pr.index <= prev_m_end)
        lm = float((1+pr[lm_mask]).prod()-1) if lm_mask.any() else 0
        blm_mask = (bench_ret.index >= prev_m_start) & (bench_ret.index <= prev_m_end)
        blm = float((1+bench_ret[blm_mask]).prod()-1) if blm_mask.any() else 0
        # Last 3 months
        l3_start = pr.index[-1] - pd.DateOffset(months=3)
        l3_mask = pr.index >= l3_start
        l3 = float((1+pr[l3_mask]).prod()-1) if l3_mask.any() else 0
        bl3_mask = bench_ret.index >= l3_start
        bl3 = float((1+bench_ret[bl3_mask]).prod()-1) if bl3_mask.any() else 0
        # Last 1 year (252 trading days)
        ly_mask = pr.index >= (pr.index[-1] - pd.DateOffset(years=1))
        ly = float((1+pr[ly_mask]).prod()-1) if ly_mask.any() else 0
        bly_mask = bench_ret.index >= (bench_ret.index[-1] - pd.DateOffset(years=1))
        bly = float((1+bench_ret[bly_mask]).prod()-1) if bly_mask.any() else 0
        ym = pr.index >= pr.index[-1].replace(month=1,day=1)
        yr = float((1+pr[ym]).prod()-1) if ym.any() else 0
        # Benchmark recent
        btr = float(bench_ret.iloc[-1]) if len(bench_ret)>0 else 0
        bwr = float((1+bench_ret.iloc[-5:]).prod()-1) if len(bench_ret)>=5 else 0
        bmm = bench_ret.index >= bench_ret.index[-1].replace(day=1)
        bmr = float((1+bench_ret[bmm]).prod()-1) if bmm.any() else 0
        bym = bench_ret.index >= bench_ret.index[-1].replace(month=1,day=1)
        byr = float((1+bench_ret[bym]).prod()-1) if bym.any() else 0

        # Cache everything per user session
        _set_state(dict(returns=returns, bench_ret=bench_ret, p_ret=pr,
            returns_full=returns_full, bench_full=bench_full,
            pr_mko=pr_mko, pr_gmv=pr_gmv, pr_hrp=pr_hrp,
            pr_lw=pr_lw, pr_mp=pr_mp, pr_ell=pr_ell, pr_cust=pr_cust,
            opt_w=opt_w, cw=cw, bw=bw, wh=wh,
            gmv_w=gmv_w, hrp_w=hrp_w, lw_w=lw_w, mp_w=mp_w, ell_w=ell_w,
            est_diag=est_diag, n_port=n_port,
            est_mode=est_mode, est_mode_label=est_mode_label,
            weights_window_range=weights_window_range, is_oos=is_oos, win_y=win_y,
            mc_res=mc_res, msi=msi, roll_df=roll_df, roll_sh=roll_sh,
            tickers=tickers, days=days, rf=rf, benchmark=benchmark,
            base=base, cs=cs, wl=wl, wmode=wmode, rebal=rebal,
            mp=mp, mb=mb, m_mko=m_mko, m_gmv=m_gmv, m_hrp=m_hrp,
            m_lw=m_lw, m_mp=m_mp, m_ell=m_ell, m_cust=m_cust, mps=mps,
            _robustness_cache=None))

        # Shortest history: use raw first_valid dates captured before dropna
        if ticker_start:
            shortest_ticker = max(ticker_start, key=lambda t: ticker_start[t] or pd.Timestamp.min)
            shortest_start = ticker_start.get(shortest_ticker)
            shortest_days = int((returns_full.index[-1] - shortest_start).days * 252 / 365) if shortest_start else 0
        else:
            shortest_ticker, shortest_days = '', 0

        return jresp(dict(status='ok',
            meta=dict(tickers=tickers, ticker_names=_get_ticker_names(tickers),
                days=days, weight_label=wl,
                est_mode=est_mode, est_mode_label=est_mode_label,
                weights_window_range=weights_window_range, is_oos=is_oos, win_y=win_y,
                eval_start=pr.index[0].strftime('%Y-%m-%d'),
                eval_end=pr.index[-1].strftime('%Y-%m-%d'),
                n_days=len(pr), benchmark=benchmark, currency=base, cs=cs, rf=rf,
                last_date=pr.index[-1].strftime('%Y-%m-%d'),
                shortest_ticker=shortest_ticker, shortest_days=int(shortest_days)),
            kpis=dict(portfolio=mp, benchmark=mb),
            quant_score=qs,
            recent=dict(today=tr, week=wr, mtd=mr, ytd=yr,
                last_date=pr.index[-1].strftime('%Y-%m-%d'),
                last_month=lm, last_3m=l3, last_year=ly,
                b_today=btr, b_week=bwr, b_mtd=bmr, b_ytd=byr,
                b_last_month=blm, b_last_3m=bl3, b_last_year=bly),
            weights=dict(optimal=opt_w, gmv=gmv_w,
                hrp=hrp_w, ledoit_wolf=lw_w, marchenko_pastur=mp_w, elliptical=ell_w,
                custom=cw if cw is not None else None, backtest=bw),
            est_diag=est_diag,
            has_custom=cw is not None,
            m_mko=m_mko, m_gmv=m_gmv, m_hrp=m_hrp,
            m_lw=m_lw, m_mp=m_mp, m_ell=m_ell, m_custom=m_cust))
    except Exception as e:
        return jsonify(status='error', message=str(e), trace=traceback.format_exc()[:500]), 500

# ── Figure endpoint (lazy generation) ───────────────────────────
@app.route('/api/figure/<name>')
def api_figure(name):
    s = _get_state()
    if not s:
        return jsonify(error='Run backtest first'), 400
    try:
        fig = _gen(name, s)
        if fig is None: return jsonify(error=f'Unknown: {name}'), 404
        return jresp(fig_json(fig))
    except Exception as e:
        return jsonify(error=str(e), trace=traceback.format_exc()[:300]), 500

def _gen(name, s):
    ret, br, pr = s['returns'], s['bench_ret'], s['p_ret']
    d, rf, cs, bm = s['days'], s['rf'], s['cs'], s['benchmark']
    tk = s['tickers']
    ow, bw, cw = s['opt_w'], s['bw'], s.get('cw')
    tag = f"({s['win_y']}Y latest)" if s.get('is_oos') else "(Full Hist.)"

    if name=='cumulative_multi':
        fig = H.fig_cumulative_multi(s['mps'], br, cs)
        mode_ttl = f"Out-of-Sample Walk-Forward ({s['win_y']}Y rolling · Zero Look-Ahead)" if s.get('is_oos') else "In-Sample Full History (Static Reference)"
        fig.update_layout(title=f"Multi-Portfolio Comparison ({cs}) · {mode_ttl} · log scale")
        return fig
    if name=='metrics_table':
        seen, ml = set(), []
        for m in [s['mp'], s.get('m_mko'), s.get('m_gmv'),
                  s.get('m_hrp'), s.get('m_lw'), s.get('m_mp'), s.get('m_ell'), s.get('m_cust')]:
            if m and m['label'] not in seen:
                ml.append(m); seen.add(m['label'])
        if s['mb']['label'] not in seen:
            ml.append(s['mb'])
        return P.fig_metrics_table(ml, rf)
    if name=='drawdown': return P.fig_drawdown(pr, br)
    if name=='annual_bars': return P.fig_annual_bars(pr, br)
    if name=='annual_max_dd': return P.fig_annual_max_drawdown(pr, br)
    if name=='pnl_vs_benchmark': return P.fig_pnl_vs_benchmark(pr, br, cs=cs)
    if name=='recovery_times': return P.fig_recovery_times(pr, br, bm)
    if name=='weights_drift': return P.fig_weights_drift(s['wh'])

    if name=='efficient_frontier':
        mc_res = s['mc_res']
        msi    = s['msi']
        rf_ret = s['returns_full']
        er     = rf_ret.mean() * d
        cm     = rf_ret.cov()  * d

        def port_vol_ret(w):
            vol_ = float(np.sqrt(np.dot(w.T, np.dot(cm, w))))
            ret_ = float(np.dot(w, er))
            return ret_, vol_

        ms_ret, ms_vol = port_vol_ret(ow)

        mc_adj = mc_res.copy()
        mc_adj[2] = (mc_adj[1] - rf) / mc_adj[0]
        mc_adj[0, msi] = ms_vol
        mc_adj[1, msi] = ms_ret
        mc_adj[2, msi] = (ms_ret - rf) / ms_vol if ms_vol > 0 else 0

        hrp_r, hrp_v = port_vol_ret(s['hrp_w'])
        lw_r,  lw_v  = port_vol_ret(s['lw_w'])
        mp_r,  mp_v  = port_vol_ret(s['mp_w'])
        ell_r, ell_v = port_vol_ret(s['ell_w'])

        extra = [
            ('HRP',              hrp_v,  hrp_r,  P.C_PURPLE, 'hexagon'),
            ('Ledoit-Wolf',      lw_v,   lw_r,   P.C_YELLOW, 'triangle-up'),
            ('Marchenko-Pastur', mp_v,   mp_r,   '#ff375f',  'square'),
            ('Elliptical MLE',   ell_v,  ell_r,  '#5ac8fa',  'pentagon'),
        ]
        if cw is not None:
            cw_ret, cw_vol = port_vol_ret(cw)
            extra.append(('Custom Weights', cw_vol, cw_ret, P.C_GREEN, 'diamond'))

        return P.fig_efficient_frontier(mc_adj, msi, rf=rf, extra_markers=extra)
    if name=='donut_opt':    return P.fig_donut(ow, tk, f'Sample MLE {tag}')
    if name=='donut_gmv':    return P.fig_donut(s['gmv_w'], tk, f'Global Min Variance {tag}')
    if name=='donut_hrp':    return P.fig_donut(s['hrp_w'], tk, f'HRP {tag}')
    if name=='donut_lw':     return P.fig_donut(s['lw_w'],  tk, f'Ledoit-Wolf {tag}')
    if name=='donut_mp':     return P.fig_donut(s['mp_w'],  tk, f'Marchenko-Pastur {tag}')
    if name=='donut_ell':    return P.fig_donut(s['ell_w'], tk, f'Elliptical MLE {tag}')
    if name=='donut_custom': return P.fig_donut(cw, tk, 'Custom Weights') if cw is not None else None
    if name=='weight_comparison':
        schemes = [
            ('Sample MLE',       ow,         P.C_BLUE),
            ('GMV',              s['gmv_w'], P.C_CYAN),
            ('HRP',              s['hrp_w'], P.C_PURPLE),
            ('Ledoit-Wolf',      s['lw_w'],  P.C_YELLOW),
            ('Marchenko-Pastur', s['mp_w'],  '#ff375f'),
            ('Elliptical MLE',   s['ell_w'], '#5ac8fa'),
        ]
        if cw is not None:
            schemes.append(('Custom', cw, P.C_GREEN))
        fig = go.Figure()
        for lbl, wv, col in schemes:
            fig.add_trace(go.Bar(name=lbl, x=tk, y=wv*100, marker_color=col))
        fig.update_layout(template=P.TEMPLATE, height=440, barmode='group',
            title=f'Weight Comparison Across All Allocation Schemes (%) · {tag}',
            yaxis=dict(ticksuffix='%'),
            legend=dict(orientation='h', y=-0.20, x=0),
            margin=dict(b=80),
            paper_bgcolor=P.C_CARD, plot_bgcolor=P.C_CARD)
        return fig

    if name=='rolling_heatmap':
        rdf = s.get('roll_df')
        if rdf is None or rdf.empty:
            fig = go.Figure()
            fig.update_layout(template=P.TEMPLATE, height=340,
                title='Rolling Weights Heatmap — Select OOS Walk-Forward (3Y or 5Y) in Panel 1 to view dynamic re-optimisation weights',
                paper_bgcolor=P.C_CARD, plot_bgcolor=P.C_CARD)
            return fig
        fig = P.fig_rolling_heatmap(rdf)
        fig.update_layout(title=f"Walk-Forward Rolling Weights Regime Heatmap — {s['wl']}")
        return fig
    if name=='wf_sharpe':
        fig = go.Figure()
        rs = s.get('roll_sh', [])
        rdf = s.get('roll_df')
        x_vals = rdf.index if (rdf is not None and len(rdf) == len(rs)) else list(range(len(rs)))
        if len(rs):
            fig.add_trace(go.Scatter(x=x_vals, y=rs if isinstance(rs, list) else rs.tolist(),
                mode='lines+markers', line=dict(color=P.C_CYAN, width=2), name='OOS Period Sharpe'))
            fig.add_hline(y=0, line_color='white', opacity=0.2)
        fig.update_layout(template=P.TEMPLATE, height=360,
            title=f"Realized Out-of-Sample Sharpe per Rebalance Period — {s['wl']}",
            yaxis_title='Realized OOS Sharpe', paper_bgcolor=P.C_CARD, plot_bgcolor=P.C_CARD)
        return fig
    if name=='individual_pnl': return P.fig_individual_pnl(ret, cs=cs)
    if name=='rolling_volatility': return P.fig_rolling_volatility(pr, br, d)
    if name=='risk_attribution': return Q.fig_risk_attribution_full(ret, bw, d)
    if name=='pca_risk_directions': return Q.fig_pca_risk_directions(ret, bw, d)
    if name=='pca_loadings': return Q.fig_pca_loadings(ret, bw, d)
    if name=='performance_attribution': return P.fig_performance_attribution(ret, bw, br, d)
    if name=='var_chart': return P.fig_var_chart(pr, br, d)
    if name=='monte_carlo':
        h = int(request.args.get('horizon', 4))
        np_ = int(request.args.get('paths', 1000))
        eng = request.args.get('engine', 'gaussian')
        return Q.fig_monte_carlo_v2(pr, ret, bw, d, n_paths=np_, horizon_years=h, rf=rf, engine=eng)
    if name=='rolling_returns': return P.fig_rolling_returns(pr, br, d)
    if name=='rolling_sharpe': return P.fig_rolling_sharpe(pr, br, d, rf)
    if name=='monthly_heatmap': return P.fig_monthly_heatmap(pr)
    if name=='return_distributions': return P.fig_return_distributions(pr, br, d)
    if name=='correlation_matrix': return P.fig_correlation_matrix(ret)
    if name=='rolling_correlation':
        rc = pr.rolling(d).corr(br.reindex(pr.index))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rc.index, y=rc, fill='tozeroy', name='Rolling 1Y Corr',
            line=dict(color=P.C_CYAN, width=2), fillcolor='rgba(100,210,255,0.15)'))
        fig.update_layout(template=P.TEMPLATE, height=360,
            title='Rolling 1Y Correlation vs Benchmark',
            yaxis=dict(range=[-1,1]), hovermode='x unified',
            paper_bgcolor=P.C_CARD, plot_bgcolor=P.C_CARD)
        fig.add_hline(y=0, line_color='white', opacity=0.2)
        return fig
    return None

# ── Out-of-Sample Estimator Validation endpoint (Static Split Stress-Test) ──
@app.route('/api/oos_validation')
def api_oos_validation():
    s = _get_state()
    if not s:
        return jsonify(error='Run backtest first'), 400
    try:
        split_ratio = float(request.args.get('split', 0.70))
        objective   = request.args.get('objective', 'max_sharpe')
        nu          = request.args.get('nu', '5')
        res = Q.run_oos_validation(
            s['returns_full'], s['bench_full'], s['days'], s['rf'], s['rebal'],
            run_sim_fn=P.run_simulation,
            compute_metrics_fn=P.compute_metrics,
            split_ratio=split_ratio, objective=objective, nu=nu,
        )
        return jresp(dict(
            status='ok',
            diagnostics=res['diagnostics'],
            fig_equity=fig_json(res['fig_equity']),
            fig_table=fig_json(res['fig_table']),
            fig_weights=fig_json(res['fig_weights']),
        ))
    except Exception as e:
        return jsonify(error=str(e), trace=traceback.format_exc()[:400]), 500

# ── Statistical Robustness endpoint ────────────────────────────
@app.route('/api/robustness')
def api_robustness():
    s = _get_state()
    if not s:
        return jsonify(error='Run backtest first'), 400
    try:
        if s.get('_robustness_cache') is None:
            # Effective independent trials estimate for DSR
            n_trials = min(max(int(s.get('n_port', 8000) / 100), 10), 200)
            res = Q.compute_robustness_analysis(
                s['p_ret'], s['bench_ret'], s['returns'],
                days=s['days'], rf=s['rf'], n_trials=n_trials, n_boot=2000,
            )
            s['_robustness_cache'] = dict(
                status='ok',
                summary=res['summary'],
                fig_bootstrap=fig_json(res['fig_bootstrap']),
                fig_null=fig_json(res['fig_null']),
            )
        return jresp(s['_robustness_cache'])
    except Exception as e:
        return jsonify(error=str(e), trace=traceback.format_exc()[:400]), 500

# ── Excel Export endpoint (multi-sheet .xlsx) ──────────────────
@app.route('/api/export_excel')
def api_export_excel():
    s = _get_state()
    if not s:
        return jsonify(error='Run backtest first'), 400
    try:
        ret, br, pr = s['returns'], s['bench_ret'], s['p_ret']
        tk = s['tickers']
        days, rf = s['days'], s['rf']
        qs = compute_quant_score(s['mp'], s['mb'])

        # Sheet 1: Portfolio_Summary
        meta_rows = [
            ('Estimation Mode', s.get('est_mode_label', '')),
            ('Active Weighting Scheme', s['wl']),
            ('Weights Estimation Window', s.get('weights_window_range', '')),
            ('Tickers', ', '.join(tk)),
            ('Number of Assets', len(tk)),
            ('Evaluated Start Date', pr.index[0].strftime('%Y-%m-%d')),
            ('Evaluated End Date', pr.index[-1].strftime('%Y-%m-%d')),
            ('Evaluated Trading Days', len(pr)),
            ('Benchmark', s['benchmark']),
            ('Base Currency', s['base']),
            ('Risk-Free Rate', rf),
            ('Rebalance Months', ', '.join(str(m) for m in s['rebal'])),
            ('Quant Score (0-100)', qs),
            ('', ''),
        ]
        kpi_keys = [
            ('Ann. Return (CAGR)', 'ann_ret'),
            ('Ann. Volatility', 'ann_vol'),
            ('Sharpe Ratio', 'sharpe'),
            ('Sortino Ratio', 'sortino'),
            ('Max Drawdown', 'max_dd'),
            ('Calmar Ratio', 'calmar'),
            ('Beta vs Benchmark', 'beta'),
            ("Jensen's Alpha", 'alpha'),
            ('Annual Win Rate', 'win_rate'),
            ('VaR 95% (daily)', 'var_95'),
            ('CVaR 95% (daily)', 'cvar_95'),
            ('Avg Recovery (days)', 'avg_recovery'),
            ('Max Recovery (days)', 'max_recovery'),
        ]
        summary_rows = [{'Item': k, 'Portfolio': v, 'Benchmark': ''} for k, v in meta_rows]
        for lbl, key in kpi_keys:
            summary_rows.append({
                'Item': lbl,
                'Portfolio': round(float(s['mp'][key]), 6),
                'Benchmark': round(float(s['mb'][key]), 6),
            })
        df_summary = pd.DataFrame(summary_rows)

        # Sheet 2: Composition_Weights
        asset_ann_ret = [float((1.0 + ret[t].dropna()).prod() ** (days / len(ret[t].dropna())) - 1.0) for t in tk]
        asset_ann_vol = [float(ret[t].std() * np.sqrt(days)) for t in tk]
        comp_dict = {
            'Ticker': tk,
            f"Active_Weight ({s['wl']})": np.round(s['bw'], 6),
            'Sample_MLE': np.round(s['opt_w'], 6),
            'Global_Min_Variance': np.round(s['gmv_w'], 6),
            'HRP': np.round(s['hrp_w'], 6),
            'Ledoit_Wolf': np.round(s['lw_w'], 6),
            'Marchenko_Pastur': np.round(s['mp_w'], 6),
            'Elliptical_MLE': np.round(s['ell_w'], 6),
            'Asset_Ann_Return': np.round(asset_ann_ret, 6),
            'Asset_Ann_Vol': np.round(asset_ann_vol, 6),
        }
        if s.get('cw') is not None:
            comp_dict['Custom_Weights'] = np.round(s['cw'], 6)
        df_comp = pd.DataFrame(comp_dict)

        # Sheet 3: Daily_Returns (Portfolio + Benchmark + Cum Growth + Drawdown + Every Asset)
        br_aligned = br.reindex(pr.index)
        p_cum = (1.0 + pr).cumprod()
        b_cum = (1.0 + br_aligned.fillna(0)).cumprod()
        p_dd = (p_cum - p_cum.cummax()) / p_cum.cummax()
        df_daily = pd.DataFrame({
            'Date': pr.index.strftime('%Y-%m-%d'),
            'Portfolio_Daily_Return': pr.values,
            'Benchmark_Daily_Return': br_aligned.values,
            'Portfolio_Cum_Growth': p_cum.values,
            'Benchmark_Cum_Growth': b_cum.values,
            'Portfolio_Drawdown': p_dd.values,
        })
        for t in tk:
            df_daily[f'Asset_{t}_Return'] = ret[t].reindex(pr.index).values

        # Sheet 4: All_Schemes_Metrics
        seen, ml = set(), []
        for m in [s['mp'], s.get('m_mko'), s.get('m_gmv'),
                  s.get('m_hrp'), s.get('m_lw'), s.get('m_mp'), s.get('m_ell'), s.get('m_cust'), s['mb']]:
            if m and m['label'] not in seen:
                ml.append(m); seen.add(m['label'])
        df_schemes = pd.DataFrame(ml)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Portfolio_Summary', index=False)
            df_comp.to_excel(writer, sheet_name='Composition_Weights', index=False)
            df_daily.to_excel(writer, sheet_name='Daily_Returns', index=False)
            df_schemes.to_excel(writer, sheet_name='All_Schemes_Metrics', index=False)
            # Auto-fit column widths nicely
            for ws in writer.sheets.values():
                for col in ws.columns:
                    max_len = max((len(str(cell.value or '')) for cell in col[:200]), default=10)
                    ws.column_dimensions[col[0].column_letter].width = min(max(max_len + 3, 12), 42)
        buf.seek(0)

        safe_lbl = s['wl'].replace(' ', '_').replace('/', '-')
        fname = f"FastbreaK_{safe_lbl}_{s['base']}.xlsx"
        return send_file(
            buf,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=fname,
        )
    except Exception as e:
        return jsonify(error=str(e), trace=traceback.format_exc()[:400]), 500

# ── N+1 Asset Suggestion scan endpoint ─────────────────────────
@app.route('/api/scan_sp500', methods=['POST'])
def api_scan():
    s = _get_state()
    if not s:
        return jsonify(error='Run backtest first'), 400
    try:
        d = request.json
        nw = float(d.get('weight', 10)) / 100.0
        ranking = d.get('ranking', 'ir')
        lookback_years = int(d.get('lookback_years', 3))
        universe = d.get('universe', 'ticks_top')
        fx_prices = None
        if s.get('base', 'USD') != 'USD':
            end_dt = s['returns'].index[-1].strftime('%Y-%m-%d')
            start_dt = s['returns'].index[0].strftime('%Y-%m-%d')
            fx_prices = H.download_fx_rate(start_dt, end_dt)
        sug = H.run_suggestion_engine(
            s['returns'], s['bw'], s['bench_ret'], s['days'],
            rf=s['rf'], new_weight=nw, top_n=15,
            lookback_years=lookback_years, ranking=ranking,
            universe=universe, base_currency=s.get('base', 'USD'),
            fx_prices=fx_prices,
        )
        rdf = sug.get('results', pd.DataFrame())
        if rdf.empty:
            return jsonify(status='empty', results=[],
                           elapsed_s=sug.get('elapsed_s', 0),
                           n_evaluated=sug.get('n_candidates_evaluated', 0))
        tfig = P.fig_suggestion_table(sug)
        return jresp(dict(status='ok',
            base_vol=sug['base_vol'], base_ret=sug['base_ret'],
            base_sharpe=sug['base_sharpe'], base_ir=sug.get('base_ir', 0),
            elapsed_s=sug.get('elapsed_s', 0),
            n_evaluated=sug.get('n_candidates_evaluated', 0),
            lookback_years=lookback_years,
            universe=universe,
            start_date=sug.get('start_date', ''),
            table=fig_json(tfig) if tfig else None,
            results=rdf.to_dict('records')))
    except Exception as e:
        return jsonify(error=str(e), trace=traceback.format_exc()[:400]), 500

# ── Portfolio A vs Portfolio B Comparison endpoint ─────────────
@app.route('/api/compare_portfolios', methods=['POST'])
def api_compare_portfolios():
    s = _get_state()
    if not s:
        return jsonify(error='Run backtest first'), 400
    try:
        d = request.json or {}
        label_b = (d.get('label') or 'Portfolio B').strip()
        raw_t = d.get('tickers', [])
        raw_c = d.get('currencies', [])
        raw_w = d.get('weights', [])

        rows_b = []
        for i, t in enumerate(raw_t):
            t = (t or '').strip().upper()
            c = raw_c[i] if i < len(raw_c) else 'USD'
            w = raw_w[i] if i < len(raw_w) else None
            if t:
                try:
                    wv = float(w)
                except (TypeError, ValueError):
                    wv = 0.0
                if wv > 0:
                    rows_b.append((t, c, wv))

        if not rows_b:
            return jsonify(status='error', message='Inserisci almeno un ticker con peso > 0 per Portfolio B'), 400

        tickers_b = [x[0] for x in rows_b]
        tcurr_b = {x[0]: x[1] for x in rows_b}
        w_b_raw = np.array([x[2] for x in rows_b], dtype=float)
        if w_b_raw.sum() <= 0:
            return jsonify(status='error', message='La somma dei pesi di Portfolio B deve essere > 0'), 400
        w_b = w_b_raw / w_b_raw.sum()

        # Download Portfolio B over Portfolio A's date span
        start_dt = s['returns'].index[0].strftime('%Y-%m-%d')
        end_dt = (s['returns'].index[-1] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        ret_b, bench_b, _ = P.download_data(tickers_b, s['benchmark'], start_dt, end_dt)
        if ret_b.empty:
            return jsonify(status='error', message='Nessun dato scaricato per Portfolio B'), 400

        # Align weights to actually downloaded columns
        valid_cols = list(ret_b.columns)
        col_w_map = dict(zip(tickers_b, w_b))
        w_b_aligned = np.array([col_w_map[c] for c in valid_cols], dtype=float)
        w_b_aligned /= w_b_aligned.sum()

        # FX conversion to match Portfolio A's base currency
        base = s['base']
        needs_fx = base != 'USD' or any(tcurr_b.get(t, 'USD') != 'USD' for t in valid_cols)
        if needs_fx:
            fx = H.download_fx_rate(start_dt, end_dt)
            ret_b, bench_b = H.convert_all_returns(ret_b, bench_b, tcurr_b, base, fx)
            ret_b = ret_b.dropna(how='all')

        _, pr_b_full = P.run_simulation(ret_b, w_b_aligned, s['rebal'])

        # Align Portfolio A, Portfolio B, and Benchmark on exact common dates
        pr_a_full = s['p_ret']
        bm_full = s['bench_ret']
        common = pr_a_full.index.intersection(pr_b_full.index).intersection(bm_full.index)
        if len(common) < 30:
            return jsonify(status='error', message='Periodo in comune tra Portfolio A e B troppo breve (<30 giorni)'), 400

        pr_a = pr_a_full.loc[common]
        pr_b = pr_b_full.loc[common]
        bm_c = bm_full.loc[common]

        days, rf, cs = s['days'], s['rf'], s['cs']
        m_a = P.compute_metrics(pr_a, bm_c, days, f"A: {s['wl']}", rf)
        m_b = P.compute_metrics(pr_b, bm_c, days, f"B: {label_b}", rf)
        m_bm = P.compute_metrics(bm_c, bm_c, days, s['benchmark'], rf)

        qs_a = compute_quant_score(m_a, m_bm)
        qs_b = compute_quant_score(m_b, m_bm)

        charts = Q.build_compare_ab_charts(
            pr_a, pr_b, bm_c, s['wl'], label_b,
            m_a, m_b, m_bm, s['tickers'], s['bw'], valid_cols, w_b_aligned,
            rf=rf, cs=cs,
        )

        return jresp(dict(
            status='ok',
            common_days=len(common),
            date_start=common[0].strftime('%Y-%m-%d'),
            date_end=common[-1].strftime('%Y-%m-%d'),
            label_a=s['wl'], label_b=label_b,
            qs_a=qs_a, qs_b=qs_b,
            m_a=m_a, m_b=m_b, m_bench=m_bm,
            corr_ab=charts['corr_ab'],
            fig_equity=fig_json(charts['fig_equity']),
            fig_table=fig_json(charts['fig_table']),
            fig_dd_ann=fig_json(charts['fig_dd_ann']),
            fig_alloc=fig_json(charts['fig_alloc']),
        ))
    except Exception as e:
        return jsonify(status='error', message=str(e), trace=traceback.format_exc()[:400]), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*60)
    print("  FastbreaK Quantitative Portfolio Engine")
    print(f"  Open  ->  http://127.0.0.1:{port}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', debug=False, port=port)
