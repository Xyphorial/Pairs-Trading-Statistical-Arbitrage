"""Pairs research: formation -> validation selection -> untouched test.
Run python project.py --help. No automatic synthetic fallback.
"""
import argparse
import hashlib
import itertools
import json
import platform
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint

ROOT = Path(__file__).resolve().parent
TRADE_COLUMNS = ['entry_date','exit_date','side','holding_sessions','net_pnl','return_on_entry_nav','reason']


def download(cfg, path, universe_path=None):
    """Current constituent snapshot, explicitly subject to survivorship bias."""
    import requests
    import yfinance as yf
    if universe_path:
        u = pd.read_csv(universe_path)
    else:
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        from io import StringIO
        u = pd.read_csv(StringIO(r.text))
    if 'Symbol' not in u:
        raise ValueError('Universe must have a Symbol column.')
    u['Symbol'] = u.Symbol.str.replace('.', '-', regex=False)
    u.to_csv(path.parent / 'universe_snapshot.csv', index=False)
    parts = []
    tickers = sorted(u.Symbol.unique())
    for i in range(0, len(tickers), 40):
        batch = tickers[i:i+40]
        raw = yf.download(batch, start=cfg['start'], end=cfg['end'],
                          auto_adjust=True, progress=False, timeout=20, threads=True)
        if raw is not None and not raw.empty:
            parts.append(raw['Close'])
        print(f'Downloaded batch {i//40+1}', flush=True)
    if not parts:
        raise RuntimeError('No Yahoo prices obtained. Supply an adjusted-close CSV using --prices. No demo fallback.')
    prices = pd.concat(parts, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.to_csv(path)
    pd.DataFrame({'requested':tickers, 'received':[t in prices and prices[t].notna().any() for t in tickers]}).to_csv(path.parent/'download_audit.csv', index=False)
    return prices


def demo(path):
    """Engineered synthetic prices, for software verification only."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range('2015-01-01', '2025-12-31')
    data = {}
    for j in range(6):
        x = 4.4 + np.cumsum(rng.normal(0.00015, 0.012, len(dates)))
        e = np.zeros(len(dates))
        for i in range(1, len(dates)):
            e[i] = 0.92*e[i-1] + rng.normal(0, 0.008)
        data[f'SIM{j}A'] = np.exp(x)
        data[f'SIM{j}B'] = np.exp(0.3 + 0.9*x + e)
    p = pd.DataFrame(data, index=dates)
    p.index.name = 'Date'
    p.to_csv(path)
    return p


def load_prices(path, cfg):
    p = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if p.index.has_duplicates or p.columns.has_duplicates:
        raise ValueError('Duplicate dates or tickers.')
    p = p.apply(pd.to_numeric, errors='raise')
    p = p.loc[(p.index >= cfg['start']) & (p.index < cfg['end'])]
    if np.isinf(p.to_numpy(dtype=float)).any() or (p <= 0).any().any():
        raise ValueError('Prices must be finite and positive.')
    formation = p.loc[:cfg['formation_end']]
    # Eligibility uses ONLY formation history. Never select using later completeness.
    eligible = formation.columns[formation.notna().all() & (formation.count() >= 504)]
    pd.DataFrame({'ticker':p.columns, 'eligible_formation':[c in eligible for c in p]}).to_csv(path.parent/'cleaning_audit.csv', index=False)
    if len(eligible) < 2:
        raise ValueError('Need >=2 stocks with complete formation history of at least 504 sessions.')
    return p[eligible]


def screen(prices):
    logp = np.log(prices)
    diagnostics = []
    for t in logp:
        a = logp[t]
        diagnostics.append({'ticker':t, 'level_adf_p':adfuller(a, maxlag=5, autolag='AIC')[1],
                            'diff_adf_p':adfuller(a.diff().dropna(), maxlag=5, autolag='AIC')[1]})
    diag = pd.DataFrame(diagnostics).set_index('ticker')
    rows = []
    # Alphabetical orientation fixed before testing, one test per unordered pair.
    for y, x in itertools.combinations(sorted(logp.columns), 2):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stat, p, _ = coint(logp[y], logp[x], trend='c', maxlag=5, autolag='AIC')
        beta, alpha = np.polyfit(logp[x], logp[y], 1)
        i1 = all(diag.loc[t,'level_adf_p'] > .05 and diag.loc[t,'diff_adf_p'] < .05 for t in [y,x])
        rows.append(dict(y=y, x=x, p_value=p, test_stat=stat, beta=beta,
                         alpha=alpha, correlation=prices[y].pct_change().corr(prices[x].pct_change()), i1_screen=bool(i1)))
    tab = pd.DataFrame(rows)
    tab['q_value'] = multipletests(tab.p_value.fillna(1), method='fdr_bh')[1]
    return tab.sort_values(['p_value','y','x']), diag.reset_index()


def choose_pairs(tab, threshold, cfg):
    candidates = tab[(tab.p_value <= threshold) & (tab.q_value <= cfg['fdr']) & tab.i1_screen & (tab.beta > 0)]
    used, pairs = set(), []
    for r in candidates.itertuples():
        if r.y in used or r.x in used:
            continue
        pairs.append((r.y,r.x)); used.update([r.y,r.x])
        if len(pairs) >= cfg['max_pairs']:
            break
    return pairs


def signals(p, lookback):
    """Fit with observations strictly before signal close; no future values."""
    y, x = np.log(p.iloc[:,0]), np.log(p.iloc[:,1])
    xp, yp = x.shift(1), y.shift(1)
    mx, my = xp.rolling(lookback).mean(), yp.rolling(lookback).mean()
    vx = xp.rolling(lookback).var()
    beta = xp.rolling(lookback).cov(yp) / vx
    alpha = my - beta*mx
    # Residual std for the SAME current regression, not a mix of historical betas.
    vy = yp.rolling(lookback).var()
    cov = xp.rolling(lookback).cov(yp)
    sd = np.sqrt((vy + beta**2*vx - 2*beta*cov).clip(lower=0))
    spread = y-alpha-beta*x
    z = spread / sd.replace(0,np.nan)
    return pd.DataFrame({'alpha':alpha,'beta':beta,'spread':spread,'z':z})


def backtest(p, sig, start, end, params, cfg):
    """Fixed adjusted units while holding; close t-1 signal fills at close t.
    Mark old units to close t FIRST, charge turnover on actual dollar trades.
    One pair sleeve starts at NAV=1; cash earns zero. Forced terminal liquidation.
    """
    block = p.loc[start:end]
    if block.empty:
        raise ValueError('Empty evaluation period')
    if block.isna().any().any():
        raise ValueError('Selected pair missing evaluation prices. No silent survivor filtering or filling.')
    lag = sig.shift(1).reindex(block.index)[['z','beta']].to_numpy(float)
    px = block.to_numpy(float)
    nav, units, state, age = 1., np.zeros(2), 0, 0
    fee = (cfg['commission_bps']+cfg['slippage_bps']) / 10000
    daily, trades = [], []
    entry_nav = entry_date = side = None
    trade_pnl = 0.
    for i, date in enumerate(block.index):
        prev_nav = nav
        gross = float(units @ (px[i]-px[i-1])) if i else 0.
        # Annual borrow prorated by calendar days; includes weekends.
        days = (date-block.index[i-1]).days if i else 0
        borrow = float(np.maximum(-units,0) @ px[max(0,i-1)]) * cfg['borrow_bps_year']/10000*days/365
        nav += gross-borrow
        if state:
            age += 1
            trade_pnl += gross-borrow
        z, beta = lag[i]
        target, reason = state, ''
        valid = np.isfinite(z) and np.isfinite(beta) and beta > 0
        if state:
            if not valid: target, reason = 0, 'invalid_signal'
            elif state*z >= -params['exit']: target, reason = 0, 'mean_reversion'
            elif abs(z) >= params['stop']: target, reason = 0, 'z_stop'
            elif age >= params['holding']: target, reason = 0, 'time_stop'
        elif valid and params['entry'] <= abs(z) < params['stop']:
            target = -int(np.sign(z))
        if i == len(block)-1:
            target, reason = 0, 'period_end'
        turnover, cost = 0., 0.
        if target != state:
            if target:
                entry_nav, entry_date, side = nav, date, target
                # Log-regression beta is a DOLLAR exposure ratio, not a share ratio.
                weights = target*np.array([1.,-beta])/(1+abs(beta))
                new_units = weights*nav/px[i]
                age, trade_pnl = 0, 0.
            else:
                new_units = np.zeros(2)
            turnover = float(np.abs(new_units-units) @ px[i])
            cost = turnover*fee
            nav -= cost; trade_pnl -= cost
            units = new_units
            if not target:
                trades.append(dict(entry_date=entry_date,exit_date=date,side=side,holding_sessions=age,
                                   net_pnl=trade_pnl,return_on_entry_nav=trade_pnl/entry_nav,reason=reason))
            state = target
        if nav <= 0:
            raise RuntimeError('Sleeve insolvent; strategy invalid under these settings.')
        daily.append(dict(date=date,nav=nav,net_return=nav/prev_nav-1,gross_pnl=gross,
                          cost=cost,borrow=borrow,turnover=turnover/prev_nav,state=state,z=z))
    return pd.DataFrame(daily).set_index('date'), pd.DataFrame(trades, columns=TRADE_COLUMNS)


def metrics(d, trades=None):
    r = d.net_return
    eq = (1+r).cumprod()
    peak = eq.cummax().clip(lower=1)
    vol = r.std(ddof=1)*np.sqrt(252)
    out = dict(total_return=eq.iloc[-1]-1, annualized_return=eq.iloc[-1]**(252/len(r))-1,
               annualized_volatility=vol, sharpe=r.mean()*252/vol if vol > 0 else 0.,
               max_drawdown=(eq/peak-1).min(), annualized_turnover=d.turnover.mean()*252)
    if trades is not None:
        out.update(trades=len(trades),win_rate=float((trades.net_pnl > 0).mean()) if len(trades) else None,
                   average_trade_return=float(trades.return_on_entry_nav.mean()) if len(trades) else None)
    return out


def evaluate(prices, pairs, cache, start, end, params, cfg):
    idx = prices.loc[start:end].index
    if not pairs:
        d = pd.DataFrame({'net_return':0.,'turnover':0.},index=idx)
        return d, pd.DataFrame(), {}
    ds, ts, detail = [], [], {}
    for y,x in pairs:
        d,t = backtest(prices[[y,x]],cache[(y,x,params['lookback'])],start,end,params,cfg)
        name = f'{y}_{x}'
        detail[name] = d
        ds.append(d)
        if len(t):
            t['pair'] = name; ts.append(t)
    # Equal initial capital in independent sleeves, NO daily rebalancing.
    equity = pd.concat([d.nav for d in ds],axis=1).mean(axis=1)
    ret = equity.pct_change(); ret.iloc[0] = equity.iloc[0]-1
    dollar_turn = pd.concat([d.turnover*d.nav.shift(1).fillna(1) for d in ds],axis=1).mean(axis=1)
    agg = pd.DataFrame({'net_return':ret,'turnover':dollar_turn/equity.shift(1).fillna(1),'nav':equity})
    trades = pd.concat(ts,ignore_index=True) if ts else pd.DataFrame(columns=TRADE_COLUMNS+['pair'])
    return agg,trades,detail


def plot_outputs(out, tab, prices, pairs, cache, params, results):
    plt.style.use('seaborn-v0_8-whitegrid')
    suffix = ' [SYNTHETIC SOFTWARE CHECK]' if all(str(c).startswith('SIM') for c in prices.columns) else ''
    fig,ax = plt.subplots(figsize=(10,5))
    top = tab.head(20).iloc[::-1]
    ax.barh(top.y+' / '+top.x,top.p_value,color='#216e8c')
    ax.set(xlabel='Formation Engle-Granger p-value',title='Candidate pairs (formation only)'+suffix)
    fig.tight_layout(); fig.savefig(out/'candidate_pvalues.png',dpi=160); plt.close(fig)
    fig,ax = plt.subplots(figsize=(10,5))
    for label,(d,_,_) in results.items():
        ax.plot(d.index,(1+d.net_return).cumprod(),label=label)
    ax.set(title='Portfolio NAV (each period starts at 1)'+suffix,ylabel='Net asset value'); ax.legend()
    fig.tight_layout(); fig.savefig(out/'equity_curve.png',dpi=160); plt.close(fig)
    for y,x in pairs:
        s = cache[(y,x,params['lookback'])]
        fig,axs = plt.subplots(2,1,figsize=(11,6),sharex=True)
        axs[0].plot(s.index,s.spread,lw=.7); axs[0].set(title=f'{y} / {x}: rolling residual'+suffix,ylabel='Log spread')
        axs[1].plot(s.index,s.z,lw=.7)
        for level in [-params['entry'],params['entry']]: axs[1].axhline(level,color='#cb6843',ls='--')
        for level in [-params['exit'],params['exit']]: axs[1].axhline(level,color='green',ls=':')
        axs[1].set(ylabel='Z-score')
        fig.tight_layout(); fig.savefig(out/f'spread_z_{y}_{x}.png',dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prices',type=Path,help='Adjusted-close CSV: Date,ticker1,ticker2,...')
    ap.add_argument('--download',action='store_true')
    ap.add_argument('--universe',type=Path,help='Optional constituent CSV with Symbol column')
    ap.add_argument('--demo',action='store_true',help='Synthetic SOFTWARE CHECK, never real performance')
    ap.add_argument('--config',type=Path,default=ROOT/'config.json')
    ap.add_argument('--out',type=Path,default=None)
    args = ap.parse_args()
    if sum([bool(args.prices),args.download,args.demo]) != 1:
        ap.error('Choose exactly one: --prices PATH, --download, --demo')
    cfg = json.loads(args.config.read_text())
    out = args.out or ROOT/'results'/('synthetic_check' if args.demo else 'real_data')
    out.mkdir(parents=True,exist_ok=True)
    path = args.prices or ROOT/'data'/('SYNTHETIC_prices.csv' if args.demo else 'adjusted_close.csv')
    path.parent.mkdir(parents=True,exist_ok=True)
    if args.download: download(cfg,path,args.universe)
    if args.demo: demo(path)
    p = load_prices(path,cfg)
    formation = p.loc[:cfg['formation_end']]
    validation_start = (pd.Timestamp(cfg['formation_end'])+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    test_start = (pd.Timestamp(cfg['validation_end'])+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    if len(p.loc[validation_start:cfg['validation_end']]) < 126 or len(p.loc[test_start:]) < 126:
        raise ValueError('Need at least 126 sessions in validation and final test.')
    tab,diag = screen(formation)
    tab.to_csv(out/'cointegration_tests.csv',index=False); diag.to_csv(out/'adf_diagnostics.csv',index=False)
    pool = set(itertools.chain.from_iterable(choose_pairs(tab,t,cfg) for t in cfg['p_thresholds']))
    cache = {(y,x,w):signals(p[[y,x]],w) for y,x in pool for w in cfg['lookbacks']}
    rows = []
    grid = itertools.product(cfg['p_thresholds'],cfg['lookbacks'],cfg['entries'],cfg['exits'],cfg['stop_z'],cfg['max_holding'])
    for pt,w,en,ex,stop,hold in grid:
        params = dict(p_threshold=pt,lookback=w,entry=en,exit=ex,stop=stop,holding=hold)
        pairs = choose_pairs(tab,pt,cfg)
        d,t,_ = evaluate(p,pairs,cache,validation_start,cfg['validation_end'],params,cfg)
        m = metrics(d,t)
        rows.append({**params,**m,'pair_count':len(pairs)})
    sens = pd.DataFrame(rows)
    sens.to_csv(out/'validation_sensitivity.csv',index=False)
    acceptable = sens[(sens.trades >= cfg['min_validation_trades']) & (sens.pair_count > 0)]
    if acceptable.empty:
        (out/'STATUS.txt').write_text('NO QUALIFYING STRATEGY. No final-test strategy selected. Inspect formation/FDR and validation constraints. Do not weaken filters after viewing the final test.\n')
        print('No qualifying strategy; see STATUS.txt'); return
    # Fixed tie-break: best net validation Sharpe, then lower turnover.
    best = acceptable.sort_values(['sharpe','annualized_turnover'],ascending=[False,True],kind='stable').iloc[0]
    params = {k:(int(best[k]) if k in ['lookback','holding'] else float(best[k])) for k in ['p_threshold','lookback','entry','exit','stop','holding']}
    pairs = choose_pairs(tab,params['p_threshold'],cfg)
    results = {}
    periods = [('formation',formation.index[0].strftime('%Y-%m-%d'),cfg['formation_end']),
               ('validation',validation_start,cfg['validation_end']),('test',test_start,cfg['end'])]
    performance = []
    # Test period is touched for performance ONLY AFTER all settings are frozen.
    for label,start,end in periods:
        result = evaluate(p,pairs,cache,start,end,params,cfg)
        d,t,detail = result; results[label] = result
        d.to_csv(out/f'{label}_portfolio.csv'); t.to_csv(out/f'{label}_trades.csv',index=False)
        for name,dd in detail.items(): dd.to_csv(out/f'{label}_{name}_daily.csv')
        performance.append({'period':label,**metrics(d,t)})
    pd.DataFrame(performance).to_csv(out/'performance.csv',index=False)
    # Cost is a stress assumption, never optimized down to inflate results.
    cost_rows = []
    for multiplier in [0,1,2,4]:
        stress = {**cfg, 'commission_bps':cfg['commission_bps']*multiplier,'slippage_bps':cfg['slippage_bps']*multiplier}
        d,t,_ = evaluate(p,pairs,cache,test_start,cfg['end'],params,stress)
        cost_rows.append({'execution_cost_multiplier':multiplier,**metrics(d,t)})
    pd.DataFrame(cost_rows).to_csv(out/'test_cost_stress.csv',index=False)
    # Post-hoc diagnostics cannot change selected pairs or parameters.
    test_pairs = []
    for y,x in pairs:
        testp = np.log(p.loc[test_start:,[y,x]])
        test_pairs.append({'y':y,'x':x,'test_coint_p':coint(testp[y],testp[x],maxlag=5,autolag='AIC')[1]})
    pd.DataFrame(test_pairs).to_csv(out/'test_cointegration_diagnostic.csv',index=False)
    meta = dict(data_kind='SYNTHETIC_SOFTWARE_CHECK' if args.demo else 'USER_OR_YAHOO_ADJUSTED_CLOSE',
                selected_parameters=params,selected_pairs=pairs,config=cfg,
                input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),python=platform.python_version(),
                formation_stocks=len(p.columns),formation_tests=len(tab),
                note='Current constituent snapshots have survivorship bias. Synthetic results are not market evidence.')
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    plot_outputs(out,tab,p,pairs,cache,params,results)
    print(json.dumps(meta,indent=2)); print(pd.DataFrame(performance).to_string(index=False))

if __name__ == '__main__':
    main()
