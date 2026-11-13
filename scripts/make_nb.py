import nbformat as nbf, sys
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# Housing effort in Australia and New Zealand: investigation notebook

Companion to the *Housing Effort Down Under* write-up, which re-runs the method of [The Housing Market is NUTS](https://nuts.logical.li/) for Australia and NZ.

**Effort** = median dwelling price ÷ median (or average) annual earnings, i.e. years of wage per dwelling. Neither country publishes price per m², so this is not the original's months-per-m² metric.

Sections:
1. Load the 2023 cross-section (50 AU labour-market regions, 57 NZ districts with metros merged)
2. Density vs effort fits, with the original's IQR outlier rule
3. Why NZ differs: price and wage elasticities, population growth
4. Mortgage simulator (change the assumptions)
5. Time series, 1986–2025: capitals, metros, national OECD ratios
6. Has the density relationship changed over time?
7. Maps
8. Where is housing dearer than density predicts?
9. The bid-rent curve: effort vs distance from the CBD
10. Have NZ districts converged? (and the Auckland halo)
11. Repayment burden with historical interest rates (AU)
12. Open questions

Rebuild everything from `raw/` with the scripts in `scripts/` (see the last cell).""")
code("""from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, math
ROOT = Path.cwd() if (Path.cwd()/'data').exists() else Path('/Volumes/T7/vibe/economics/housing-policy-research/au-nz-density')
DATA, TS, RAW = ROOT/'data', ROOT/'data'/'timeseries', ROOT/'raw'
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 30)
plt.rcParams.update({'figure.figsize': (9, 5), 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .3})
AU, NZ = '#2a78d6', '#eb6834'
au = pd.read_csv(DATA/'au_labour_market_regions.csv', index_col=0)
nz = pd.read_csv(DATA/'nz_labour_market_regions.csv', index_col=0).dropna(subset=['eff'])
print(len(au), 'AU regions;', len(nz), 'NZ districts')
au[['eff','eff_unit','dens','accfood','price','wage','workers_pow_2021','capital']].sort_values('eff', ascending=False).head(10).round(2)""")
md("## 2. Density vs effort\n`fit_density` reproduces the original's method: OLS of effort on ln(jobs/km²), outliers are residuals beyond 1.5×IQR, then refit without them.")
code("""def ols(X, y):
    X = np.column_stack(list(X) + [np.ones(len(y))]); b, *_ = np.linalg.lstsq(X, y, rcond=None); yh = X @ b
    return b, 1 - ((y - yh)**2).sum() / ((y - y.mean())**2).sum(), y - yh

def iqr_outliers(res, k=1.5):
    q1, q3 = np.percentile(res, [25, 75]); i = q3 - q1; return (res < q1 - k*i) | (res > q3 + k*i)

def fit_density(df, y='eff', x='dens', tourism=None):
    d = df.dropna(subset=[y, x]); lx = np.log(d[x].values); yy = d[y].values
    b, r2, res = ols([lx], yy); out = iqr_outliers(res); bc, r2c, _ = ols([lx[~out]], yy[~out])
    r = dict(n=len(d), slope=b[0], intercept=b[1], r2=r2, r2_clean=r2c, outliers=list(d.index[out]))
    if tourism:
        t = d.dropna(subset=[tourism]); r['r2_tourism'] = ols([t[tourism].values], t[y].values)[1]
        r['r2_density_tourism'] = ols([np.log(t[x].values), t[tourism].values], t[y].values)[1]
    return r

res_au = fit_density(au, tourism='accfood'); res_nz = fit_density(nz, tourism='gn_pw')
pd.DataFrame({'Australia': res_au, 'New Zealand': res_nz}).T""")
code("""fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
for ax, df, r, col, title in [(axes[0], au, res_au, AU, 'Australia, 50 labour-market regions'), (axes[1], nz, res_nz, NZ, 'New Zealand, 57 districts')]:
    ax.scatter(df.dens, df.eff, s=28, color=col, edgecolor='white', lw=.6, zorder=3)
    o = df.loc[r['outliers']]; ax.scatter(o.dens, o.eff, s=120, facecolor='none', edgecolor='k', lw=1.2, zorder=4)
    xs = np.logspace(np.log10(df.dens.min()), np.log10(df.dens.max()), 50); ax.plot(xs, r['slope']*np.log(xs) + r['intercept'], '--', color=col)
    for n in df.sort_values('eff').tail(4).index.tolist() + df.sort_values('dens').tail(2).index.tolist():
        ax.annotate(n.replace('Greater ', ''), (df.dens[n], df.eff[n]), fontsize=8, xytext=(4, 4), textcoords='offset points')
    ax.set_xscale('log'); ax.set_xlabel('jobs per km² (log)'); ax.set_ylabel('years of median wage per median dwelling')
    ax.set_title(f"{title}\\nR² {r['r2']:.2f} (clean {r['r2_clean']:.2f})", fontsize=11)
plt.tight_layout()""")
md("**Try:** swap `eff` for `eff_unit` (units only) or `eff_house` (houses only) in Australia, or raise `k` in `iqr_outliers` to see how sensitive the cleaned R² is.")
code("""pd.DataFrame({k: fit_density(au, y=k) for k in ['eff', 'eff_unit', 'eff_house']}).T[['n','slope','r2','r2_clean','outliers']]""")
md("## 3. Why NZ differs\nSplit effort into numerator and denominator. Elasticity = change in log price (or log wage) per unit change in log density. Then test population growth as the demand measure.")
code("""raw_nz = pd.read_csv(RAW/'nz'/'nz_raw.csv', index_col=0).drop(index=['Chatham Islands']).dropna()
raw_nz['dens'] = raw_nz.employees / raw_nz.area_km2; raw_nz['eff'] = raw_nz.median_sale_price / raw_nz.wage
rows = {}
for name, d, p in [('Australia', au, 'price'), ('New Zealand', raw_nz, 'median_sale_price')]:
    lx = np.log(d.dens.values)
    bp, r2p, _ = ols([lx], np.log(d[p].values)); bw, r2w, _ = ols([lx], np.log(d.wage.values))
    rows[name] = {'price +% per 10x density': (10**bp[0]-1)*100, 'wage +% per 10x density': (10**bw[0]-1)*100, 'price R²': r2p, 'wage R²': r2w}
pd.DataFrame(rows).T.round(2)""")
code("""rear = pd.read_csv(RAW/'nz'/'rear'/'all-data.csv', dtype=str, low_memory=False)
import unicodedata, re
def norm(x):
    x = unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode(); x = re.sub(r'\\s+(District|City)$', '', x).strip()
    return {'Wanganui':'Whanganui','Southland district':'Southland','Waikato district':'Waikato','Wellington city':'Wellington'}.get(x, x)
pop = rear[(rear.indicator=='population estimates') & (rear['transform']=='original')]
pop = pop.assign(ta=pop.area.map(norm), value=pop.value.astype(float), year=pop.year.astype(int)).pivot_table(index='ta', columns='year', values='value')
raw_nz['pop_growth'] = (pop[2023] / pop[2013] - 1).reindex(raw_nz.index) * 100
d = raw_nz.dropna(subset=['pop_growth'])
b, r2, _ = ols([d.pop_growth.values], d.eff.values); b2, r2b, _ = ols([np.log(d.dens.values), d.pop_growth.values], d.eff.values)
print(f'growth only R² {r2:.2f}; density + growth R² {r2b:.2f}')
ax = d.plot.scatter('pop_growth', 'eff', color=NZ, s=28); xs = np.linspace(d.pop_growth.min(), d.pop_growth.max(), 10); ax.plot(xs, b[0]*xs + b[1], '--', color=NZ)
for n in d.sort_values('eff').tail(5).index: ax.annotate(n, (d.pop_growth[n], d.eff[n]), fontsize=8, xytext=(4, 4), textcoords='offset points')
ax.set_xlabel('population growth 2013–2023 (%)'); ax.set_ylabel('years of wage per dwelling'); ax.set_title('NZ districts: growth predicts effort');""")
md("## 4. Mortgage simulator\nThe original's assumptions: 90% LTV, 4% interest, 45% of net pay (net ≈ 70% of gross), one earner. `inf` = the payment never covers the interest. Change the defaults below.")
code("""def years_to_repay(effort, earners=1, rate=0.04, ltv=0.9, dsti=0.45, net=0.7):
    L = ltv * effort; P = dsti * net * earners / 12; r = rate / 12
    return math.inf if r * L >= P else -math.log(1 - r * L / P) / math.log(1 + r) / 12

def cliff_table(df, weight, earners=2, rate=0.04, **kw):
    d = df.copy(); d['years'] = [years_to_repay(e, earners, rate, **kw) for e in d.eff]
    w = d[weight] / d[weight].sum()
    return pd.Series({'jobs share >15y': w[d.years > 15].sum(), 'jobs share >25y': w[d.years > 25].sum(), 'jobs share never': w[np.isinf(d.years)].sum()})

out = {}
for rate in [0.03, 0.04, 0.05, 0.06, 0.07]:
    out[('AU', rate)] = cliff_table(au, 'workers_pow_2021', rate=rate); out[('NZ', rate)] = cliff_table(nz, 'employees', rate=rate)
pd.DataFrame(out).T.round(2)""")
code("""caps = ['Greater Sydney','Greater Melbourne','Greater Brisbane','Greater Perth','Greater Adelaide','Canberra (ACT)']
nzc = ['Auckland','Wellington metro','Christchurch metro','Hamilton metro','Tauranga metro','Queenstown-Lakes']
pd.DataFrame({r: pd.concat([au.loc[caps,'eff'], nz.loc[nzc,'eff']]).map(lambda e: years_to_repay(e, 2, r)) for r in [0.04, 0.05, 0.06]}).round(1).rename(columns=lambda r: f'{r:.0%}, two earners')""")
md("""## 5. Time series, 1986–2025

* **AU capitals**: ABS median established-house price (Total Value of Dwellings, 2002–2025), back-cast to 1986 with the ABS pre-2005 Price Index of Established Homes, ÷ annualised AWE (all employees, persons, total earnings) for the capital's state from 1995. Before 1995, earnings follow the OECD household disposable income index (spliced, labelled).
* **NZ metros**: HUD 12-month median sale price to December (from 1980) ÷ annualised QES average weekly earnings, all employees (from 1989). 1986–88 earnings follow the OECD income index; 1980–85 have prices only.
* These use *average earnings of all employees*, not the median employee income used in the 2023 cross-section, so levels are not identical to the snapshot.""")
code("""au_ts = pd.read_csv(TS/'au_capital_effort_1986_2025.csv', index_col=0); nz_ts = pd.read_csv(TS/'nz_metro_effort_1986_2025.csv', index_col=0)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
au_ts.plot(ax=axes[0], lw=1.6); axes[0].axvspan(1986, 1995, color='grey', alpha=.12); axes[0].set_title('Australian capitals: median house ÷ average earnings')
nz_ts.plot(ax=axes[1], lw=1.6); axes[1].axvspan(1986, 1989, color='grey', alpha=.12); axes[1].set_title('NZ metros: median sale price ÷ average earnings')
for a in axes: a.set_ylabel('years of earnings'); a.set_xlabel(''); a.legend(fontsize=8, ncol=2)
plt.tight_layout(); print('shaded = earnings spliced on OECD income index')
pd.concat([au_ts.loc[[1986,1995,2005,2015,2020,2025]].T, nz_ts.loc[[1986,1995,2005,2015,2020,2025]].T]).round(1)""")
code("""nat = pd.read_csv(TS/'oecd_national.csv', index_col=0)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
nat[['AUS_pti','NZL_pti']].plot(ax=axes[0], color=[AU, NZ]); axes[0].set_title('OECD price-to-income ratio (2015 = 100)')
nat[['AUS_real_hpi','NZL_real_hpi']].plot(ax=axes[1], color=[AU, NZ]); axes[1].set_title('OECD real house price index (2015 = 100)')
plt.tight_layout()""")
md("### Capital premium\nIs the gap between the main city and the rest of the country widening? AU: capital median ÷ rest-of-state median (ABS). NZ: Auckland ÷ national median (HUD).")
code("""prem = pd.read_csv(TS/'au_capital_vs_rest_price_ratio.csv', index_col=0); akl = pd.read_csv(TS/'nz_auckland_price_premium.csv', index_col=0).iloc[:, 0]
ax = prem.plot(lw=1.5); akl.loc[2002:].plot(ax=ax, color='k', lw=2.2, label='Auckland ÷ NZ'); ax.legend(fontsize=8, ncol=2)
ax.set_title('Main-city price premium'); ax.set_ylabel('ratio of medians')
pd.concat([prem, akl.rename('Auckland/NZ')], axis=1).loc[[2005, 2010, 2015, 2017, 2020, 2022, 2025]].round(2)""")
md("## 6. Has the density relationship changed?\nNZ: re-estimate the district fit every year 2000–2024 (LEED earnings; density held at 2016 employee counts; industry weights fixed at 2016 before 2016). AU: labour-market regions 2020–2023 only, because ABS regional prices start in 2020.")
code("""f_nz = pd.read_csv(TS/'nz_district_fits_2000_2024.csv', index_col=0); f_au = pd.read_csv(TS/'au_region_fits_2020_2023.csv', index_col=0)
ax = f_nz[['r2_density','r2_growth10']].plot(marker='o', ms=3, color=[NZ, '#6250d6'])
f_au.r2_density.plot(ax=ax, marker='s', color=AU, label='AU r2_density'); ax.legend(['NZ: density', 'NZ: 10-yr population growth', 'AU: density'])
ax.set_ylim(0, .7); ax.set_title('R² of single-variable fits, by year'); f_nz.round(3).tail(8)""")
code("""panel = pd.read_csv(TS/'nz_district_effort_panel_2000_2024.csv')
wide = panel.pivot(index='ta', columns='year', values='eff')
chg = (wide[2024] - wide[2004]).sort_values()
print('Largest rises in effort, 2004 → 2024 (years of wage):'); print(chg.tail(10).round(1)); print('\\nSmallest:'); print(chg.head(5).round(1))""")
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import nb_extra; nb_extra.add(md, code)
md("""## 12. Open questions
* **Supply.** Add dwelling stock or building consents per worker (ABS building approvals by SA2; Stats NZ consents by TA) to test the mechanism instead of density as a proxy.
* **Dwelling mix.** Control for the house/unit share (AU) or floor area (NZ consents give new-build floor area) to separate composition from price.
* **NZ interest rates.** Section 12 does AU only; RBNZ table B20 (blocked to scripts) would extend it to NZ.
* **Workplace vs residence.** Australian wages are by residence (ATO). The Census WPP has income by place of work in bands, which could replace them.
* **Longer AU regional history.** ABS Data by Region only has regional prices from 2020; state Valuer-General sales files (NSW from 1990, Vic) could extend this.""")
code("""# Rebuild everything from raw/ (outputs written next to the inputs):
# uv run --with pandas --with openpyxl --with numpy python scripts/timeseries.py raw
# build_au.py / build_nz.py / analyse.py / robust.py take the same directory argument (they expect raw/au, raw/nz, raw/ts).""")
nb.cells=C; nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nbf.write(nb,sys.argv[1])
