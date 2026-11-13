# Extra notebook sections (maps + explorations). Imported by make_nb.py; md/code are its cell helpers.
def add(md, code):
    md("""## 7. Maps
Matplotlib versions of the page's choropleths, drawn straight from the boundary files in `raw/geo` (ABS ASGS 2021 SA4, generalised; Stats NZ TALB 2023, clipped to the coastline, Auckland local boards dissolved). No GIS library needed: each ring becomes a polygon in a `PolyCollection`.

Australia is mapped at **SA4** level (88 regions), finer than the 50 labour-market regions used in the fits, so inner-city SA4s show up separately. SA4 wages are **by residence** (ATO), which is why effort inside Sydney peaks in the Eastern Suburbs rather than the CBD.""")
    code("""import json, re, unicodedata
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib as mpl

def rings(g):
    return g['coordinates'] if g['type'] == 'Polygon' else [r for p in g['coordinates'] for r in p]
def norm_ta(x):
    x = unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode()
    x = re.sub(r'\\s+(District|City|Local Board Area|Local Board)$', '', x).strip()
    return {'Wanganui': 'Whanganui'}.get(x, x)
def ring_centroid(r):
    x = np.array([p[0] for p in r]); y = np.array([p[1] for p in r]); a = x*np.roll(y, -1) - np.roll(x, -1)*y; A = a.sum()/2
    return None if abs(A) < 1e-12 else (abs(A), ((x+np.roll(x, -1))*a).sum()/(6*A), ((y+np.roll(y, -1))*a).sum()/(6*A))
def centroid(rs):
    cs = [c for c in map(ring_centroid, rs) if c]; w = sum(c[0] for c in cs)
    return sum(c[0]*c[1] for c in cs)/w, sum(c[0]*c[2] for c in cs)/w
def km(lo1, la1, lo2, la2):
    p = np.pi/180; a = np.sin((la2-la1)*p/2)**2 + np.cos(la1*p)*np.cos(la2*p)*np.sin((lo2-lo1)*p/2)**2; return 12742*np.arcsin(np.sqrt(a))

# --- AU: SA4 geometry and SA4-level data
g = json.load(open(RAW/'geo'/'sa4_0.002.geojson'))
au_geo = {f['properties']['sa4_code_2021']: [r for r in rings(f['geometry']) if 112 < np.mean([p[0] for p in r]) < 154.2 and -44 < np.mean([p[1] for p in r]) < -9]
          for f in g['features'] if f['geometry']}
sa4 = pd.read_csv(RAW/'au'/'au_raw.csv', dtype={'sa4': str}).set_index('sa4')
sa4['price'] = (sa4.house_price*sa4.house_n + sa4.unit_price*sa4.unit_n) / (sa4.house_n + sa4.unit_n)
sa4['eff'] = sa4.price / sa4.wage; sa4['dens'] = sa4.workers_pow_2021 / sa4.area_km2
dbr = pd.read_csv(RAW/'au'/'dbr_sa4.csv', low_memory=False, dtype={'Code': str}); dbr = dbr[dbr.Code.str.fullmatch(r'\\d{3}')]
J = 'Jobs in Australia - year ended 30 June :: Number of employee jobs - total'
jobs = dbr.assign(v=pd.to_numeric(dbr[J].astype(str).str.replace(',', ''), errors='coerce')).pivot_table(index='Code', columns='Year', values='v')
sa4['jobs_g'] = (jobs[2023]/jobs[2018] - 1)*100; nat_au_g = (jobs[2023].sum()/jobs[2018].sum() - 1)*100
for c, rs in au_geo.items():
    if c in sa4.index and rs: sa4.loc[c, ['lon', 'lat']] = centroid(rs)

# --- NZ: district geometry (Auckland local boards dissolved) and district-level data
t = json.load(open(RAW/'geo'/'talb.geojson')); nz_geo, land = {}, {}
for f in t['features']:
    p = f['properties']; ta = 'Auckland' if p['TALB2023_V1_00'].startswith('076') else norm_ta(p['TALB2023_V1_00_NAME_ASCII'])
    if ta in ('Chatham Islands', 'Area Outside Territorial Authority'): continue
    nz_geo.setdefault(ta, []).extend(rings(f['geometry'])); land[ta] = land.get(ta, 0) + p['LAND_AREA_SQ_KM']
ta_ = pd.read_csv(RAW/'nz'/'nz_raw.csv', index_col=0).drop(index=['Chatham Islands'])
ta_['eff'] = ta_.median_sale_price / ta_.wage; ta_['dens'] = ta_.employees / pd.Series(land)
bd = pd.read_csv(RAW/'nz'/'fnz_bd.csv'); bd = bd[(bd.Measure == 'Employee count') & (bd.Industry == 'Total all industries')]
emp = bd.assign(ta=bd['Territorial authority'].map(norm_ta)).pivot_table(index='ta', columns='Year as at February', values='Value', aggfunc='sum')
ta_['emp_g'] = (emp[2025]/emp[2016] - 1)*100; nat_nz_g = (emp.loc['New Zealand', 2025]/emp.loc['New Zealand', 2016] - 1)*100
nzw = pd.read_csv(TS/'nz_district_effort_panel_2000_2024.csv').pivot(index='ta', columns='year', values='eff')
ta_['eff_chg'] = nzw[2024] - nzw[2004]
for k, rs in nz_geo.items(): ta_.loc[k, ['lon', 'lat']] = centroid(rs)
print(f'{len(au_geo)} SA4 shapes, {len(nz_geo)} NZ districts; national job growth AU {nat_au_g:.1f}% (2018-23), NZ {nat_nz_g:.1f}% (2016-25)')""")
    code("""DIV = ['#2166ac', '#67a9cf', '#d1e5f0', '#e0e0e0', '#fddbc7', '#ef8a62', '#b2182b']   # blue = sparse/cheap, red = dense/dear
SEQ = ['#fff5eb', '#fdd0a2', '#fdae6b', '#fd8d3c', '#e6550d', '#a63603', '#7f2704']

def choropleth(ax, geo, values, breaks, colors=DIV, bbox=None, lw=.25):
    \"\"\"geo: {key: [rings]}, values: Series keyed the same. bbox = (lon0, lat0, lon1, lat1) to zoom.\"\"\"
    cmap, nrm = ListedColormap(colors), BoundaryNorm([-1e12] + list(breaks) + [1e12], len(colors))
    lat0 = np.mean([bbox[1], bbox[3]]) if bbox else np.mean([p[1] for rs in geo.values() for r in rs[:1] for p in r[:1]])
    k = np.cos(np.radians(lat0)); polys, cols = [], []
    for key, rs in geo.items():
        v = values.get(key, np.nan); c = '#f4f4f4' if pd.isna(v) else cmap(nrm(v))
        for r in rs:
            if bbox and not any(bbox[0]-.3 < p[0] < bbox[2]+.3 and bbox[1]-.3 < p[1] < bbox[3]+.3 for p in r): continue
            polys.append([(p[0]*k, p[1]) for p in r]); cols.append(c)
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors='white', linewidths=lw))
    if bbox: ax.set_xlim(bbox[0]*k, bbox[2]*k); ax.set_ylim(bbox[1], bbox[3])
    else: ax.autoscale_view()
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    return cmap, nrm

def legend_bar(fig, ax, cmap, nrm, breaks, label, fmt='{:g}'):
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=BoundaryNorm(range(len(breaks)+2), len(breaks)+1), cmap=cmap), ax=ax, orientation='horizontal', fraction=.04, pad=.02)
    cb.set_ticks(np.arange(1, len(breaks)+1)); cb.set_ticklabels([fmt.format(b) for b in breaks]); cb.set_label(label, fontsize=9); cb.ax.tick_params(labelsize=8)

def inset(ax, geo, values, breaks, bbox, title, pos, colors=DIV):
    a = ax.inset_axes(pos); choropleth(a, geo, values, breaks, colors, bbox=bbox, lw=.4)
    for s in a.spines.values(): s.set_visible(True); s.set_color('#888'); s.set_linewidth(.6)
    a.set_title(title, fontsize=8, pad=2)

AU_BOX = (112.8, -43.8, 153.8, -10.3)
SYD, MEL, BNE, PER = (150.35, -34.25, 151.4, -33.45), (144.55, -38.25, 145.55, -37.55), (152.65, -27.75, 153.25, -27.2), (115.65, -32.3, 116.15, -31.65)
EB, DB, GB = [7, 9, 10.5, 12, 14, 17], [0.5, 2, 8, 30, 120, 500], [-15, -7, -2, 2, 7, 15]

def au_insets(ax, vals, br, cl=DIV):   # inset column to the right of the map, so the coast stays visible
    for bb, nm, y in [(SYD, 'Sydney', .68), (MEL, 'Melbourne', .35), (BNE, 'Brisbane', .02)]: inset(ax, au_geo, vals, br, bb, nm, [1.02, y, .3, .3], cl)

def au_pair(cols, breaks, labels, fmts, titles, colors=(DIV, DIV)):
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={'wspace': .42})
    for ax, col, br, lab, fm, ti, cl in zip(axes, cols, breaks, labels, fmts, titles, colors):
        cm, nm = choropleth(ax, au_geo, sa4[col], br, cl, bbox=AU_BOX); ax.set_title(ti, fontsize=11)
        au_insets(ax, sa4[col], br, cl); legend_bar(fig, ax, cm, nm, br, lab, fm)

au_pair(['dens', 'eff'], [DB, EB], ['jobs per km² (by workplace, 2021)', 'years of median wage per median dwelling (2023)'], ['{:g}', '{:g}'],
        ['Australia: job density by SA4', 'Australia: effort by SA4'])""")
    md("The two Australian maps line up along the east coast (Sydney to the Sunshine Coast) and in inner Melbourne; the interior and the mining regions are blue on both. That is the visual version of the R² 0.42 fit.")
    code("""NZ_BOX = (166.3, -47.4, 178.7, -34.3); WLG, AKL = (174.6, -41.42, 175.25, -40.95), (174.3, -38.2, 175.9, -36.6)
def nz_trio(cols, breaks, labels, titles, colors, fmts):
    fig, axes = plt.subplots(1, len(cols), figsize=(5.4*len(cols), 7.5))
    for ax, col, br, lab, ti, cl, fm in zip(axes, cols, breaks, labels, titles, colors, fmts):
        cm, nm = choropleth(ax, nz_geo, ta_[col], br, cl, bbox=NZ_BOX); ax.set_title(ti, fontsize=11)
        inset(ax, nz_geo, ta_[col], br, AKL, 'Auckland & Waikato', [.0, .58, .38, .4], cl); inset(ax, nz_geo, ta_[col], br, WLG, 'Wellington', [.66, .3, .34, .3], cl)
        legend_bar(fig, ax, cm, nm, br, lab, fm)
    plt.tight_layout()
ta_['emp_rel'] = ta_.emp_g - nat_nz_g
nz_trio(['dens', 'eff', 'eff_chg'], [DB, EB, [1, 2, 3, 4, 5, 6]], ['jobs per km² (2023)', 'years of wage per dwelling (2023)', 'change in effort, years (2004→2024)'],
        ['NZ: job density', 'NZ: effort', 'NZ: rise in effort 2004–2024'], [DIV, DIV, SEQ], ['{:g}', '{:g}', '+{:g}'])""")
    md("NZ density and effort do **not** line up: Coromandel, Queenstown, Tasman and Central Otago are as red as Auckland on effort without being dense. The third map shows where effort *rose*: a belt around Auckland and Hamilton, plus Queenstown and Central Otago.")
    code("""fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={'wspace': .42})
sa4['jobs_rel'] = sa4.jobs_g - nat_au_g
cm, nm = choropleth(axes[0], au_geo, sa4.jobs_rel, GB, bbox=AU_BOX); axes[0].set_title(f'Australia: employee-job growth 2018–23 vs national (+{nat_au_g:.1f}%)', fontsize=11)
au_insets(axes[0], sa4.jobs_rel, GB)
legend_bar(fig, axes[0], cm, nm, GB, 'percentage points vs national', '{:+g}')
cm, nm = choropleth(axes[1], nz_geo, ta_.emp_rel, GB, bbox=NZ_BOX); axes[1].set_title(f'NZ: employee growth 2016–25 vs national (+{nat_nz_g:.1f}%)', fontsize=11)
inset(axes[1], nz_geo, ta_.emp_rel, GB, AKL, 'Auckland & Waikato', [.0, .58, .38, .4])
legend_bar(fig, axes[1], cm, nm, GB, 'percentage points vs national', '{:+g}')
print('AU fastest:', sa4.nlargest(5, 'jobs_g')[['name', 'jobs_g']].round(1).values.tolist()); print('NZ fastest:', ta_.emp_g.nlargest(5).round(1).to_dict())""")
    md("Caveat: the ABS *Jobs in Australia* series counts jobs by where the **worker lives**, so the AU growth map shows where workers settle (outer suburbs), not where jobs are created.")

    md("""## 8. Exploration: where is housing dearer than density predicts?
Fit effort on ln(density) at the mapped level (AU SA4, NZ district), then map the residual. Red = dearer than its density implies (amenity, coast, tourism, supply limits); blue = cheaper.""")
    code("""def resid(df, x='dens', y='eff'):
    d = df.dropna(subset=[x, y]); b, r2, res = ols([np.log(d[x].values)], d[y].values); return pd.Series(res, d.index), r2
ra, r2a = resid(sa4); rn, r2n = resid(ta_)
RB = [-4, -2, -0.75, 0.75, 2, 4]
fig, axes = plt.subplots(1, 2, figsize=(17, 6.8), gridspec_kw={'wspace': .42})
cm, nm = choropleth(axes[0], au_geo, ra, RB, bbox=AU_BOX); axes[0].set_title(f'Australia SA4: effort minus density prediction (fit R² {r2a:.2f})', fontsize=11)
au_insets(axes[0], ra, RB)
legend_bar(fig, axes[0], cm, nm, RB, 'years of wage above (+) / below (−) prediction', '{:+g}')
cm, nm = choropleth(axes[1], nz_geo, rn, RB, bbox=NZ_BOX); axes[1].set_title(f'NZ district: effort minus density prediction (fit R² {r2n:.2f})', fontsize=11)
inset(axes[1], nz_geo, rn, RB, AKL, 'Auckland & Waikato', [.0, .58, .38, .4]); legend_bar(fig, axes[1], cm, nm, RB, 'years above / below prediction', '{:+g}')
pd.DataFrame({'AU dearer than predicted': ra.nlargest(8).rename(sa4.name).round(1).items(), 'AU cheaper': ra.nsmallest(8).rename(sa4.name).round(1).items(),
              'NZ dearer': rn.nlargest(8).round(1).items(), 'NZ cheaper': rn.nsmallest(8).round(1).items()})""")
    md("""* **Australia, dearer than density predicts:** Sydney's Northern Beaches (+11.2 years) and Eastern Suburbs (+10.8), the Hills, the NSW North Coast (Richmond–Tweed, Coffs Harbour) and the Southern Highlands. The common factors are coast, amenity and high residential incomes.
* **Australia, cheaper:** all of suburban Perth, inner Melbourne and inner Brisbane (dense job centres with plenty of apartments), and Darwin.
* **NZ, dearer:** Queenstown (+15.4), Coromandel, Western Bay of Plenty, Tasman and Central Otago. These are lifestyle and retirement districts, not job centres. **NZ, cheaper:** Kawerau, Grey, Invercargill, Buller and Gore, which are small industrial or mining towns.

The AU residual map is close to a map of the coast. That suggests the "tourism" variable in the AU two-variable model is partly measuring coastal amenity.""")
    md("""## 9. Exploration: the bid-rent curve — does effort fall with distance from the CBD?
Classic urban economics says land is dearest at the centre. Each point is an SA4 within 200 km of its state capital's CBD (distance from the SA4's area-weighted centroid, so large outer SA4s sit further out than their populated edge).""")
    code("""CBD = {'NSW': ('Sydney', 151.207, -33.868), 'VIC': ('Melbourne', 144.963, -37.814), 'QLD': ('Brisbane', 153.026, -27.469), 'SA': ('Adelaide', 138.600, -34.929), 'WA': ('Perth', 115.860, -31.952)}
sa4['km_cbd'] = [km(CBD[s][1], CBD[s][2], lo, la) if s in CBD else np.nan for s, lo, la in zip(sa4.state, sa4.lon, sa4.lat)]
near = sa4[sa4.km_cbd < 200]
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True); rows = {}
for ax, (st, (city, *_)) in zip(axes, CBD.items()):
    d = near[near.state == st]; b = np.polyfit(d.km_cbd, d.eff, 1)
    ax.scatter(d.km_cbd, d.eff, color=AU, s=30, edgecolor='white', zorder=3); xs = np.array([0, d.km_cbd.max()]); ax.plot(xs, b[0]*xs + b[1], '--', color=AU)
    for n in d.nlargest(2, 'eff').index.tolist() + d.nsmallest(1, 'km_cbd').index.tolist():
        ax.annotate(d.name[n].replace(city+' - ', '').replace('Sydney - ', ''), (d.km_cbd[n], d.eff[n]), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.set_title(city, fontsize=10); ax.set_xlabel('km from CBD')
    rows[city] = {'SA4s': len(d), 'effort change per 10 km': b[0]*10, 'corr(effort, km)': np.corrcoef(d.km_cbd, d.eff)[0, 1],
                  'corr(ln price, km)': np.corrcoef(d.km_cbd, np.log(d.price))[0, 1], 'corr(ln wage, km)': np.corrcoef(d.km_cbd, np.log(d.wage))[0, 1],
                  'peak SA4': d.name[d.eff.idxmax()], 'peak km': d.km_cbd[d.eff.idxmax()]}
axes[0].set_ylabel('years of wage'); plt.tight_layout(); pd.DataFrame(rows).T.round(2)""")
    md("""What to look for: prices fall with distance everywhere, but so do wages (measured by residence), so effort falls more gently than price.
* **Sydney** loses about 0.5 years of wage per 10 km, but the peak is not the CBD SA4 (15.8 years). It is the Eastern Suburbs (25.9) and the Northern Beaches (25.0, about 20 km out): coast and amenity beat proximity.
* **Brisbane** has no gradient (corr ≈ 0.07). The Sunshine Coast (15.9) and Gold Coast (14.3), 60–100 km out, are dearer than inner Brisbane (11.6). This is the Australian version of NZ's lifestyle-district pattern.
* **Perth** and **Adelaide** follow the textbook curve most closely, with wages falling almost in step with prices.""")
    md("""## 10. Exploration: have NZ districts converged?
Two standard tests from growth economics, applied to the 2000–2024 district panel:
* **β-convergence**: did districts that started cheap see effort rise faster?
* **σ-convergence**: is the spread (coefficient of variation) across districts shrinking?
And a spatial one: is effort rising fastest *near* Auckland (spillover from the main labour market)?""")
    code("""cv = nzw.std() / nzw.mean(); g = np.log(nzw[2024] / nzw[2004]); b = np.polyfit(np.log(nzw[2004]), g, 1)
ta_['km_akl'] = [km(174.763, -36.848, lo, la) for lo, la in zip(ta_.lon, ta_.lat)]
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
axes[0].scatter(nzw[2004], g*100, color=NZ, s=26, edgecolor='white'); xs = np.linspace(nzw[2004].min(), nzw[2004].max(), 20); axes[0].plot(xs, (b[0]*np.log(xs) + b[1])*100, '--', color=NZ)
for n in ['Queenstown-Lakes', 'Auckland', 'Kawerau', 'Waikato', 'Buller']:
    if n in nzw.index: axes[0].annotate(n, (nzw.at[n, 2004], g[n]*100), fontsize=8, xytext=(3, 3), textcoords='offset points')
axes[0].set_xlabel('effort in 2004 (years)'); axes[0].set_ylabel('growth in effort 2004→2024 (%, log)'); axes[0].set_title(f'β-convergence: corr {np.corrcoef(np.log(nzw[2004]), g)[0, 1]:.2f}', fontsize=10)
cv.plot(ax=axes[1], color=NZ, marker='o', ms=3); axes[1].set_title('σ-convergence: CV of effort across districts', fontsize=10); axes[1].set_xlabel('')
d = ta_.dropna(subset=['eff_chg', 'km_akl']); bands = pd.cut(d.km_akl, [0, 50, 150, 300, 600, 1200], labels=['<50 km', '50–150', '150–300', '300–600', '600+'])
s = d.groupby(bands, observed=True).eff_chg.agg(['median', 'count']); axes[2].bar(s.index.astype(str), s['median'], color=NZ, width=.6)
for i, (m, c) in enumerate(zip(s['median'], s['count'])): axes[2].text(i, m + .08, f'{m:.1f}\\n(n={c})', ha='center', fontsize=8)
axes[2].set_title('Median rise in effort 2004→2024, by distance from Auckland CBD', fontsize=10); axes[2].set_ylabel('years of wage'); plt.tight_layout()
print('Spearman rank corr of effort 2004 vs 2024:', round(nzw[2004].rank().corr(nzw[2024].rank()), 2))
print('CV:', cv.loc[[2000, 2004, 2008, 2016, 2021, 2024]].round(2).to_dict()); print(s)""")
    md("""* **β-convergence: strong.** Districts that started cheap saw the largest percentage rises (corr ≈ −0.89 between the 2004 level and 2004→24 log growth).
* **σ-convergence: since 2016.** The spread across districts fell from 0.45 to 0.31 (CV). But ratios converging still means dollar gaps widened.
* **The ranking barely moved.** The rank correlation between 2004 and 2024 is 0.92. Queenstown, Coromandel and Tauranga/Western Bay were at the top then and are now.
* **An Auckland halo.** Districts 50–150 km from Auckland's CBD rose most (median +5.4 years; Waikato +6.9, Kaipara +6.6, Waipa and Matamata-Piako +6.1). Districts more than 600 km away rose least (+3.4). This fits buyers priced out of Auckland moving into the commuter belt, but it is a correlation, not a test of migration.""")
    md("""## 11. Exploration: the repayment burden with the interest rates people actually faced (Australia)
Section 4 holds the rate at 4%. Here each year uses the **RBA standard variable owner-occupier rate** (table F5, annual average) and that year's capital-city effort from section 5 (median house ÷ average earnings).
The measure: the monthly repayment on a **30-year loan for 80% of the median house**, as a share of **two average earners' gross pay**. This is a standard affordability measure, used here because the original's never-repay rule (90% LTV, 45% of net pay) puts Sydney at "never" in almost every year since 1986 at the rates of the day, which tells you little.
From about 2004 most borrowers paid less than the standard rate, so the dashed lines use the RBA *discounted* variable rate where it exists.
NZ is not shown: the RBNZ mortgage-rate tables (B20/B3) block scripted downloads. Save `hb20.xlsx` from rbnz.govt.nz into `raw/rates/` to extend this.""")
    code("""f5 = pd.read_excel(RAW/'rates'/'rba_f05hist.xlsx', header=None, skiprows=11).set_index(0)
svr = f5[3].astype(float).groupby(f5.index.year).mean().rename('standard variable'); disc = f5[4].astype(float).groupby(f5.index.year).mean().rename('discounted variable')
rates = pd.concat([svr, disc], axis=1).loc[1986:]

def burden(effort, rate, lvr=0.8, years=30, earners=2):
    \"\"\"Annual repayment as a share of household gross pay. effort is in years of one earner's pay.\"\"\"
    r, n = rate/12, years*12
    return lvr * effort * r / (1 - (1 + r)**-n) * 12 / earners

caps4 = ['Sydney', 'Melbourne', 'Brisbane', 'Perth']
fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
rates.plot(ax=axes[0], color=['k', 'grey'], style=['-', '--']); axes[0].set_title('RBA housing loan rates (% p.a., annual average)', fontsize=10)
B = pd.DataFrame({c: [burden(au_ts.at[y, c], rates.at[y, 'standard variable']/100) for y in rates.index if y in au_ts.index] for c in caps4},
                 index=[y for y in rates.index if y in au_ts.index])
Bd = pd.DataFrame({c: [burden(au_ts.at[y, c], rates.at[y, 'discounted variable']/100) for y in B.index] for c in caps4}, index=B.index)
for c in caps4:
    l, = axes[1].plot(B.index, B[c]*100, lw=1.6, label=c); axes[1].plot(Bd.index, Bd[c]*100, '--', lw=1, color=l.get_color())
axes[1].axhline(30, color='grey', lw=.8, ls=':'); axes[1].text(1987, 31, '30% "stress" line', fontsize=8, color='grey')
axes[1].set_title('Repayment on 80% of the median house, % of gross pay of two average earners', fontsize=10); axes[1].legend(fontsize=8); axes[1].set_ylabel('%')
plt.tight_layout()
yrs = [y for y in [1986, 1989, 1995, 2000, 2008, 2016, 2021, 2023, 2025] if y in B.index]
pd.concat({'SVR %': rates.loc[yrs, 'standard variable'], 'discounted %': rates.loc[yrs, 'discounted variable'], 'Sydney effort': au_ts.loc[yrs, 'Sydney'],
           **{f'{c} burden %': B.loc[yrs, c]*100 for c in caps4}, 'Sydney burden at discounted %': Bd.loc[yrs, 'Sydney']*100}, axis=1).round(1)""")
    md("""**Reading it.**
* At the standard variable rate, a Sydney household needed about **66% of two average earners' gross pay** in 2023–25, against **53% at the 1989 peak**, when rates were 16.5%. At the discounted rate most borrowers actually got, it was 54–57%: still above 1989.
* The 2021 low in rates (4.5%) hid an effort ratio more than twice the 1989 level. Once rates went back above 8%, the high price level and the high rate arrived together.
* **Brisbane** is at a record (46% in 2025, against 30% in 1989). **Perth** (37%) is level with its 1989 and 2008 peaks. **Melbourne** (40%) is just below its 2023–24 high and above 1989 (36%).
* Caveats: average earnings, not household income. No allowance for the second earner's lower pay, tax changes, or the smaller deposits common in the 1980s. Before 1995 earnings follow the OECD income index (section 5).""")
