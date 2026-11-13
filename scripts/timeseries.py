"""Build AU/NZ housing-effort time series (1980s-2025). Usage: python timeseries.py <scratch_dir>"""
import pandas as pd, numpy as np, json, sys, re, unicodedata
S=sys.argv[1]; T=f'{S}/ts'
def abs_sheet(path):
    d=pd.read_excel(path,sheet_name='Data1',header=None)
    cols=[str(c).strip() for c in d.iloc[0]]; x=d.iloc[10:].copy(); x.columns=['date']+cols[1:]
    x['date']=pd.to_datetime(x['date']); x=x.set_index('date').apply(pd.to_numeric,errors='coerce'); return x
def annual(x): return x.groupby(x.index.year).mean()
CAP=['Sydney','Melbourne','Brisbane','Adelaide','Perth','Hobart','Darwin','Canberra']
STATE={'Sydney':'A','Melbourne':'B','Brisbane':'C','Adelaide':'D','Perth':'E','Hobart':'F','Darwin':'G','Canberra':'H'}
# ---- OECD national ----
o=pd.read_csv(f'{T}/oecd_hp.csv')
o=o.pivot_table(index='TIME_PERIOD',columns=['REF_AREA','MEASURE'],values='OBS_VALUE')
oecd={c:dict(pti=o[(c,'HPI_YDH')].dropna() if (c,'HPI_YDH') in o else None) for c in ['AUS','NZL']}
meas=sorted(set(m for _,m in o.columns)); print('OECD measures',meas,file=sys.stderr)
def ser(c,m): return o[(c,m)].dropna()
# income index implied by nominal HPI / price-to-income
inc_idx={c:(ser(c,'HPI')/ser(c,'HPI_YDH')).dropna() for c in ['AUS','NZL']}
# ---- AU prices ----
tvd=annual(abs_sheet(f'{T}/tvd_643202.xlsx'))*1000  # ABS medians are in $'000
old=annual(abs_sheet(f'{T}/d21_641608.xlsx'))
au_price={}; au_unit={}
for c in CAP:
    hc=[k for k in tvd.columns if k.startswith('Median Price of Established House') and f'; {c} ;' in k.replace('  ',' ')]
    uc=[k for k in tvd.columns if k.startswith('Median Price of Attached') and f'; {c} ;' in k.replace('  ',' ')]
    h=tvd[hc[0]].dropna(); oc=[k for k in old.columns if f'; {c} ;' in k.replace('  ',' ')][0]; idx=old[oc].dropna()
    back=h.loc[2002]*idx/idx.loc[2002]; s=pd.concat([back[back.index<2002],h]); au_price[c]=s
    au_unit[c]=tvd[uc[0]].dropna()
rest=[k for k in tvd.columns if k.startswith('Median Price of Established House') and 'Rest of' in k]
au_rest={re.search(r'Rest of (\w+)',k).group(1):tvd[k].dropna() for k in rest}
# ---- AU earnings: AWE persons total earnings (all employees) by state, annualised; pre-1995 spliced on OECD income index
au_wage={}
for c,s in STATE.items():
    a=abs_sheet(f'{T}/awe_63020013{s}.xlsx'); col=[k for k in a.columns if k.startswith('Earnings; Persons; Total earnings')][0]
    w=annual(a[col].dropna())*52
    ii=inc_idx['AUS']; first=w.index.min()+1  # 1994 has only Nov; use 1995 as anchor
    w=w[w.index>=first]; back=w.loc[first]*ii/ii.loc[first]; au_wage[c]=pd.concat([back[(back.index<first)&(back.index>=1986)],w])
au=pd.DataFrame({c:au_price[c]/au_wage[c] for c in CAP}).loc[1986:2025]
au_units=pd.DataFrame({c:au_unit[c]/au_wage[c] for c in CAP}).loc[2002:2025]
# capital vs rest of state premium
prem=pd.DataFrame({k:au_price[c]/au_rest[k] for c,k in [('Sydney','NSW'),('Melbourne','Vic'),('Brisbane','Qld'),('Adelaide','SA'),('Perth','WA'),('Hobart','Tas')] if k in au_rest}).loc[2002:2025]
print('rest keys',list(au_rest),file=sys.stderr)
# ---- NZ ----
def norm(x):
    x=unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode(); x=re.sub(r'\s+(District|City)$','',x).strip()
    return {'Wanganui':'Whanganui','Southland district':'Southland','Waikato district':'Waikato','Wellington city':'Wellington'}.get(x,x)
h=pd.read_csv(f'{S}/nz/hud_sales.csv'); h=h[(h.stat_type=='12-Month rolling')&(h.period.str[5:7]=='12')]
h['year']=h.period.str[:4].astype(int); h['ta']=h.area.map(norm)
nzp=h[h.area_type=='Territorial Authority'].pivot_table(index='year',columns='ta',values='median_sale_price')
nzn=h[h.area_type=='National'].set_index('year').median_sale_price
nzs=h[h.area_type=='Territorial Authority'].pivot_table(index='year',columns='ta',values='residential_sales')
q=pd.read_csv(f'{T}/nz_qes.csv'); q=q[(q['Industry Code']=='TOTAL')&(q.Sex=='Total')&(q['Time paid'].str.startswith('Total'))]
q['year']=q.Quarter.str[:4].astype(int); qw=q.groupby('year').Value.mean()*52
qw=qw[qw.index<=2025]; ii=inc_idx['NZL']; back=qw.loc[1989]*ii/ii.loc[1989]; nzw=pd.concat([back[(back.index<1989)],qw])
MET={'Auckland':['Auckland'],'Wellington metro':['Wellington','Lower Hutt','Upper Hutt','Porirua'],'Christchurch metro':['Christchurch','Selwyn','Waimakariri'],
     'Hamilton metro':['Hamilton','Waipa','Waikato'],'Tauranga metro':['Tauranga','Western Bay of Plenty'],'Dunedin':['Dunedin'],'Queenstown-Lakes':['Queenstown-Lakes']}
nzm=pd.DataFrame({g:(nzp[cs]*nzs[cs]).sum(1,min_count=len(cs))/nzs[cs].sum(1,min_count=len(cs)) for g,cs in MET.items()})
nz=nzm.div(nzw,axis=0).loc[1986:2025]; nz['New Zealand']=(nzn/nzw).loc[1986:2025]
nz_price=nzm.copy(); nz_price['New Zealand']=nzn
akl_prem=(nzm['Auckland']/nzn).loc[1980:2025]
# ---- NZ district panel 2000-2024: LEED earnings by TA, density fit per year ----
r=pd.read_csv(f'{S}/nz/rear/all-data.csv',dtype=str,low_memory=False)
REG={'Northland','Bay of Plenty',"Hawke's Bay",'Manawatu-Wanganui','Taranaki','Canterbury','Otago','West Coast','New Zealand'}
e=r[(r.indicator=='earnings by industry')&(r['transform']=='original')&(~r.area.isin(REG))].copy(); e['ta']=e.area.map(norm); e['value']=e.value.astype(float); e['year']=e.year.astype(int); e['feature']=e.feature.str.lower()
b=pd.read_csv(f'{S}/nz/fnz_bd.csv'); b=b[(b.Measure=='Employee count')&(b['Industry Code'].str.len()==1)]; b['ta']=b['Territorial authority'].map(norm); b['feature']=b.Industry.str.lower()
wts=b.pivot_table(index=['Year as at February','ta'],columns='feature',values='Value',aggfunc='sum')
tot=pd.read_csv(f'{S}/nz/fnz_bd.csv'); tot=tot[(tot.Measure=='Employee count')&(tot.Industry=='Total all industries')]; tot['ta']=tot['Territorial authority'].map(norm)
emp=tot.pivot_table(index='ta',columns='Year as at February',values='Value',aggfunc='sum')
pop=r[(r.indicator=='population estimates')&(r['transform']=='original')]; pop=pop.assign(ta=pop.area.map(norm),value=pop.value.astype(float),year=pop.year.astype(int)).pivot_table(index='ta',columns='year',values='value')
area=pd.read_csv(f'{S}/nz/nz_raw.csv',index_col=0).area_km2
dens16=(emp[2016]/area).dropna()
def fit(x,y):
    X=np.column_stack([x,np.ones(len(x))]); bb,*_=np.linalg.lstsq(X,y,rcond=None); return float(1-((y-X@bb)**2).sum()/((y-y.mean())**2).sum()),float(bb[0])
panel=[]; fits=[]
for y in range(2000,2025):
    ey=e[e.year==y].pivot_table(index='ta',columns='feature',values='value')
    wy=wts.loc[max(y,2016)] if max(y,2016) in wts.index.get_level_values(0) else wts.loc[2016]
    com=[c for c in ey.columns if c in wy.columns]; v=ey[com]; w=wy[com].reindex(v.index); mk=v.notna()&w.notna()
    wage=(v.where(mk)*w.where(mk)).sum(axis=1)/w.where(mk).sum(axis=1)
    pr=nzp.loc[y] if y in nzp.index else None
    df=pd.DataFrame({'wage':wage,'price':pr}).dropna(); df=df[df.index.isin(dens16.index)]; df['eff']=df.price/df.wage
    df['ld']=np.log(dens16[df.index])
    r2d,sl=fit(df.ld.values,df.eff.values)
    row=dict(year=y,n=len(df),r2_density=r2d,slope_density=sl,mean_eff=float(df.eff.mean()),cv_eff=float(df.eff.std()/df.eff.mean()))
    if y-10 in pop.columns:
        g=(pop[y]/pop[y-10]-1).reindex(df.index); ok=g.notna(); row['r2_growth10']=fit(g[ok].values,df.eff[ok].values)[0]
    fits.append(row)
    for ta,vv in df.iterrows(): panel.append(dict(year=y,ta=ta,price=vv.price,wage=vv.wage,eff=vv.eff))
fits=pd.DataFrame(fits).set_index('year'); panel=pd.DataFrame(panel)
# ---- AU labour-market panel 2020-2023 ----
dbr=pd.read_csv(f'{S}/au/dbr_sa4.csv',low_memory=False,dtype={'Code':str})
P='Residential property transfers - year ended 30 June :: '
cols=dict(hp=P+'Median price of established house transfers ($)',hn=P+'Number of established house transfers (no.)',up=P+'Median price of attached dwelling transfers ($)',un=P+'Number of attached dwelling transfers (no.)',w='Personal income in Australia - year ended 30 June :: Median employee income ($)')
for k,c in cols.items(): dbr[k]=pd.to_numeric(dbr[c].astype(str).str.replace(',',''),errors='coerce')
lm=pd.read_csv(f'{S}/au_labour_market_regions.csv',index_col=0)
G={'Greater Sydney':['102']+[str(i) for i in range(115,129)],'Greater Melbourne':[str(i) for i in range(206,215)],'Greater Brisbane':['301','302','303','304','305','310','311','313','314'],'Greater Adelaide':['401','402','403','404'],'Greater Perth':[str(i) for i in range(502,508)],'Greater Hobart':['601'],'Greater Darwin':['701'],'Canberra (ACT)':['801']}
GC={'Greater Sydney':'1GSYD','Greater Melbourne':'2GMEL','Greater Brisbane':'3GBRI','Greater Adelaide':'4GADE','Greater Perth':'5GPER','Greater Hobart':'601','Greater Darwin':'701','Canberra (ACT)':'801'}
inG={c for cs in G.values() for c in cs}
sa4=dbr[dbr.Code.str.fullmatch(r'\d{3}')&~dbr.Code.isin(inG)&(dbr.Code.str[0]!='9')]
au_fits=[]; au_panel=[]
for y in [2020,2021,2022,2023]:
    rows=[]
    for g,code in GC.items():
        q_=dbr[(dbr.Code==code)&(dbr.Year==y)].iloc[0]; rows.append((g,q_))
    for _,q_ in sa4[sa4.Year==y].iterrows(): rows.append((q_.Label,q_))
    df=pd.DataFrame([dict(name=n,price=(q_.hp*q_.hn+q_.up*q_.un)/(q_.hn+q_.un),wage=q_.w) for n,q_ in rows]).set_index('name')
    df=df.join(lm[['dens']]).dropna(); df['eff']=df.price/df.wage
    r2d,sl=fit(np.log(df.dens.values),df.eff.values)
    au_fits.append(dict(year=y,n=len(df),r2_density=r2d,slope_density=sl,mean_eff=float(df.eff.mean())))
    for n,vv in df.iterrows(): au_panel.append(dict(year=y,region=n,price=vv.price,wage=vv.wage,eff=vv.eff))
au_fits=pd.DataFrame(au_fits).set_index('year'); au_panel=pd.DataFrame(au_panel)
# ---- save ----
out=f'{S}/ts_out'; import os; os.makedirs(out,exist_ok=True)
au.to_csv(f'{out}/au_capital_effort_1986_2025.csv'); au_units.to_csv(f'{out}/au_capital_unit_effort_2002_2025.csv'); prem.to_csv(f'{out}/au_capital_vs_rest_price_ratio.csv')
pd.DataFrame(au_price).to_csv(f'{out}/au_capital_median_house_price.csv'); pd.DataFrame(au_wage).to_csv(f'{out}/au_state_awe_annual.csv')
nz.to_csv(f'{out}/nz_metro_effort_1986_2025.csv'); nz_price.to_csv(f'{out}/nz_metro_median_price_1980_2025.csv'); nzw.to_csv(f'{out}/nz_qes_awe_annual.csv'); akl_prem.to_csv(f'{out}/nz_auckland_price_premium.csv')
fits.to_csv(f'{out}/nz_district_fits_2000_2024.csv'); panel.to_csv(f'{out}/nz_district_effort_panel_2000_2024.csv',index=False)
au_fits.to_csv(f'{out}/au_region_fits_2020_2023.csv'); au_panel.to_csv(f'{out}/au_region_effort_panel_2020_2023.csv',index=False)
nat=pd.DataFrame({'AUS_pti':ser('AUS','HPI_YDH'),'NZL_pti':ser('NZL','HPI_YDH'),'AUS_real_hpi':ser('AUS','RHP'),'NZL_real_hpi':ser('NZL','RHP')})
nat.to_csv(f'{out}/oecd_national.csv')
pd.set_option('display.width',200)
print(au.loc[[1986,1990,1995,2000,2005,2010,2015,2020,2023,2025]].round(1)); print(nz.loc[[1986,1990,1995,2000,2005,2010,2015,2020,2023,2025]].round(1))
print(fits.round(3)); print(au_fits.round(3)); print(nat.loc[[1975,1980,1986,1990,2000,2010,2020,2025]].round(0)); print(prem.round(2).iloc[[0,5,10,15,18,20,22,-1]]); print(akl_prem.round(2).iloc[::5])
