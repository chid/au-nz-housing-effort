import pandas as pd, numpy as np, json, sys, unicodedata, re
S=sys.argv[1]
def fit(cols,y):
    X=np.column_stack(cols+[np.ones(len(y))]); b,*_=np.linalg.lstsq(X,y,rcond=None); return float(1-((y-X@b)**2).sum()/((y-y.mean())**2).sum()),[float(v) for v in b]
R={}
# AU
a=pd.read_csv(f'{S}/au_labour_market_regions.csv',index_col=0)
dbr=pd.read_csv(f'{S}/au/dbr_sa4.csv',low_memory=False,dtype={'Code':str}); dbr=dbr[dbr.Code.str.fullmatch(r'\d{3}')]
E='Estimated resident population - 30 June :: Estimated resident population (no.)'; J='Jobs in Australia - year ended 30 June :: Number of employee jobs - total'
for c in [E,J]: dbr[c]=pd.to_numeric(dbr[c].astype(str).str.replace(',',''),errors='coerce')
pe=dbr.pivot_table(index='Code',columns='Year',values=E); jj=dbr.pivot_table(index='Code',columns='Year',values=J)
G={'Greater Sydney':['102']+[str(i) for i in range(115,129)],'Greater Melbourne':[str(i) for i in range(206,215)],
   'Greater Brisbane':['301','302','303','304','305','310','311','313','314'],'Greater Adelaide':['401','402','403','404'],
   'Greater Perth':[str(i) for i in range(502,508)],'Greater Hobart':['601'],'Greater Darwin':['701'],'Canberra (ACT)':['801']}
lab=dbr.drop_duplicates('Code').set_index('Code').Label
code2lm={c:g for g,cs in G.items() for c in cs}
key=pd.Series({c:code2lm.get(c,lab[c]) for c in pe.index})
pg=pe.groupby(key).sum(); jg=jj.groupby(key).sum()
a['pop_growth_20_23']=pg[2023]/pg[2020]-1
a['jobs_2018']=jg[2018]; a['jobs_2023']=jg[2023]
d=a.dropna(subset=['pop_growth_20_23'])
R['au_growth']=fit([d.pop_growth_20_23.values],d.eff.values)
R['au_dens_growth']=fit([np.log(d.dens.values),d.pop_growth_20_23.values],d.eff.values)
R['au_dens_growth_tour']=fit([np.log(d.dens.values),d.pop_growth_20_23.values,d.accfood.values],d.eff.values)
R['au_price_dens']=fit([np.log(d.dens.values)],np.log(d.price.values)); R['au_wage_dens']=fit([np.log(d.dens.values)],np.log(d.wage.values))
R['au_jobs_share']={g:dict(s2018=float(jg.loc[g,2018]/jg[2018].sum()),s2023=float(jg.loc[g,2023]/jg[2023].sum()),growth=float(jg.loc[g,2023]/jg.loc[g,2018]-1)) for g in G}
R['au_jobs_total_growth']=float(jg[2023].sum()/jg[2018].sum()-1)
# NZ
def norm(x):
    x=unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode(); x=re.sub(r'\s+(District|City)$','',x).strip()
    return {'Wanganui':'Whanganui','Southland district':'Southland','Waikato district':'Waikato','Wellington city':'Wellington'}.get(x,x)
r=pd.read_csv(f'{S}/nz/rear/all-data.csv',dtype=str,low_memory=False)
p=r[(r.indicator=='population estimates')&(r['transform']=='original')]; p=p.assign(ta=p.area.map(norm),value=p.value.astype(float)).pivot_table(index='ta',columns='year',values='value')
n=pd.read_csv(f'{S}/nz/nz_raw.csv',index_col=0).drop(index=['Chatham Islands'])
n['eff']=n.median_sale_price/n.wage; n['dens']=n.employees/n.area_km2
n['pop_growth_13_23']=p['2023']/p['2013']-1
d=n.dropna(subset=['pop_growth_13_23','eff'])
R['nz_growth']=fit([d.pop_growth_13_23.values],d.eff.values)
R['nz_growth_exQL']=fit([d.drop(index='Queenstown-Lakes').pop_growth_13_23.values],d.drop(index='Queenstown-Lakes').eff.values)
R['nz_dens_growth']=fit([np.log(d.dens.values),d.pop_growth_13_23.values],d.eff.values)
R['nz_price_dens']=fit([np.log(d.dens.values)],np.log(d.median_sale_price.values)); R['nz_wage_dens']=fit([np.log(d.dens.values)],np.log(d.wage.values))
R['rows_nz_growth']=[dict(name=i,pg=float(v.pop_growth_13_23),eff=float(v.eff)) for i,v in d.iterrows()]
R['rows_au_growth']=[dict(name=i,pg=float(v.pop_growth_20_23),eff=float(v.eff),capital=bool(v.capital)) for i,v in a.dropna(subset=['pop_growth_20_23']).iterrows()]
R['nz_pop_growth_national']=float(p.loc['New Zealand','2023']/p.loc['New Zealand','2013']-1) if 'New Zealand' in p.index else None
json.dump(R,open(f'{S}/robust.json','w'),indent=1)
for k,v in R.items():
    if not k.startswith('rows'): print(k,v)
print(a.sort_values('pop_growth_20_23',ascending=False)[['eff','pop_growth_20_23']].head(10).round(3))
