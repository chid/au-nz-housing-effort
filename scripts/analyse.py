import pandas as pd, numpy as np, json, sys, glob, math
S=sys.argv[1]
# ---------- mortgage (site assumptions) ----------
def years_to_repay(E,earners=1,rate=0.04,ltv=0.9,dsti=0.45,net=0.7):
    L=ltv*E; P=dsti*net*earners/12; r=rate/12   # in units of annual wage, monthly payment
    if r*L>=P: return math.inf
    return -math.log(1-r*L/P)/math.log(1+r)/12
def fit(x,y):
    X=np.column_stack([x,np.ones(len(x))]); b,*_=np.linalg.lstsq(X,y,rcond=None); yh=X@b
    return b, 1-((y-yh)**2).sum()/((y-y.mean())**2).sum(), y-yh
def iqr_out(res):
    q1,q3=np.percentile(res,[25,75]); i=q3-q1; return (res<q1-1.5*i)|(res>q3+1.5*i)
def mfit(X,y):
    X=np.column_stack(X+[np.ones(len(y))]); b,*_=np.linalg.lstsq(X,y,rcond=None); yh=X@b
    return b,1-((y-yh)**2).sum()/((y-y.mean())**2).sum()
def analyse(df,label,tour):
    d=df.dropna(subset=['eff','dens']).copy(); x=np.log(d.dens.values); y=d.eff.values
    b,r2,res=fit(x,y); out=iqr_out(res); b2,r2c,_=fit(x[~out],y[~out])
    t=d.dropna(subset=[tour]); bt,r2t,rest=fit(t[tour].values,t.eff.values)
    ot=iqr_out(rest); _,r2tc,_=fit(t[tour].values[~ot],t.eff.values[~ot])
    bm,r2m=mfit([np.log(t.dens.values),t[tour].values],t.eff.values)
    d['outlier']=out
    return dict(label=label,n=int(len(d)),slope=float(b[0]),icpt=float(b[1]),r2=float(r2),r2_clean=float(r2c),slope_clean=float(b2[0]),icpt_clean=float(b2[1]),
        outliers=d.index[out].tolist(),n_tour=int(len(t)),r2_tour=float(r2t),r2_tour_clean=float(r2tc),r2_multi=float(r2m),coef_multi=[float(v) for v in bm]),d
def conc(emp,area):
    o=(emp/area).sort_values(ascending=False); e=emp[o.index]; a=area[o.index]
    ce=e.cumsum()/e.sum(); k=(ce<0.5).sum()+1; land=a.iloc[:k].sum()/a.sum()
    # regions to reach 50% by employment rank
    er=emp.sort_values(ascending=False); nreg=int(((er.cumsum()/er.sum())<0.5).sum()+1)
    v=np.sort(emp.values); n=len(v); gini=(2*np.arange(1,n+1)-n-1).dot(v)/(n*v.sum())
    return dict(land_share_50=float(land),regions_50=nreg,gini=float(gini),n=n)
R={}
# ================= AUSTRALIA =================
a=pd.read_csv(f'{S}/au/au_raw.csv',index_col=0,dtype={'sa4':str}); a.index=a.index.astype(str)
a['price']=(a.house_price*a.house_n+a.unit_price*a.unit_n)/(a.house_n+a.unit_n)
a['eff']=a.price/a.wage; a['eff_unit']=a.unit_price/a.wage; a['eff_house']=a.house_price/a.wage
a['dens']=a.workers_pow_2021/a.area_km2; a['accfood']=a.jobs_accfood/a.jobs_emp_total*100
G={'Greater Sydney':['102']+[str(i) for i in range(115,129)],'Greater Melbourne':[str(i) for i in range(206,215)],
   'Greater Brisbane':['301','302','303','304','305','310','311','313','314'],'Greater Adelaide':['401','402','403','404'],
   'Greater Perth':[str(i) for i in range(502,508)],'Greater Hobart':['601'],'Greater Darwin':['701'],'Canberra (ACT)':['801']}
dbr=pd.read_csv(f'{S}/au/dbr_sa4.csv',low_memory=False,dtype={'Code':str}); dbr=dbr[(dbr.Year==2023)]
P='Residential property transfers - year ended 30 June :: '
def num(s): return pd.to_numeric(s.astype(str).str.replace(',',''),errors='coerce')
gcc={'Greater Sydney':'1GSYD','Greater Melbourne':'2GMEL','Greater Brisbane':'3GBRI','Greater Adelaide':'4GADE','Greater Perth':'5GPER'}
rows=[]
sa2g={c:g for g,cs in G.items() for c in cs}
for g,cs in G.items():
    s=a.loc[cs]; r=dict(name=g,state=s.state.iloc[0],workers_pow_2021=s.workers_pow_2021.sum(),workers_pow_2016=s.workers_pow_2016.sum(),area_km2=s.area_km2.sum(),
        jobs_accfood=s.jobs_accfood.sum(),jobs_emp_total=s.jobs_emp_total.sum(),population=s.population.sum(),capital=True)
    if g in gcc:
        q=dbr[dbr.Code==gcc[g]].iloc[0]
        hp,hn,up,un,w=[float(num(pd.Series([q[P+k]]))[0]) for k in ['Median price of established house transfers ($)','Number of established house transfers (no.)','Median price of attached dwelling transfers ($)','Number of attached dwelling transfers (no.)']]+[float(num(pd.Series([q['Personal income in Australia - year ended 30 June :: Median employee income ($)']]))[0])]
    else:
        q=s.iloc[0]; hp,hn,up,un,w=q.house_price,q.house_n,q.unit_price,q.unit_n,q.wage
    r.update(house_price=hp,unit_price=up,house_n=hn,unit_n=un,wage=w); rows.append(r)
lm=pd.DataFrame(rows).set_index('name')
rest=a[~a.index.isin(sa2g)].copy(); rest['capital']=False; rest=rest.set_index('name')
lm=pd.concat([lm,rest[lm.columns.intersection(rest.columns)]])
lm['price']=(lm.house_price*lm.house_n+lm.unit_price*lm.unit_n)/(lm.house_n+lm.unit_n)
lm['eff']=lm.price/lm.wage; lm['eff_unit']=lm.unit_price/lm.wage; lm['eff_house']=lm.house_price/lm.wage
lm['dens']=lm.workers_pow_2021/lm.area_km2; lm['accfood']=lm.jobs_accfood/lm.jobs_emp_total*100
au_sa4=a.set_index('name')
R['au_sa4'],d1=analyse(au_sa4,'Australia, 88 SA4s','accfood')
R['au_lm'],d2=analyse(lm,'Australia, labour-market regions','accfood')
# ================= NEW ZEALAND =================
n=pd.read_csv(f'{S}/nz/nz_raw.csv',index_col=0).drop(index=['Chatham Islands'])
n.loc[n.guest_nights_pc<=0,'guest_nights_pc']=np.nan
n['guest_nights']=n.guest_nights_pc*n.population
b=pd.read_csv(f'{S}/nz/fnz_bd.csv'); b=b[(b.Measure=='Employee count')&(b.Industry=='Total all industries')]
import unicodedata,re
def norm(x):
    x=unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode(); return re.sub(r'\s+(District|City)$','',x).strip()
b['ta']=b['Territorial authority'].map(norm)
emp_y=b.pivot_table(index='ta',columns='Year as at February',values='Value',aggfunc='sum')
n['employees_2016']=emp_y[2016]; n['employees_2025']=emp_y[2025]
acc=pd.read_csv(f'{S}/nz/fnz_bd.csv'); acc=acc[(acc.Measure=='Employee count')&(acc['Year as at February']==2023)&(acc['Industry Code']=='H')]
def finish(x):
    x['eff']=x.median_sale_price/x.wage; x['dens']=x.employees/x.area_km2; x['gn_pw']=x.guest_nights/x.employees; x['accfood']=x.accfood_share*100; return x
n=finish(n)
M={'Wellington metro':['Wellington','Lower Hutt','Upper Hutt','Porirua'],'Christchurch metro':['Christchurch','Selwyn','Waimakariri'],
   'Hamilton metro':['Hamilton','Waipa','Waikato'],'Tauranga metro':['Tauranga','Western Bay of Plenty'],'Napier-Hastings':['Napier','Hastings']}
m=n.copy(); m['members']=m.index
for g,cs in M.items():
    s=n.loc[cs]; r=dict(median_sale_price=(s.median_sale_price*s.residential_sales).sum()/s.residential_sales.sum(),residential_sales=s.residential_sales.sum(),
      employees=s.employees.sum(),wage=(s.wage*s.employees).sum()/s.employees.sum(),accfood_share=(s.accfood_share*s.employees).sum()/s.employees.sum(),
      guest_nights=s.guest_nights.sum(min_count=1),population=s.population.sum(),area_km2=s.area_km2.sum(),employees_2016=s.employees_2016.sum(),employees_2025=s.employees_2025.sum(),members=', '.join(cs))
    m=m.drop(index=cs); m.loc[g]=pd.Series(r)
m=finish(m)
R['nz_ta'],d3=analyse(n,'New Zealand, 66 districts','gn_pw')
R['nz_lm'],d4=analyse(m,'New Zealand, metros merged','gn_pw')
# proxy check: NZ acc&food share vs guest nights
t=n.dropna(subset=['gn_pw']); R['nz_proxy_corr']=float(np.corrcoef(t.accfood,t.gn_pw)[0,1])
R['nz_ta_accfood']=analyse(n,'NZ acc&food','accfood')[0]
# ---------- country summaries ----------
def wmean(d,col,w): d=d.dropna(subset=[col,w]); return float((d[col]*d[w]).sum()/d[w].sum())
R['summary']=dict(au=dict(eff=wmean(lm,'eff','workers_pow_2021'),eff_unit=wmean(lm,'eff_unit','workers_pow_2021'),eff_house=wmean(lm,'eff_house','workers_pow_2021')),
                  nz=dict(eff=wmean(m,'eff','employees')))
# mortgage years
for d,w in [(lm,'workers_pow_2021'),(m,'employees')]:
    d['yrs1']=[years_to_repay(e,1) for e in d.eff]; d['yrs2']=[years_to_repay(e,2) for e in d.eff]
    d['yrs2_6']=[years_to_repay(e,2,rate=0.06) for e in d.eff]
lm['yrs1_unit']=[years_to_repay(e,1) for e in lm.eff_unit]; lm['yrs2_unit']=[years_to_repay(e,2) for e in lm.eff_unit]
def share_over(d,col,w,th=15): return float(d.loc[d[col]>th,w].sum()/d[w].sum())
R['cliff']=dict(au_two_15=share_over(lm,'yrs2','workers_pow_2021'),au_two_25=share_over(lm,'yrs2','workers_pow_2021',25),au_one_inf=share_over(lm,'yrs1','workers_pow_2021',1e9-1) if False else float(lm.loc[np.isinf(lm.yrs1),'workers_pow_2021'].sum()/lm.workers_pow_2021.sum()),
  au_unit_one_15=share_over(lm,'yrs1_unit','workers_pow_2021'),au_unit_one_inf=float(lm.loc[np.isinf(lm.yrs1_unit),'workers_pow_2021'].sum()/lm.workers_pow_2021.sum()),
  nz_two_15=share_over(m,'yrs2','employees'),nz_two_25=share_over(m,'yrs2','employees',25),nz_one_inf=float(m.loc[np.isinf(m.yrs1),'employees'].sum()/m.employees.sum()),
  au_two6_inf=float(lm.loc[np.isinf(lm.yrs2_6),'workers_pow_2021'].sum()/lm.workers_pow_2021.sum()),nz_two6_inf=float(m.loc[np.isinf(m.yrs2_6),'employees'].sum()/m.employees.sum()))
# ---------- concentration ----------
R['conc']=dict(au_2021=conc(a.workers_pow_2021,a.area_km2),au_2016=conc(a.workers_pow_2016,a.area_km2),
  nz_2025=conc(n.employees_2025,n.area_km2),nz_2016=conc(n.employees_2016,n.area_km2),nz_2023=conc(n.employees,n.area_km2))
# capital/top shares
R['top']=dict(
  au={g:dict(s2016=float(lm.loc[g,'workers_pow_2016']/lm.workers_pow_2016.sum()),s2021=float(lm.loc[g,'workers_pow_2021']/lm.workers_pow_2021.sum()),growth=float(lm.loc[g,'workers_pow_2021']/lm.loc[g,'workers_pow_2016']-1)) for g in G},
  nz={g:dict(s2016=float(m.loc[g,'employees_2016']/m.employees_2016.sum()),s2025=float(m.loc[g,'employees_2025']/m.employees_2025.sum()),growth=float(m.loc[g,'employees_2025']/m.loc[g,'employees_2016']-1)) for g in ['Auckland','Wellington metro','Christchurch metro','Hamilton metro','Tauranga metro','Queenstown-Lakes']})
R['au_total_growth']=float(a.workers_pow_2021.sum()/a.workers_pow_2016.sum()-1); R['nz_total_growth']=float(n.employees_2025.sum()/n.employees_2016.sum()-1)
# ---------- export rows ----------
def rows_(d,cols): 
    x=d[cols].replace([np.inf],None).reset_index(); x.columns=['name']+cols; return json.loads(x.to_json(orient='records'))
R['rows']=dict(
  au_lm=rows_(d2.join(lm[['yrs1','yrs2','yrs2_6','yrs1_unit','eff_unit','eff_house','price','wage','workers_pow_2021','workers_pow_2016','state','capital']],rsuffix='_'),['eff','dens','accfood','outlier','yrs1','yrs2','yrs2_6','yrs1_unit','eff_unit','eff_house','price','wage','workers_pow_2021','workers_pow_2016','state','capital','area_km2']),
  au_sa4=rows_(d1,['eff','dens','accfood','outlier','state','price','wage','workers_pow_2021','workers_pow_2016','area_km2']),
  nz_ta=rows_(d3,['eff','dens','gn_pw','accfood','outlier','median_sale_price','wage','employees','employees_2016','employees_2025','area_km2']),
  nz_lm=rows_(d4.join(m[['yrs1','yrs2','yrs2_6']]),['eff','dens','gn_pw','accfood','outlier','yrs1','yrs2','yrs2_6','median_sale_price','wage','employees','employees_2016','employees_2025','area_km2']))
json.dump(R,open(f'{S}/results.json','w'),indent=1,default=lambda o: None if (isinstance(o,float) and math.isinf(o)) else (bool(o) if isinstance(o,np.bool_) else str(o)))
lm.to_csv(f'{S}/au_labour_market_regions.csv'); m.to_csv(f'{S}/nz_labour_market_regions.csv')
for k in ['au_sa4','au_lm','nz_ta','nz_lm','nz_ta_accfood']: print(k,{kk:(round(v,3) if isinstance(v,float) else v) for kk,v in R[k].items()})
for k in ['nz_proxy_corr','summary','cliff','conc','top','au_total_growth','nz_total_growth']: print(k,R[k])
print(lm.sort_values('eff',ascending=False)[['eff','eff_unit','dens','accfood','yrs1','yrs2','yrs2_6','yrs1_unit']].round(2).head(20).to_string())
print(m.sort_values('eff',ascending=False)[['eff','dens','gn_pw','yrs1','yrs2','yrs2_6']].round(2).head(15).to_string())
