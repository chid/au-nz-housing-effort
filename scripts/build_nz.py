import pandas as pd, re, unicodedata, sys
NZ=sys.argv[1]
def norm(x):
    x=unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode()
    x=re.sub(r'\s+(District|City)$','',x,flags=re.I).strip()
    return {'Wanganui':'Whanganui','Southland district':'Southland','Waikato district':'Waikato','Wellington city':'Wellington'}.get(x,x)
REGIONS={'Northland','Bay of Plenty',"Hawke's Bay",'Manawatu-Wanganui','Taranaki','Canterbury','Otago','West Coast','New Zealand'}
# prices: HUD 12m rolling median sale price, calendar 2023
s=pd.read_csv(f'{NZ}/hud_sales.csv')
p=s[(s.area_type=='Territorial Authority')&(s.stat_type=='12-Month rolling')&(s.period.str.startswith('2023-12'))].copy()
p['ta']=p.area.map(norm); p=p.set_index('ta')[['median_sale_price','residential_sales']]
# employees Feb 2023 by TA and industry
b=pd.read_csv(f'{NZ}/fnz_bd.csv'); b=b[(b['Measure']=='Employee count')&(b['Year as at February']==2023)]
b['ta']=b['Territorial authority'].map(norm)
tot=b[b.Industry=='Total all industries'].groupby('ta').Value.sum().rename('employees')
ind=b[b['Industry Code'].str.len()==1].pivot_table(index='ta',columns='Industry',values='Value',aggfunc='sum')
# LEED median earnings by industry by TA, 2023
r=pd.read_csv(f'{NZ}/rear/all-data.csv',dtype=str,low_memory=False)
e=r[(r.indicator=='earnings by industry')&(r['transform']=='original')&(r.year=='2023')].copy()
e['ta']=e.area.map(norm); e['value']=e.value.astype(float)
e=e[~e.area.isin(REGIONS)]
ep=e.pivot_table(index='ta',columns='feature',values='value')
ep.columns=[c.lower() for c in ep.columns]; ind.columns=[c.lower() for c in ind.columns]
ind=ind.rename(columns={'financial and insurance services':'financial and insurance services'})
common=[c for c in ind.columns if c in ep.columns]
miss=[c for c in ind.columns if c not in ep.columns]; print('unmatched industries',miss,file=sys.stderr)
w=ind[common].reindex(ep.index); v=ep[common]
mask=v.notna()&w.notna()
wage=(v.where(mask)*w.where(mask)).sum(1)/w.where(mask).sum(1)
wage.name='wage'
acc=(ind['accommodation and food services']/ind.sum(1)).rename('accfood_share')
# guest nights per capita and population (2023)
def ind1(name,year='2023'):
    x=r[(r.indicator==name)&(r['transform']=='original')&(r.year==year)&(~r.area.isin(REGIONS))].copy()
    x['ta']=x.area.map(norm); return x.groupby('ta').value.first().astype(float)
gn=ind1('guest nights per capita').rename('guest_nights_pc'); pop=ind1('population estimates').rename('population')
# area
w_=open(f'{NZ}/ta_wiki.txt').read()
rows=re.findall(r'\|\[\[([^|\]]+)(?:\|([^\]]+))?\]\]\n\|\[\[[^\n]*\n\|[^\n]*\n\|\s*align="right"\s*\|\s*\{\{Nts\|([\d.]+)\}\}',w_)
area=pd.Series({norm((b2 or a2).replace('Chatham Islands Territory','Chatham Islands')):float(c) for a2,b2,c in rows},name='area_km2')
df=pd.concat([p,tot,wage,acc,gn,pop,area],axis=1)
df=df.drop(index=[i for i in df.index if i in ('Area Outside Territorial Authority','New Zealand')],errors='ignore')
print(df.isna().sum().to_dict(),file=sys.stderr); print(df[df.isna().any(axis=1)],file=sys.stderr)
df.index.name='ta'; df.to_csv(f'{NZ}/nz_raw.csv')
