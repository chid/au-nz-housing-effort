import pandas as pd, sys, glob
AU=sys.argv[1]
d=pd.read_csv(f'{AU}/dbr_sa4.csv',low_memory=False,dtype={'Code':str})
d=d[d.Code.str.fullmatch(r'\d{3}')]
P='Residential property transfers - year ended 30 June :: '
cols={P+'Median price of established house transfers ($)':'house_price',P+'Number of established house transfers (no.)':'house_n',
 P+'Median price of attached dwelling transfers ($)':'unit_price',P+'Number of attached dwelling transfers (no.)':'unit_n',
 'Personal income in Australia - year ended 30 June :: Median employee income ($)':'wage',
 'Personal income in Australia - year ended 30 June :: Employee income earners (no.)':'earners',
 'Jobs in Australia - year ended 30 June :: Number of employee jobs - accommodation and food services':'jobs_accfood',
 'Jobs in Australia - year ended 30 June :: Number of employee jobs - total':'jobs_emp_total',
 'Estimated resident population - 30 June :: Estimated resident population (no.)':'population',
 'Land area :: Land area (ha)':'area_ha'}
out={}
for c,n in cols.items():
    s=d[['Code','Label','Year',c]].copy(); s[c]=pd.to_numeric(s[c].astype(str).str.replace(',',''),errors='coerce')
    yr=2021 if n=='area_ha' else 2023
    out[n]=s[s.Year==yr].set_index('Code')[c]
lab=d.drop_duplicates('Code').set_index('Code').Label
df=pd.DataFrame(out); df.insert(0,'name',lab)
def pow_(f,code):
    w=pd.read_csv(glob.glob(f)[0],dtype={code:str}); return w.set_index(code)['P_Tot_Tot']
df['workers_pow_2021']=pow_(f'{AU}/wpp21/*/2021Census_W01B_AUST_POW_SA4.csv','POW_SA4_CODE_2021')
df['workers_pow_2016']=pow_(f'{AU}/wpp16/*/2016Census_W01B_AUS_SA4.csv','SA4_CODE_2016')
df['area_km2']=df.area_ha/100
df['state']=df.index.str[0].map({'1':'NSW','2':'VIC','3':'QLD','4':'SA','5':'WA','6':'TAS','7':'NT','8':'ACT','9':'OT'})
df=df[df.state!='OT']
print(df.isna().sum().to_dict(),file=sys.stderr)
df.index.name='sa4'; df.to_csv(f'{AU}/au_raw.csv')
