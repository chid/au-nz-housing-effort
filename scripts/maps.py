"""Build inline-SVG choropleth maps for AU SA4s and NZ territorial authorities. Usage: python maps.py <scratch_dir> <out.json>"""
import json, sys, math, re, unicodedata, html
import pandas as pd, numpy as np
S=sys.argv[1]; OUT=sys.argv[2]
def norm(x):
    x=unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode()
    x=re.sub(r'\s+(District|City|Local Board Area|Local Board)$','',x).strip()
    return {'Wanganui':'Whanganui'}.get(x,x)
# ---------------- geometry ----------------
def rings(geom):
    if geom['type']=='Polygon': return geom['coordinates']
    if geom['type']=='MultiPolygon': return [r for p in geom['coordinates'] for r in p]
    return []
def make_proj(lon0,lat0,lon1,lat1,x,y,w,h):
    k=math.cos(math.radians((lat0+lat1)/2)); sx=w/((lon1-lon0)*k); sy=h/(lat1-lat0); s=min(sx,sy)
    ox=x+(w-(lon1-lon0)*k*s)/2; oy=y+(h-(lat1-lat0)*s)/2
    return lambda lo,la:(ox+(lo-lon0)*k*s, oy+(lat1-la)*s)
def path_d(feat_rings,proj,keep=lambda r:True,minpts=3):
    out=[]
    for r in feat_rings:
        if not keep(r): continue
        pts=[proj(lo,la) for lo,la in r]; q=[]
        for x,y in pts:
            p=(round(x,1),round(y,1))
            if not q or q[-1]!=p: q.append(p)
        if len(q)<minpts: continue
        out.append('M'+'L'.join(f'{x:g},{y:g}' for x,y in q)+'Z')
    return ''.join(out)
# ---------------- bins / palette ----------------
BIN=['b0','b1','b2','b3','b4','b5','b6']
def binner(breaks): return lambda v: None if v is None or (isinstance(v,float) and math.isnan(v)) else BIN[int(np.searchsorted(breaks,v,side='right'))]
def legend(breaks,fmt,low,high,seq=False):
    labs=[f'< {fmt(breaks[0])}']+[f'{fmt(a)} to {fmt(b)}' for a,b in zip(breaks[:-1],breaks[1:])]+[f'≥ {fmt(breaks[-1])}']
    bins=BIN if not seq else ['s0','s1','s2','s3','s4','s5','s6'][:len(labs)]
    cells=''.join(f'<span class="lg-cell"><i class="lg-sw {b}"></i>{l}</span>' for b,l in zip(bins,labs))
    return f'<div class="lg"><span class="lg-end">{low}</span>{cells}<span class="lg-end">{high}</span></div>'
# ---------------- AU data ----------------
au=pd.read_csv(f'{S}/au/au_raw.csv',dtype={'sa4':str}).set_index('sa4')
au['price']=(au.house_price*au.house_n+au.unit_price*au.unit_n)/(au.house_n+au.unit_n)
au['eff']=au.price/au.wage; au['dens']=au.workers_pow_2021/au.area_km2
dbr=pd.read_csv(f'{S}/au/dbr_sa4.csv',low_memory=False,dtype={'Code':str}); dbr=dbr[dbr.Code.str.fullmatch(r'\d{3}')]
J='Jobs in Australia - year ended 30 June :: Number of employee jobs - total'; dbr[J]=pd.to_numeric(dbr[J].astype(str).str.replace(',',''),errors='coerce')
jj=dbr.pivot_table(index='Code',columns='Year',values=J); au['jobs_g']=(jj[2023]/jj[2018]-1)*100
nat_au=(jj[2023].sum()/jj[2018].sum()-1)*100; au['jobs_rel']=au.jobs_g-nat_au
# ---------------- NZ data ----------------
nz=pd.read_csv(f'{S}/nz/nz_raw.csv',index_col=0).drop(index=['Chatham Islands']); nz['eff']=nz.median_sale_price/nz.wage
b=pd.read_csv(f'{S}/nz/fnz_bd.csv'); b=b[(b.Measure=='Employee count')&(b.Industry=='Total all industries')]; b['ta']=b['Territorial authority'].map(norm)
emp=b.pivot_table(index='ta',columns='Year as at February',values='Value',aggfunc='sum')
panel=pd.read_csv(f'{S}/ts_out/nz_district_effort_panel_2000_2024.csv'); pw=panel.pivot(index='ta',columns='year',values='eff')
nz['emp_g']=(emp[2025]/emp[2016]-1)*100; nat_nz=(emp.loc['New Zealand',2025]/emp.loc['New Zealand',2016]-1)*100; nz['emp_rel']=nz.emp_g-nat_nz
nz['eff_chg']=pw[2024]-pw[2004]
# official land area from boundary layer replaces wiki area for density
talb=json.load(open(f'{S}/geo/talb.geojson'))
ta_feats={}
for f in talb['features']:
    p=f['properties']; code=p['TALB2023_V1_00']; name=p['TALB2023_V1_00_NAME_ASCII']
    ta='Auckland' if code.startswith('076') else norm(name)
    if ta in ('Chatham Islands','Area Outside Territorial Authority'): continue
    ta_feats.setdefault(ta,[]).extend(rings(f['geometry']))
land={}
for f in talb['features']:
    p=f['properties']; code=p['TALB2023_V1_00']; ta='Auckland' if code.startswith('076') else norm(p['TALB2023_V1_00_NAME_ASCII']); land[ta]=land.get(ta,0)+p['LAND_AREA_SQ_KM']
nz['land_km2']=pd.Series(land); nz['dens']=nz.employees/nz.land_km2
missing=[t for t in nz.index if t not in ta_feats]; print('NZ unmatched',missing,file=sys.stderr)
# ---------------- AU geometry ----------------
g=json.load(open(f'{S}/geo/sa4_0.02.geojson')); gi=json.load(open(f'{S}/geo/sa4_0.002.geojson'))
def au_keep(r):  # drop Macquarie Is., Lord Howe, Norfolk etc.
    lo=np.mean([p[0] for p in r]); la=np.mean([p[1] for p in r]); return 112<lo<154.2 and -44<la<-9
W,H=780,560
main=make_proj(112.8,-43.8,153.8,-10.3,4,4,600,552)
INS=[('Sydney',(150.35,-34.25,151.4,-33.45)),('Melbourne',(144.55,-38.25,145.55,-37.55)),('Brisbane',(152.65,-27.75,153.25,-27.2)),('Perth',(115.65,-32.3,116.15,-31.65))]
ix,iw,ih=616,160,131
au_defs=[]; au_keys=[]
for f in g['features']:
    c=f['properties']['sa4_code_2021']
    if c not in au.index: continue
    d=path_d(rings(f['geometry']),main,au_keep)
    if d: au_defs.append(f'<path id="au{c}" d="{d}"/>'); au_keys.append(c)
inset_defs=[]; inset_uses={}
fi={f['properties']['sa4_code_2021']:f for f in gi['features']}
for n in range(len(INS)):
    name,(a,bb,cc,dd)=INS[n][0],INS[n][1]; y=4+n*(ih+7); pr=make_proj(a,bb,cc,dd,ix,y,iw,ih)
    codes=[]
    for c,f in fi.items():
        if c not in au.index: continue
        def inbox(r,a=a,bb=bb,cc=cc,dd=dd): return any(a-.3<p[0]<cc+.3 and bb-.3<p[1]<dd+.3 for p in r)
        dpath=path_d(rings(f['geometry']),pr,inbox)
        if dpath: inset_defs.append(f'<path id="ai{n}_{c}" d="{dpath}"/>'); codes.append(c)
    inset_uses[n]=(name,y,codes)
# ---------------- NZ geometry ----------------
nzW,nzH=600,600
nzmain=make_proj(166.3,-47.4,178.7,-34.3,4,4,420,592)
NZINS=[('Wellington',(174.6,-41.42,175.25,-40.95)),('Auckland and Waikato',(174.3,-38.2,175.9,-36.6))]
nz_defs=[]; nz_inset_defs=[]; nz_inset_uses={}
for ta,rs in ta_feats.items():
    d=path_d(rs,nzmain)
    if d: nz_defs.append(f'<path id="nz{re.sub("[^A-Za-z]","",ta)}" d="{d}"/>')
for n,(name,(a,bb,cc,dd)) in enumerate(NZINS):
    y=4+n*300; pr=make_proj(a,bb,cc,dd,440,y,156,290); codes=[]
    for ta,rs in ta_feats.items():
        keep=lambda r,a=a,bb=bb,cc=cc,dd=dd: any(a-.2<p[0]<cc+.2 and bb-.2<p[1]<dd+.2 for p in r)
        d=path_d(rs,pr,keep)
        if d: nz_inset_defs.append(f'<path id="ni{n}{re.sub("[^A-Za-z]","",ta)}" d="{d}"/>'); codes.append(ta)
    nz_inset_uses[n]=(name,y,codes)
# ---------------- map builders ----------------
def esc(s): return html.escape(str(s),quote=True)
def au_map(mid,col,breaks,fmt,title,low,high,seq=False):
    bn=binner(breaks); o=[f'<svg viewBox="0 0 {W} {H}" class="map" role="img" aria-label="{esc(title)}" id="{mid}">']
    def use(href,c):
        v=au.at[c,col]; v=None if pd.isna(v) else float(v); cl=bn(v) or 'na'
        if seq and cl!='na': cl='s'+cl[1:]
        return f'<use href="#{href}" class="{cl}" data-n="{esc(au.at[c,"name"])}" data-v="{esc(fmt(v) if v is not None else "n/a")}"/>'
    o+= [use(f'au{c}',c) for c in au_keys]
    o.append(f'<defs>'+''.join(f'<clipPath id="{mid}c{n}"><rect x="{ix}" y="{y}" width="{iw}" height="{ih}"/></clipPath>' for n,(nm,y,cs) in inset_uses.items())+'</defs>')
    for n,(nm,y,cs) in inset_uses.items():
        o.append(f'<g clip-path="url(#{mid}c{n})">'+''.join(use(f'ai{n}_{c}',c) for c in cs)+'</g>')
        o.append(f'<rect x="{ix}" y="{y}" width="{iw}" height="{ih}" class="inset-frame"/><text x="{ix+6}" y="{y+14}" class="inset-lbl">{nm}</text>')
    o.append('</svg>')
    return dict(svg=''.join(o),legend=legend(breaks,fmt,low,high,seq))
def nz_map(mid,col,breaks,fmt,title,low,high,seq=False):
    bn=binner(breaks); o=[f'<svg viewBox="0 0 {nzW} {nzH}" class="map" role="img" aria-label="{esc(title)}" id="{mid}">']
    def use(href,ta):
        v=nz[col].get(ta,np.nan); v=None if pd.isna(v) else float(v); cl=bn(v) or 'na'
        if seq and cl!='na': cl='s'+cl[1:]
        return f'<use href="#{href}" class="{cl}" data-n="{esc(ta)}" data-v="{esc(fmt(v) if v is not None else "n/a")}"/>'
    o+=[use('nz'+re.sub('[^A-Za-z]','',ta),ta) for ta in ta_feats]
    o.append('<defs>'+''.join(f'<clipPath id="{mid}c{n}"><rect x="440" y="{y}" width="156" height="290"/></clipPath>' for n,(nm,y,cs) in nz_inset_uses.items())+'</defs>')
    for n,(nm,y,cs) in nz_inset_uses.items():
        o.append(f'<g clip-path="url(#{mid}c{n})">'+''.join(use(f'ni{n}'+re.sub('[^A-Za-z]','',ta),ta) for ta in cs)+'</g>')
        o.append(f'<rect x="440" y="{y}" width="156" height="290" class="inset-frame"/><text x="446" y="{y+14}" class="inset-lbl">{nm}</text>')
    o.append('</svg>'); return dict(svg=''.join(o),legend=legend(breaks,fmt,low,high,seq))
yrs=lambda v:f'{v:.1f} yrs' if v is not None else ''
def num(v): return f'{v:,.0f}' if v>=10 else (f'{v:.1f}' if v>=1 else f'{v:.2f}')
EB=[7,9,10.5,12,14,17]; DB=[0.5,2,8,30,120,500]; GB=[-15,-7,-2,2,7,15]
fe=lambda v: f'{v:g}'
M={}
M['au_eff']=au_map('m_au_eff','eff',EB,lambda v:f'{v:.1f} yrs' if v>30 or v<1e9 else '', 'Australia effort by SA4','more affordable','less affordable')
M['au_dens']=au_map('m_au_dens','dens',DB,lambda v:f'{num(v)} jobs/km²','Australia job density by SA4','sparser','denser')
M['au_jobs']=au_map('m_au_jobs','jobs_rel',GB,lambda v:f'{v:+.1f} pp','Australia job growth vs national','slower','faster')
M['nz_eff']=nz_map('m_nz_eff','eff',EB,lambda v:f'{v:.1f} yrs','NZ effort by district','more affordable','less affordable')
M['nz_dens']=nz_map('m_nz_dens','dens',DB,lambda v:f'{num(v)} jobs/km²','NZ job density by district','sparser','denser')
M['nz_jobs']=nz_map('m_nz_jobs','emp_rel',GB,lambda v:f'{v:+.1f} pp','NZ employee growth vs national','slower','faster')
M['nz_chg']=nz_map('m_nz_chg','eff_chg',[1,2,3,4,5,6],lambda v:f'{v:+.1f} yrs','NZ change in effort 2004–2024','smaller rise','larger rise',seq=True)
# legends need plain labels (override fmt for legend ranges)
M['au_eff']['legend']=legend(EB,fe,'more affordable','less affordable'); M['nz_eff']['legend']=legend(EB,fe,'more affordable','less affordable')
M['au_dens']['legend']=legend(DB,num,'sparser','denser'); M['nz_dens']['legend']=legend(DB,num,'sparser','denser')
M['au_jobs']['legend']=legend(GB,lambda v:(f'{v:+g}').replace('-','−'),'slower','faster'); M['nz_jobs']['legend']=legend(GB,lambda v:(f'{v:+g}').replace('-','−'),'slower','faster')
M['nz_chg']['legend']=legend([1,2,3,4,5,6],lambda v:f'+{v:g}','smaller rise','larger rise',seq=True)
defs=f'<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>{"".join(au_defs)}{"".join(inset_defs)}{"".join(nz_defs)}{"".join(nz_inset_defs)}</defs></svg>'
facts=dict(nat_au_jobs=round(float(nat_au),1),nat_nz_emp=round(float(nat_nz),1),
  au_top_growth=au.sort_values('jobs_g',ascending=False)[['name','jobs_g']].head(6).round(1).values.tolist(),
  au_low_growth=au.sort_values('jobs_g')[['name','jobs_g']].head(5).round(1).values.tolist(),
  nz_top_growth=nz.sort_values('emp_g',ascending=False).emp_g.head(6).round(1).to_dict(), nz_low_growth=nz.sort_values('emp_g').emp_g.head(5).round(1).to_dict(),
  au_median_eff=round(float(au.eff.median()),1), nz_median_eff=round(float(nz.eff.median()),1))
json.dump(dict(defs=defs,maps=M,facts=facts),open(OUT,'w'))
print(facts); print('defs KB',len(defs)//1024, {k:len(v['svg'])//1024 for k,v in M.items()})
