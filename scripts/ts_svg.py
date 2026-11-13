import pandas as pd, numpy as np, json, sys
TS=sys.argv[1]; OUT=sys.argv[2]
au=pd.read_csv(f'{TS}/au_capital_effort_1986_2025.csv',index_col=0); nz=pd.read_csv(f'{TS}/nz_metro_effort_1986_2025.csv',index_col=0)
prem=pd.read_csv(f'{TS}/au_capital_vs_rest_price_ratio.csv',index_col=0); akl=pd.read_csv(f'{TS}/nz_auckland_price_premium.csv',index_col=0).iloc[:,0]
fn=pd.read_csv(f'{TS}/nz_district_fits_2000_2024.csv',index_col=0); fa=pd.read_csv(f'{TS}/au_region_fits_2020_2023.csv',index_col=0)
S1,S2,S3,S4='#2a78d6','#eb6834','#1baf7a','#c98500'   # categorical slots 1-4 (slot 4 stepped darker for contrast)
def chart(series,x0,x1,y0,y1,yt,ylab,shade=None,others=None,W=720,H=300,fmt=lambda v:str(v)):
    m=dict(l=46,r=130,t=12,b=30); sx=lambda x:m['l']+(x-x0)/(x1-x0)*(W-m['l']-m['r']); sy=lambda y:H-m['b']-(y-y0)/(y1-y0)*(H-m['t']-m['b'])
    o=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{ylab}" class="ts">']
    if shade: o.append(f'<rect x="{sx(shade[0]):.1f}" y="{m["t"]}" width="{sx(shade[1])-sx(shade[0]):.1f}" height="{H-m["t"]-m["b"]}" class="ts-shade"/>')
    for v in yt: o.append(f'<line x1="{m["l"]}" x2="{W-m["r"]}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="ts-grid"/><text x="{m["l"]-8}" y="{sy(v)+4:.1f}" text-anchor="end" class="ts-ax">{fmt(v)}</text>')
    for x in range(x0 - x0%5 + (5 if x0%5 else 0), x1+1, 5): o.append(f'<text x="{sx(x):.1f}" y="{H-10}" text-anchor="middle" class="ts-ax">{x}</text>')
    o.append(f'<line x1="{m["l"]}" x2="{W-m["r"]}" y1="{H-m["b"]}" y2="{H-m["b"]}" class="ts-base"/>')
    def path(s): s=s.dropna(); return ' '.join(f'{"M" if i==0 else "L"}{sx(x):.1f},{sy(y):.1f}' for i,(x,y) in enumerate(s.items()))
    for s in (others or []): o.append(f'<path d="{path(s)}" class="ts-other"/>')
    ends=[]
    for name,s,col,dash in series:
        o.append(f'<path d="{path(s)}" fill="none" stroke="{col}" stroke-width="2"{" stroke-dasharray=\"5 4\"" if dash else ""}><title>{name}</title></path>')
        s=s.dropna(); o.append(f'<circle cx="{sx(s.index[-1]):.1f}" cy="{sy(s.iloc[-1]):.1f}" r="3.5" fill="{col}" class="ts-dot"/>'); ends.append([sy(s.iloc[-1]),name,s.iloc[-1],sx(s.index[-1])])
    ends.sort(); 
    for i in range(1,len(ends)):
        if ends[i][0]-ends[i-1][0]<13: ends[i][0]=ends[i-1][0]+13
    for y,name,v,x in ends: o.append(f'<text x="{x+8:.1f}" y="{y+4:.1f}" class="ts-lbl">{name} {fmt(round(v,1)) if isinstance(v,float) else v}</text>')
    o.append('</svg>'); return ''.join(o)
c1=chart([('Sydney',au.Sydney,S1,0),('Melbourne',au.Melbourne,S2,0),('Brisbane',au.Brisbane,S3,0),('Perth',au.Perth,S4,0)],1986,2025,0,20,[0,5,10,15,20],'years of earnings',shade=(1986,1995),others=[au[c] for c in ['Adelaide','Hobart','Darwin','Canberra']],fmt=lambda v:f'{v:g}')
c2=chart([('Auckland',nz['Auckland'],S1,0),('Queenstown',nz['Queenstown-Lakes'],S2,0),('Wellington',nz['Wellington metro'],S3,0),('NZ',nz['New Zealand'],S4,1)],1986,2025,0,25,[0,5,10,15,20,25],'years of earnings',shade=(1986,1989),others=[nz[c] for c in ['Christchurch metro','Hamilton metro','Tauranga metro','Dunedin']],fmt=lambda v:f'{v:g}')
c3=chart([('Sydney ÷ rest NSW',prem.NSW,S1,0),('Melbourne ÷ rest Vic',prem.Vic,S2,0),('Auckland ÷ NZ',akl.loc[1990:],S3,0)],1990,2025,1,2.25,[1,1.25,1.5,1.75,2,2.25],'price ratio',others=[prem[c] for c in ['Qld','SA','WA','Tas']],fmt=lambda v:f'{v:.2f}'.rstrip('0').rstrip('.') if v!=1 else '1')
c4=chart([('NZ: 10-yr pop. growth',fn.r2_growth10,S2,0),('NZ: job density',fn.r2_density,S2,1),('AU: job density',fa.r2_density,S1,0)],2000,2024,0,0.7,[0,.1,.2,.3,.4,.5,.6,.7],'R squared',fmt=lambda v:f'{v:.1f}')
facts=dict(akl_peak=[int(akl.idxmax()),round(float(akl.max()),2)],nz_peak=[int(nz['New Zealand'].idxmax()),round(float(nz['New Zealand'].max()),1)],akl_eff_peak=[int(nz.Auckland.idxmax()),round(float(nz.Auckland.max()),1)],
  syd_peak=[int(au.Sydney.idxmax()),round(float(au.Sydney.max()),1)],prem_nsw_peak=[int(prem.NSW.idxmax()),round(float(prem.NSW.max()),2)],prem_vic_peak=[int(prem.Vic.idxmax()),round(float(prem.Vic.max()),2)])
print(facts); print(au.loc[[1986,2000,2025]].round(1).to_string()); print(nz.loc[[1986,2000,2025]].round(1).to_string())
json.dump(dict(c1=c1,c2=c2,c3=c3,c4=c4),open(OUT,'w'))
