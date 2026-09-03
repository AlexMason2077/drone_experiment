"""Exhaustive formation and position rankings for every wind condition."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'swarm_analysis'/'multidimensional'/'swarm_rows_with_baseline_adjustment.csv'
OUT=ROOT/'swarm_analysis'/'condition_rankings'; CH=OUT/'charts'; OUT.mkdir(parents=True,exist_ok=True); CH.mkdir(parents=True,exist_ok=True)
FORMS=['front','vee','diamond','echalon','column']; DIRS=['head','side','tail']; CONDS=[f'{d} lv{l}' for d in DIRS for l in [1,2]]
COL={'front':'#2878B5','vee':'#D99A22','diamond':'#D9534F','echalon':'#658B38','column':'#A14F86'}

def load():
    x=pd.read_csv(SRC,low_memory=False); x['position']=x.csv_takeoff_order.astype(int)
    x['condition']=x.wind_direction_short+' lv'+x.wind_level.astype(int).astype(str)
    x['rate']=x.csv_battery_drop/x.csv_node_duration_sec*60
    return x

def balanced(x,dims):
    # One median per SOC band first; repeated front_50 runs do not get extra weight.
    return (x.groupby(dims+['soc_bin'],observed=True)
             .agg(adjusted_drop=('excess_vs_baseline','median'),raw_drop=('csv_battery_drop','median'),
                  duration_sec=('csv_node_duration_sec','median'),rate=('rate','median'),runs=('csv_run_id','nunique'))
             .reset_index().groupby(dims,observed=True)
             .agg(adjusted_drop=('adjusted_drop','mean'),raw_drop=('raw_drop','mean'),duration_sec=('duration_sec','mean'),
                  rate=('rate','mean'),soc_bands=('soc_bin','nunique'),runs=('runs','sum')).reset_index())

def add_rank_reason(q,groups,item):
    q=q.copy(); q['rank']=q.groupby(groups).adjusted_drop.rank(method='min').astype(int)
    q['gap_from_best']=q.adjusted_drop-q.groupby(groups).adjusted_drop.transform('min')
    # Relative driver description within the exact comparison group.
    q['duration_index']=q.duration_sec/q.groupby(groups).duration_sec.transform('median')*100
    q['rate_index']=q.rate/q.groupby(groups).rate.transform('median')*100
    q['empirical_reason']=np.select(
        [(q.duration_index<=100)&(q.rate_index<=100),
         (q.duration_index-100)>(q.rate_index-100)+8,(q.rate_index-100)>(q.duration_index-100)+8],
        ['lower duration and/or discharge intensity','mainly longer mission duration','mainly higher discharge intensity'],
        default='both duration and discharge intensity')
    return q.sort_values(groups+['rank',item])

def formation_tables(x):
    q=balanced(x,['condition','distance','formation']); q=add_rank_reason(q,['condition','distance'],'formation')
    q.to_csv(OUT/'formation_ranking_by_wind_and_distance.csv',index=False)
    # Distance-balanced only when a formation has both distances for the condition.
    complete=q.groupby(['condition','formation']).distance.nunique().reset_index(name='n_distances')
    z=q.merge(complete,on=['condition','formation']); z=z[z.n_distances==2]
    z=(z.groupby(['condition','formation']).agg(adjusted_drop=('adjusted_drop','mean'),raw_drop=('raw_drop','mean'),
          duration_sec=('duration_sec','mean'),rate=('rate','mean'),runs=('runs','sum'),soc_bands=('soc_bands','sum')).reset_index())
    z=add_rank_reason(z,['condition'],'formation'); z.to_csv(OUT/'formation_ranking_by_wind_distance_balanced.csv',index=False)
    return q,z

def position_tables(x):
    q=balanced(x,['formation','condition','distance','position'])
    # Rank positions inside exact formation × wind × distance comparisons.
    q['rank']=q.groupby(['formation','condition','distance']).adjusted_drop.rank(method='min').astype(int)
    q['gap_from_best']=q.adjusted_drop-q.groupby(['formation','condition','distance']).adjusted_drop.transform('min')
    # Same run duration applies to all positions; differences are discharge intensity after battery/SOC correction.
    q['empirical_reason']='same mission duration; difference is position-specific discharge intensity after battery/SOC correction'
    q.sort_values(['formation','condition','distance','rank']).to_csv(OUT/'position_ranking_by_formation_wind_distance.csv',index=False)
    overall=(q.groupby(['formation','position']).agg(adjusted_drop=('adjusted_drop','mean'),raw_drop=('raw_drop','mean'),
              rate=('rate','mean'),condition_distance_cells=('condition','size')).reset_index())
    overall['rank']=overall.groupby('formation').adjusted_drop.rank(method='min').astype(int)
    overall['gap_from_best']=overall.adjusted_drop-overall.groupby('formation').adjusted_drop.transform('min')
    overall.sort_values(['formation','rank']).to_csv(OUT/'position_ranking_within_formation_overall.csv',index=False)
    wind=(q.groupby(['formation','condition','position']).agg(adjusted_drop=('adjusted_drop','mean'),raw_drop=('raw_drop','mean'),rate=('rate','mean'),distances=('distance','nunique')).reset_index())
    wind['rank']=wind.groupby(['formation','condition']).adjusted_drop.rank(method='min').astype(int)
    wind['gap_from_best']=wind.adjusted_drop-wind.groupby(['formation','condition']).adjusted_drop.transform('min')
    wind.sort_values(['formation','condition','rank']).to_csv(OUT/'position_ranking_by_formation_and_wind.csv',index=False)
    return q,overall,wind

def plot_formation_rank(q):
    fig,axs=plt.subplots(2,3,figsize=(15,9),dpi=180,sharey=True)
    for ax,c in zip(axs.flat,CONDS):
        z=q[q.condition==c]
        for d,marker,ls in [(50,'o','-'),(75,'s','--')]:
            a=z[z.distance==d].sort_values('adjusted_drop')
            ax.plot(a.formation,a.adjusted_drop,marker=marker,ls='',ms=9,label=f'{d} cm')
            for _,r in a.iterrows(): ax.text(r.formation,r.adjusted_drop+.18,f'#{r["rank"]}',ha='center',fontsize=8)
        ax.axhline(0,color='#333',ls=':'); ax.set_title(c,loc='left',weight='bold'); ax.tick_params(axis='x',rotation=30)
        ax.set_ylabel('Extra battery use vs baseline (% points)'); ax.grid(axis='y',color='#e1e5e8')
    axs[0,0].legend(frameon=False); fig.suptitle('Formation ranking for every wind condition and distance',x=.05,ha='left',weight='bold',fontsize=16)
    fig.text(.05,.94,'Lower is better; rank numbers are calculated separately at 50 cm and 75 cm',color='#59636e')
    fig.tight_layout(rect=[0,0,1,.92]); fig.savefig(CH/'01_formation_ranking_all_wind_conditions.png',bbox_inches='tight'); plt.close(fig)

def plot_balanced_rank(z):
    pv=z.pivot(index='condition',columns='formation',values='rank').reindex(index=CONDS,columns=FORMS)
    val=z.pivot(index='condition',columns='formation',values='adjusted_drop').reindex(index=CONDS,columns=FORMS)
    ann=pv.copy().astype(object)
    for i in pv.index:
        for f in pv.columns:
            ann.loc[i,f]='—' if pd.isna(pv.loc[i,f]) else f'#{int(pv.loc[i,f])}\n{val.loc[i,f]:.1f}'
    fig,ax=plt.subplots(figsize=(10,6),dpi=180)
    sns.heatmap(val,annot=ann,fmt='',cmap='YlOrRd',linewidths=.7,cbar_kws={'label':'Extra use vs baseline'},ax=ax)
    ax.set_xlabel('Formation'); ax.set_ylabel('Wind condition'); ax.set_title('Distance-balanced formation lookup table',loc='left',weight='bold',fontsize=15,pad=18)
    fig.tight_layout(); fig.savefig(CH/'02_distance_balanced_formation_lookup.png',bbox_inches='tight'); plt.close(fig)

def plot_position_overall(overall):
    p=overall.pivot(index='formation',columns='position',values='adjusted_drop').reindex(index=FORMS,columns=[1,2,3,4,5])
    ranks=overall.pivot(index='formation',columns='position',values='rank').reindex(index=FORMS,columns=[1,2,3,4,5])
    ann=p.copy().astype(object)
    for f in p.index:
        for d in p.columns: ann.loc[f,d]=f'#{int(ranks.loc[f,d])}\n{p.loc[f,d]:.1f}'
    fig,ax=plt.subplots(figsize=(10,6),dpi=180)
    sns.heatmap(p,annot=ann,fmt='',center=0,cmap='RdBu_r',linewidths=.7,cbar_kws={'label':'Extra use vs baseline'},ax=ax)
    ax.set_xlabel('Position / drone number'); ax.set_ylabel('Formation')
    ax.set_title('Within-formation position ranking across observed conditions',loc='left',weight='bold',fontsize=15,pad=18)
    fig.tight_layout(); fig.savefig(CH/'03_position_ranking_within_each_formation.png',bbox_inches='tight'); plt.close(fig)

def plot_position_wind(wind):
    for f in FORMS:
        z=wind[wind.formation==f]
        p=z.pivot(index='condition',columns='position',values='adjusted_drop').reindex(index=CONDS,columns=[1,2,3,4,5])
        rk=z.pivot(index='condition',columns='position',values='rank').reindex(index=CONDS,columns=[1,2,3,4,5])
        ann=p.copy().astype(object)
        for c in p.index:
            for d in p.columns: ann.loc[c,d]='—' if pd.isna(p.loc[c,d]) else f'#{int(rk.loc[c,d])}\n{p.loc[c,d]:.1f}'
        fig,ax=plt.subplots(figsize=(10,6),dpi=180)
        sns.heatmap(p,annot=ann,fmt='',center=0,cmap='RdBu_r',linewidths=.7,cbar_kws={'label':'Extra use vs baseline'},ax=ax)
        ax.set_xlabel('Position / drone number'); ax.set_ylabel('Wind condition')
        ax.set_title(f'{f}: position ranking changes by wind condition',loc='left',weight='bold',fontsize=15,pad=18)
        fig.tight_layout(); fig.savefig(CH/f'04_position_ranking_by_wind_{f}.png',bbox_inches='tight'); plt.close(fig)

def write_text(q,z,overall,wind):
    lines=['# Conditional ranking digest','',
      'Metric: SOC-adjusted extra battery percentage points versus the matching independent-flight battery baseline. Lower is better.',
      'Repeated runs are collapsed inside condition × starting-SOC strata. B12 is treated as B15.','']
    for c in CONDS:
        lines += [f'## {c}']
        for d in [50,75]:
            a=q[(q.condition==c)&(q.distance==d)].sort_values('rank')
            if len(a): lines.append(f"- {d} cm: "+' < '.join(f"{r.formation} ({r.adjusted_drop:.2f}, {r.empirical_reason})" for _,r in a.iterrows()))
        lines.append('')
    lines += ['# Position rankings by formation','']
    for f in FORMS:
        a=overall[overall.formation==f].sort_values('rank')
        lines.append(f"- {f}: "+' < '.join(f"position {int(r.position)} ({r.adjusted_drop:.2f})" for _,r in a.iterrows()))
    (OUT/'ranking_digest.md').write_text('\n'.join(lines))

def main():
    x=load(); q,z=formation_tables(x); exact,overall,wind=position_tables(x)
    plot_formation_rank(q); plot_balanced_rank(z); plot_position_overall(overall); plot_position_wind(wind); write_text(q,z,overall,wind)
    print('formation comparisons',len(q),'distance-balanced',len(z),'position comparisons',len(exact),'charts',len(list(CH.glob('*.png'))))
if __name__=='__main__': main()
