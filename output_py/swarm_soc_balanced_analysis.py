"""SOC- and condition-balanced swarm battery comparison.

Raw data are read only from db_copy_for_cleaning. Repeated runs are collapsed
within formation × condition × starting-SOC strata before formations are compared.
"""
from pathlib import Path
import json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]
DB=ROOT/'db_copy_for_cleaning'
OUT=ROOT/'swarm_analysis'/'soc_balanced'
CH=OUT/'charts'
OUT.mkdir(parents=True,exist_ok=True); CH.mkdir(parents=True,exist_ok=True)
FORMS=['front','vee','diamond','echalon','column']
COL={'front':'#2878B5','vee':'#D99A22','diamond':'#D9534F','echalon':'#658B38','column':'#A14F86'}
SOC_LABELS=['40–49','50–59','60–69','70–79']

def load_rows():
    x=pd.read_csv(ROOT/'swarm_analysis'/'clean_swarm_drone_rows.csv',low_memory=False)
    x=x[x.csv_battery_hover_start.between(40,79)].copy()
    x['soc_bin']=pd.cut(x.csv_battery_hover_start,[40,50,60,70,80],right=False,labels=SOC_LABELS)
    x['rate']=x.csv_battery_drop/x.csv_node_duration_sec*60
    return x

def matched_strata(x):
    keys=['distance','wind_direction_short','wind_level','soc_bin']
    # Collapse repeated experiments first: one median per formation/condition/SOC stratum.
    cells=(x.groupby(['formation']+keys,observed=True)
             .agg(n_drone_obs=('rate','size'),n_runs=('csv_run_id','nunique'),
                  median_rate=('rate','median'),median_drop=('csv_battery_drop','median'),
                  median_duration=('csv_node_duration_sec','median'))
             .reset_index())
    common=(cells.groupby(keys,observed=True).formation.nunique().reset_index(name='n_forms'))
    common=common[common.n_forms==5][keys]
    m=cells.merge(common,on=keys,how='inner')
    return cells,m,keys

def summaries(x,m,keys):
    overall=(m.groupby('formation',observed=True)
      .agg(matched_strata=('median_rate','size'),soc_condition_balanced_rate=('median_rate','mean'),
           soc_condition_balanced_drop=('median_drop','mean'),
           soc_condition_balanced_duration=('median_duration','mean'),
           median_stratum_rate=('median_rate','median'))
      .reindex(FORMS).reset_index())
    # Sensitivity check using batteries present in every run (avoids B12/B15 imbalance).
    core=x[x.csv_battery_id.isin(['B10','B11','B13','B14'])]
    _,cm,_=matched_strata(core)
    core_summary=(cm.groupby('formation').agg(core4_balanced_rate=('median_rate','mean'),
                                              core4_balanced_drop=('median_drop','mean')).reindex(FORMS).reset_index())
    overall=overall.merge(core_summary,on='formation',how='left')
    bysoc=(m.groupby(['formation','soc_bin'],observed=True)
            .agg(strata=('median_rate','size'),balanced_rate=('median_rate','mean')).reset_index())
    return overall,bysoc,core

def plot_soc_effect(x):
    # Equalize formation-condition cells within each SOC band.
    c=(x.groupby(['formation','distance','wind_direction_short','wind_level','soc_bin'],observed=True)
         .rate.median().reset_index())
    s=(c.groupby('soc_bin',observed=True).rate.agg(['mean','median','count']).reset_index())
    fig,ax=plt.subplots(figsize=(9,5.8),dpi=180)
    ax.plot(s.soc_bin.astype(str),s['mean'],marker='o',lw=2.5,color='#273746')
    for i,r in s.iterrows(): ax.text(i,r['mean']+.35,f"{r['mean']:.1f}",ha='center',weight='bold')
    ax.set_ylim(0,max(s['mean'])*1.25); ax.set_ylabel('Discharge rate (% points/min)')
    ax.set_xlabel('Battery level at node start')
    ax.set_title('Lower starting SOC is associated with faster observed discharge',loc='left',weight='bold',fontsize=15)
    ax.text(0,1.02,'Each formation × wind × spacing cell receives equal weight within each SOC band',transform=ax.transAxes,color='#59636e')
    ax.grid(axis='y',color='#dfe4e8'); ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(); fig.savefig(CH/'01_start_soc_effect.png',bbox_inches='tight'); plt.close(fig)

def plot_balanced(overall):
    z=overall.copy(); pos=np.arange(len(z)); w=.34
    fig,ax=plt.subplots(figsize=(10,6.2),dpi=180)
    ax.bar(pos-w/2,z.soc_condition_balanced_drop,w,label='All 5 drones',color=[COL[f] for f in z.formation],edgecolor='#37434d')
    ax.bar(pos+w/2,z.core4_balanced_drop,w,label='Core batteries B10/B11/B13/B14',color='white',edgecolor=[COL[f] for f in z.formation],linewidth=2)
    for i,r in z.iterrows():
        ax.text(i-w/2,r.soc_condition_balanced_drop+.18,f'{r.soc_condition_balanced_drop:.2f}',ha='center',fontsize=8)
    ax.set_xticks(pos,z.formation); ax.set_ylim(0,max(z.soc_condition_balanced_drop.max(),z.core4_balanced_drop.max())*1.25)
    ax.set_ylabel('SOC- and condition-balanced battery use (% points/node)')
    fig.suptitle('Formation comparison after controlling starting SOC and repeat imbalance',x=.08,y=.98,ha='left',weight='bold',fontsize=15)
    fig.text(.08,.925,'25 matched condition × SOC strata; repeated front_50 trials are collapsed within their stratum',color='#59636e')
    ax.legend(frameon=False); ax.grid(axis='y',color='#dfe4e8'); ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(rect=[0,0,1,.90]); fig.savefig(CH/'02_soc_condition_balanced_formation.png',bbox_inches='tight'); plt.close(fig)

def plot_heatmap(bysoc):
    p=bysoc.pivot(index='formation',columns='soc_bin',values='balanced_rate').reindex(index=FORMS,columns=SOC_LABELS)
    fig,ax=plt.subplots(figsize=(9,5.8),dpi=180)
    sns.heatmap(p,annot=True,fmt='.1f',cmap='YlOrBr',linewidths=.7,cbar_kws={'label':'% points/min'},ax=ax)
    ax.set_xlabel('Starting SOC band'); ax.set_ylabel('Formation')
    ax.set_title('Direct comparison within the same starting-SOC range',loc='left',weight='bold',fontsize=15,pad=18)
    fig.tight_layout(); fig.savefig(CH/'03_formation_by_start_soc.png',bbox_inches='tight'); plt.close(fig)

def load_timeseries(clean):
    valid=set(zip(clean.experiment_id.astype(str),clean.csv_run_id.astype(str),clean.csv_drone_name.astype(str)))
    frames=[]
    for f in glob.glob(str(DB/'**/*_all_battery_timeseries.csv'),recursive=True):
        try: q=pd.read_csv(f,usecols=['run_id','experiment_id','formation','drone_name','node_elapsed_time','battery','battery_start'])
        except Exception: continue
        q['run_id']=q.run_id.astype(str)
        keep=[(str(a),str(b),str(c)) in valid for a,b,c in zip(q.experiment_id,q.run_id,q.drone_name)]
        q=q[keep]
        if len(q): frames.append(q)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def plot_curves(ts):
    ts=ts[ts.battery_start.between(40,79)].copy()
    ts['soc_bin']=pd.cut(ts.battery_start,[40,50,60,70,80],right=False,labels=SOC_LABELS)
    max_t=ts.groupby(['experiment_id','run_id','drone_name']).node_elapsed_time.transform('max')
    ts=ts[max_t>0].copy(); ts['progress']=(ts.node_elapsed_time/max_t*20).round()/20
    ts['drop']=ts.battery_start-ts.battery
    g=(ts.groupby(['formation','soc_bin','experiment_id','run_id','drone_name','progress'],observed=True).drop.median().reset_index()
       .groupby(['formation','soc_bin','progress'],observed=True).drop.median().reset_index())
    fig,axs=plt.subplots(2,2,figsize=(12,8.5),dpi=180,sharex=True,sharey=True)
    for ax,b in zip(axs.flat,SOC_LABELS):
        for f in FORMS:
            q=g[(g.soc_bin==b)&(g.formation==f)]
            ax.plot(q.progress*100,q['drop'],lw=2,color=COL[f],label=f)
        ax.set_title(f'Starting SOC {b}%',loc='left',weight='bold'); ax.grid(color='#e1e5e8'); ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('Mission progress (%)'); ax.set_ylabel('Battery used (% points)')
    axs[0,1].legend(frameon=False,ncol=2)
    fig.suptitle('Battery-use curves compared within the same starting-SOC band',x=.07,ha='left',weight='bold',fontsize=16)
    fig.text(.07,.94,'Curves show the median drone trajectory; time is normalized from node start to node arrival',color='#59636e')
    fig.tight_layout(rect=[0,0,1,.92]); fig.savefig(CH/'04_direct_battery_curves_by_soc.png',bbox_inches='tight'); plt.close(fig)

def main():
    x=load_rows(); cells,m,keys=matched_strata(x); overall,bysoc,core=summaries(x,m,keys)
    cells.to_csv(OUT/'all_soc_strata.csv',index=False); m.to_csv(OUT/'matched_soc_condition_strata.csv',index=False)
    overall.to_csv(OUT/'soc_balanced_formation_summary.csv',index=False); bysoc.to_csv(OUT/'formation_by_soc_summary.csv',index=False)
    plot_soc_effect(x); plot_balanced(overall); plot_heatmap(bysoc)
    ts=load_timeseries(x); plot_curves(ts)
    print(overall.to_string(index=False)); print('matched strata',m[keys].drop_duplicates().shape[0],'timeseries rows',len(ts))

if __name__=='__main__': main()
