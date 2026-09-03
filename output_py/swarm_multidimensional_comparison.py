"""Conditional swarm comparisons against five-drone virtual single-flight baseline."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'swarm_analysis'/'multidimensional'; CH=OUT/'charts'
OUT.mkdir(parents=True,exist_ok=True); CH.mkdir(parents=True,exist_ok=True)
FORMS=['front','vee','diamond','echalon','column']; DIRS=['head','side','tail']; POS=[1,2,3,4,5]
COL={'front':'#2878B5','vee':'#D99A22','diamond':'#D9534F','echalon':'#658B38','column':'#A14F86'}

def data():
    x=pd.read_csv(ROOT/'swarm_analysis'/'clean_swarm_drone_rows.csv',low_memory=False)
    x=x[x.csv_battery_hover_start.between(40,80)].copy()
    x['soc_bin']=pd.cut(x.csv_battery_hover_start,[40,50,60,70,81],right=False,labels=['40–49','50–59','60–69','70–80'])
    x['position']=x.csv_takeoff_order.astype(int)
    x['baseline_battery']=x.csv_battery_id.replace({'B12':'B15'})
    b=pd.read_csv(ROOT/'analysis_outputs'/'initial_baseline_quality.csv',low_memory=False)
    b=b[(b['mode']=='head_forward_250')&b.baseline_wind_level.isna()&b.battery_id.isin(['B10','B11','B13','B14','B15'])]
    b=b[b.battery_drop.notna()&b.node_duration_sec.gt(0)&b.battery_hover_start.between(40,85)].copy()
    models={}
    for bat,g in b.groupby('battery_id'):
        # Linear SOC correction within each battery; robust enough for sparse integer telemetry.
        models[bat]=np.polyfit(g.battery_hover_start,g.battery_drop,1)
    x['baseline_expected_drop']=[float(np.polyval(models[b],s)) for b,s in zip(x.baseline_battery,x.csv_battery_hover_start)]
    x['excess_vs_baseline']=x.csv_battery_drop-x.baseline_expected_drop
    return x,b,models

def collapse(x, dims, metric):
    # Repeats first become one median per condition/SOC cell, so front_50 cannot dominate.
    return (x.groupby(dims+['soc_bin'],observed=True)[metric].median().reset_index()
             .groupby(dims,observed=True)[metric].mean().reset_index())

def baseline_chart(b,models):
    fig,ax=plt.subplots(figsize=(9.5,6),dpi=180); grid=np.linspace(45,80,100)
    mapping=[('drone1','B11'),('drone2','B10'),('drone3','B13'),('drone4','B14'),('drone5','B15/B12')]
    for i,(dr,lab) in enumerate(mapping):
        bat='B15' if lab=='B15/B12' else lab; g=b[b.battery_id==bat]
        ax.scatter(g.battery_hover_start,g.battery_drop,s=18,alpha=.25)
        ax.plot(grid,np.polyval(models[bat],grid),lw=2.5,label=f'{dr} · {lab}')
    ax.set_xlabel('Starting battery (%)'); ax.set_ylabel('Battery used over 250 cm (% points)')
    fig.suptitle('Virtual five-drone baseline: independent 250 cm forward flights, no wind',x=.08,y=.98,ha='left',weight='bold',fontsize=14)
    fig.text(.08,.925,'B12 is assumed to follow the B15 discharge curve',color='#59636e')
    ax.legend(frameon=False,ncol=2); ax.grid(color='#e1e5e8'); ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(rect=[0,0,1,.90]); fig.savefig(CH/'01_virtual_five_drone_baseline.png',bbox_inches='tight'); plt.close(fig)

def head_lv1(x):
    q=collapse(x[(x.wind_direction_short=='head')&(x.wind_level==1)],['formation','distance','position'],'excess_vs_baseline')
    fig,axs=plt.subplots(1,2,figsize=(13,6),dpi=180,sharey=True)
    for ax,d in zip(axs,[50,75]):
        z=q[q.distance==d]
        for f in FORMS:
            a=z[z.formation==f].set_index('position').reindex(POS)
            ax.plot(POS,a.excess_vs_baseline,marker='o',lw=2,label=f,color=COL[f])
        ax.axhline(0,color='#222',ls='--'); ax.set_title(f'Head wind lv1 · {d} cm',loc='left',weight='bold')
        ax.set_xticks(POS,[f'drone{i}' for i in POS]); ax.set_ylabel('Extra battery use vs single-flight baseline (% points)'); ax.grid(color='#e1e5e8')
    axs[1].legend(frameon=False,ncol=2); fig.suptitle('All formations under head wind level 1, by drone position',x=.06,ha='left',weight='bold',fontsize=15)
    fig.text(.06,.92,'SOC-adjusted; repeated trials collapsed within condition and SOC band',color='#59636e')
    fig.tight_layout(rect=[0,0,1,.90]); fig.savefig(CH/'02_head_lv1_all_formations_positions.png',bbox_inches='tight'); plt.close(fig)
    q.to_csv(OUT/'head_lv1_all_formations_positions.csv',index=False)

def formation_wind(x):
    q=collapse(x,['formation','distance','wind_direction_short','wind_level'],'excess_vs_baseline')
    fig,axs=plt.subplots(1,5,figsize=(18,5.2),dpi=180,sharey=True)
    for ax,f in zip(axs,FORMS):
        z=q[q.formation==f].copy(); z['condition']=z.wind_direction_short+' lv'+z.wind_level.astype(int).astype(str)
        p=z.pivot(index='condition',columns='distance',values='excess_vs_baseline').reindex([d+' lv'+str(l) for d in DIRS for l in [1,2]])
        sns.heatmap(p,annot=True,fmt='.1f',center=0,cmap='RdBu_r',linewidths=.5,cbar=f==FORMS[-1],ax=ax,vmin=-4,vmax=8)
        ax.set_title(f,weight='bold'); ax.set_xlabel('Distance (cm)'); ax.set_ylabel('Wind condition' if f=='front' else '')
    fig.suptitle('Within each formation: wind condition and spacing compared with independent-flight baseline',x=.04,ha='left',weight='bold',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.93]); fig.savefig(CH/'03_each_formation_wind_distance_heatmaps.png',bbox_inches='tight'); plt.close(fig)
    q.to_csv(OUT/'formation_wind_distance_summary.csv',index=False)

def position_panels(x):
    q=collapse(x,['formation','distance','wind_direction_short','wind_level','position'],'excess_vs_baseline')
    for f in FORMS:
        fig,axs=plt.subplots(1,2,figsize=(12,5.4),dpi=170,sharey=True)
        for ax,d in zip(axs,[50,75]):
            z=q[(q.formation==f)&(q.distance==d)].copy(); z['condition']=z.wind_direction_short+' lv'+z.wind_level.astype(int).astype(str)
            p=z.pivot(index='condition',columns='position',values='excess_vs_baseline').reindex([a+' lv'+str(l) for a in DIRS for l in [1,2]],columns=POS)
            sns.heatmap(p,annot=True,fmt='.1f',center=0,cmap='RdBu_r',vmin=-5,vmax=10,linewidths=.4,cbar=d==75,ax=ax)
            ax.set_title(f'{d} cm',loc='left',weight='bold'); ax.set_xlabel('Drone position'); ax.set_ylabel('Wind condition')
        fig.suptitle(f'{f}: position-specific battery use across wind conditions',x=.06,ha='left',weight='bold',fontsize=15)
        fig.text(.06,.92,'Values are extra percentage points versus that drone battery’s SOC-matched single-flight baseline',color='#59636e')
        fig.tight_layout(rect=[0,0,1,.90]); fig.savefig(CH/f'04_position_wind_{f}.png',bbox_inches='tight'); plt.close(fig)
    q.to_csv(OUT/'position_wind_distance_summary.csv',index=False)

def distance_comparison(x):
    q=collapse(x,['formation','wind_direction_short','wind_level','distance'],'excess_vs_baseline')
    fig,axs=plt.subplots(2,3,figsize=(14,9),dpi=180,sharey=True)
    for ax,(d,l) in zip(axs.flat,[(d,l) for d in DIRS for l in [1,2]]):
        z=q[(q.wind_direction_short==d)&(q.wind_level==l)]
        for f in FORMS:
            a=z[z.formation==f].set_index('distance').reindex([50,75])
            ax.plot([50,75],a.excess_vs_baseline,marker='o',lw=2,color=COL[f],label=f)
        ax.axhline(0,color='#222',ls='--'); ax.set_xticks([50,75]); ax.set_title(f'{d} wind · lv{l}',loc='left',weight='bold'); ax.grid(color='#e1e5e8')
        ax.set_xlabel('Inter-drone distance (cm)'); ax.set_ylabel('Extra use vs baseline')
    axs[0,2].legend(frameon=False,ncol=2); fig.suptitle('50 cm versus 75 cm under the same formation and wind condition',x=.05,ha='left',weight='bold',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(CH/'05_distance_comparison_same_formation_wind.png',bbox_inches='tight'); plt.close(fig)
    q.to_csv(OUT/'distance_comparison_summary.csv',index=False)

def main():
    x,b,m=data(); baseline_chart(b,m); head_lv1(x); formation_wind(x); position_panels(x); distance_comparison(x)
    x.to_csv(OUT/'swarm_rows_with_baseline_adjustment.csv',index=False)
    print('rows',len(x),'baseline',len(b),'charts',len(list(CH.glob('*.png'))))
if __name__=='__main__': main()
