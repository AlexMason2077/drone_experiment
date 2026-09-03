"""Estimate pure 250 cm forward-flight battery use from swarm trajectories.

For every drone, crop from actual motion onset to 250 cm projected displacement.
Stationary time inside this window is detected from smoothed trajectory speed and
its expected hover consumption is subtracted. All comparisons remain against the
SOC- and battery-matched independent 250 cm forward-flight baseline.
"""
from pathlib import Path
import glob, re, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; DB=ROOT/'db_copy_for_cleaning'; OUT=ROOT/'swarm_analysis'/'pure_forward'; CH=OUT/'charts'
OUT.mkdir(parents=True,exist_ok=True); CH.mkdir(parents=True,exist_ok=True)
FORMS=['front','vee','diamond','echalon','column']; DIRS=['head','side','tail']; COL=dict(zip(FORMS,['#2878B5','#D99A22','#D9534F','#658B38','#A14F86']))

def baseline_models():
    b=pd.read_csv(ROOT/'analysis_outputs'/'initial_baseline_quality.csv',low_memory=False)
    b=b[(b['mode']=='head_forward_250')&b.baseline_wind_level.isna()&b.battery_id.isin(['B10','B11','B13','B14','B15'])]
    b=b[b.battery_drop.notna()&b.battery_hover_start.between(40,85)]
    return {k:np.polyfit(g.battery_hover_start,g.battery_drop,1) for k,g in b.groupby('battery_id')}

def hover_rate_model():
    """SOC-shaped hover rate with battery-specific multiplicative adjustment."""
    s=pd.read_csv(ROOT/'output_graph'/'hover_battery_runs_summary.csv')
    rows=[]
    for r in s[s.status.eq('included')].itertuples():
        try: x=pd.read_csv(r.source_file).sort_values('elapsed_time')
        except Exception: continue
        for lo in [40,50,60,70]:
            hi=lo+10; t0=x[x.battery<=hi].elapsed_time.min(); t1=x[x.battery<=lo].elapsed_time.min()
            if pd.notna(t0) and pd.notna(t1) and t1>t0: rows.append((r.battery_id,lo+5,600/(t1-t0)))
    z=pd.DataFrame(rows,columns=['battery','soc','rate'])
    # Winsorize within SOC band to suppress takeoff/telemetry boundary artifacts.
    for soc,g in z.groupby('soc'):
        lo,hi=g.rate.quantile([.1,.9]); z.loc[g.index,'rate']=g.rate.clip(lo,hi)
    profile=z.groupby('soc').rate.median().sort_index()
    battery=z.groupby('battery').rate.median(); factor=(battery/profile.median()).clip(.8,1.2).to_dict()
    def predict(battery_id,soc):
        base=float(np.interp(soc,profile.index.to_numpy(float),profile.to_numpy(float)))
        return base*factor.get(battery_id,1.0)
    return predict,profile,factor

def first_sustained(t,v,thr,n=5):
    hit=np.asarray(v)>=thr; ok=np.convolve(hit.astype(int),np.ones(n,dtype=int),mode='same')>=n
    return float(np.asarray(t)[np.argmax(ok)]) if ok.any() else np.nan

def unwrap_coordinate(values,threshold=100.0):
    raw=np.asarray(values,dtype=float); out=np.full(len(raw),np.nan); off=0.0; last_raw=np.nan; last_corr=np.nan
    for i,v in enumerate(raw):
        if not np.isfinite(v): continue
        if np.isfinite(last_raw) and abs(v-last_raw)>threshold: off += last_corr-(v+off)
        out[i]=v+off; last_raw=v; last_corr=out[i]
    return out

def battery_at(bt,t):
    i=np.searchsorted(bt.node_elapsed_time.to_numpy(),t,side='right')-1
    return float(bt.battery.iloc[max(0,min(i,len(bt)-1))])

def extract():
    clean=pd.read_csv(ROOT/'swarm_analysis'/'clean_swarm_drone_rows.csv',low_memory=False)
    valid={(str(r.experiment_id),str(r.csv_run_id),str(r.csv_drone_name)):r for r in clean.itertuples()}
    bm=baseline_models(); hover_predict,hover_profile,hover_factor=hover_rate_model(); rows=[]; excluded=[]
    coord_files=glob.glob(str(DB/'**/*_all_coordination.csv'),recursive=True)
    for cf in coord_files:
        bf=cf.replace('_all_coordination.csv','_all_battery_timeseries.csv')
        if not Path(bf).exists(): continue
        try:
            c=pd.read_csv(cf,usecols=['run_id','experiment_id','drone_name','phase','node_elapsed_time','X_global','Y_global'],low_memory=False)
            b=pd.read_csv(bf,usecols=['run_id','experiment_id','drone_name','node_elapsed_time','battery'],low_memory=False)
        except Exception: continue
        for dn,g in c.groupby('drone_name'):
            key=(str(g.experiment_id.iloc[0]),str(g.run_id.iloc[0]),str(dn)); meta=valid.get(key)
            if meta is None: continue
            g=g.sort_values('node_elapsed_time').dropna(subset=['node_elapsed_time','X_global','Y_global']); bt=b[b.drone_name==dn].sort_values('node_elapsed_time')
            reason=''
            if len(g)<30 or len(bt)<10: reason='insufficient trajectory/battery samples'
            if reason: excluded.append((*key,reason)); continue
            t=g.node_elapsed_time.to_numpy(float); commanded=float(meta.csv_node_forward_distance_cm)
            raw_xy=g[['X_global','Y_global']].to_numpy(float)
            unwrapped_xy=np.column_stack([unwrap_coordinate(g.X_global),unwrap_coordinate(g.Y_global)])
            def endpoints(arr):
                st=np.nanmedian(arr[t<=min(1,t.max())],axis=0); en=np.nanmedian(arr[t>=max(t.max()-1,0)],axis=0); return st,en,np.linalg.norm(en-st)
            raw_start,raw_end,raw_total=endpoints(raw_xy); unwrap_start,unwrap_end,unwrap_total=endpoints(unwrapped_xy)
            # Use unwrapped coordinates only when they improve agreement with the
            # known commanded distance; this avoids altering already-correct tracks.
            if abs(unwrap_total-commanded)+5 < abs(raw_total-commanded): xy,start,end,total=unwrapped_xy,unwrap_start,unwrap_end,unwrap_total
            else: xy,start,end,total=raw_xy,raw_start,raw_end,raw_total
            vec=end-start
            if total<180: excluded.append((*key,'observed displacement too short')); continue
            unit=vec/total; proj=(xy-start)@unit
            # Smooth marker noise before estimating active movement.
            ps=pd.Series(proj).rolling(11,center=True,min_periods=1).median().to_numpy()
            # Marker coordinates under/overshoot the pad-to-pad command. Calibrate the
            # observed final displacement to the commanded 250/300 cm distance.
            # Convert observed marker displacement onto the commanded-distance axis.
            pcmd=ps/total*commanded
            onset=first_sustained(t,pcmd,7.5,5); finish=first_sustained(t,pcmd,245.0,4)
            if not np.isfinite(onset+finish) or finish<=onset: excluded.append((*key,'cannot locate onset or 250 cm crossing')); continue
            # Recover active time segment-by-segment. The 10%-to-90% crossing
            # excludes pad waiting before/after each programmed movement segment.
            counts=[]
            for phase in g.phase.astype(str):
                m=re.search(r'_of_(\d+)',phase)
                if m: counts.append(int(m.group(1)))
            nseg=max(counts or [1]); seg=commanded/nseg; active=0.0
            for k in range(nseg):
                a=k*seg; z=(k+1)*seg
                span=z-a; ta=t[pcmd>=a+.1*span]; tz=t[pcmd>=a+.9*span]
                if len(ta) and len(tz): active+=max(0.0,float(tz[0]-ta[0]))
            # Full summary drop captures delayed 1%-step battery updates. Treat all
            # logged node time outside detected motion as wait/hover, then normalize
            # the net active-flight energy from 300 cm to 250 cm where necessary.
            full_duration=float(meta.csv_node_duration_sec); active=min(active,full_duration); stationary=max(0,full_duration-active)
            window=finish-onset
            bat0=battery_at(bt,onset); bat1=battery_at(bt,finish); observed=bat0-bat1
            bid=str(meta.csv_battery_id); base_bid='B15' if bid=='B12' else bid
            hover_rate=hover_predict(base_bid,float(meta.csv_battery_hover_start)); wait_drop=stationary*hover_rate/60
            full_drop=float(meta.csv_battery_drop); pure=(full_drop-wait_drop)*(250.0/commanded)
            expected=float(np.polyval(bm[base_bid],float(meta.csv_battery_hover_start))); excess=pure-expected
            rows.append(dict(experiment_id=key[0],run_id=key[1],drone_name=dn,position=int(meta.csv_takeoff_order),battery_id=bid,
              formation=meta.formation,distance=int(meta.distance),wind_direction=meta.wind_direction_short,wind_level=int(meta.wind_level),
              starting_soc_at_motion=bat0,battery_at_250cm=bat1,observed_window_drop=full_drop,motion_window_sec=window,
              active_forward_sec=active,stationary_wait_sec=stationary,estimated_wait_hover_drop=wait_drop,
              pure_forward_drop_250cm=pure,baseline_expected_drop_250cm=expected,excess_vs_baseline=excess,
              commanded_distance_cm=int(meta.csv_node_forward_distance_cm),observed_total_displacement_cm=total))
    d=pd.DataFrame(rows); ex=pd.DataFrame(excluded,columns=['experiment_id','run_id','drone_name','reason'])
    # Require all five drones so a run is comparable.
    good=d.groupby(['experiment_id','run_id']).drone_name.nunique(); keys=set(good[good==5].index)
    d=d[d.apply(lambda r:(r.experiment_id,r.run_id) in keys,axis=1)].copy()
    d['soc_bin']=pd.cut(d.starting_soc_at_motion,[35,45,55,65,75,86],right=False)
    d.to_csv(OUT/'pure_forward_drone_rows.csv',index=False); ex.to_csv(OUT/'excluded_trajectory_rows.csv',index=False)
    return d

def balanced(d,dims,min_observations=5):
    cells=(d.groupby(dims+['soc_bin'],observed=True).agg(value=('excess_vs_baseline','median'),pure=('pure_forward_drop_250cm','median'),
       wait=('estimated_wait_hover_drop','median'),active=('active_forward_sec','median'),runs=('run_id','nunique'),observations=('drone_name','size')).reset_index())
    # A single battery in a sparse SOC bin must not receive the same weight as a
    # well-supported bin; require at least one full swarm-equivalent (5 drones).
    cells=cells[cells.observations>=min_observations]
    return (cells.groupby(dims,observed=True).agg(value=('value','mean'),pure=('pure','mean'),wait=('wait','mean'),active=('active','mean'),runs=('runs','sum')).reset_index())

def plot_conditions(d):
    q=balanced(d,['wind_direction','wind_level','distance','formation'])
    for wd in DIRS:
      for lv in [1,2]:
        fig,axs=plt.subplots(1,2,figsize=(12,5.5),dpi=180,sharey=True)
        for ax,dist in zip(axs,[50,75]):
            z=q[(q.wind_direction==wd)&(q.wind_level==lv)&(q.distance==dist)].set_index('formation').reindex(FORMS).dropna(subset=['value'])
            ax.bar(z.index,z.value,color=[COL[f] for f in z.index],edgecolor='#39434b')
            ax.axhline(0,color='#222',ls='--',lw=1.5,label='Independent-flight baseline')
            for i,r in enumerate(z.itertuples()): ax.text(i,r.value+(.15 if r.value>=0 else -.25),f'{r.value:.1f}',ha='center',va='bottom' if r.value>=0 else 'top')
            ax.set_title(f'{dist} cm spacing',loc='left',weight='bold'); ax.set_ylabel('Pure-forward extra use vs baseline (% points / 250 cm)'); ax.grid(axis='y',color='#e1e5e8'); ax.spines[['top','right']].set_visible(False)
        fig.suptitle(f'{wd} wind · level {lv}: pure 250 cm forward-flight comparison',x=.07,ha='left',weight='bold',fontsize=15)
        fig.text(.07,.92,'Waiting/hover consumption removed; every bar is compared with the matching single-drone baseline',color='#59636e')
        fig.tight_layout(rect=[0,0,1,.90]); fig.savefig(CH/f'formation_vs_baseline_{wd}_lv{lv}.png',bbox_inches='tight'); plt.close(fig)
    q.to_csv(OUT/'formation_condition_summary.csv',index=False)

def plot_positions(d):
    q=balanced(d,['formation','wind_direction','wind_level','distance','position'],min_observations=2)
    for f in FORMS:
        fig,axs=plt.subplots(1,2,figsize=(13,5.8),dpi=180,sharey=True)
        for ax,dist in zip(axs,[50,75]):
            z=q[(q.formation==f)&(q.distance==dist)]
            for wd,ls in zip(DIRS,['-','--',':']):
                for lv,marker in [(1,'o'),(2,'s')]:
                    a=z[(z.wind_direction==wd)&(z.wind_level==lv)].set_index('position').reindex([1,2,3,4,5])
                    ax.plot([1,2,3,4,5],a.value,ls=ls,marker=marker,lw=1.8,label=f'{wd} lv{lv}')
            ax.axhline(0,color='#222',lw=1.3); ax.set_title(f'{dist} cm spacing',loc='left',weight='bold'); ax.set_xticks([1,2,3,4,5]); ax.set_xlabel('Formation position'); ax.set_ylabel('Pure-forward extra use vs baseline'); ax.grid(color='#e1e5e8')
        axs[1].legend(frameon=False,ncol=2,fontsize=8); fig.suptitle(f'{f}: position-specific pure-forward battery use',x=.06,ha='left',weight='bold',fontsize=15)
        fig.text(.06,.92,'Each line is one wind condition; zero is the position battery’s independent-flight baseline',color='#59636e')
        fig.tight_layout(rect=[0,0,1,.90]); fig.savefig(CH/f'positions_vs_baseline_{f}.png',bbox_inches='tight'); plt.close(fig)
    q.to_csv(OUT/'position_condition_summary.csv',index=False)

def plot_wait_correction(d):
    q=d.groupby('formation').agg(window_drop=('observed_window_drop','median'),wait_drop=('estimated_wait_hover_drop','median'),pure=('pure_forward_drop_250cm','median'),wait_sec=('stationary_wait_sec','median')).reindex(FORMS)
    fig,ax=plt.subplots(figsize=(10,6),dpi=180); x=np.arange(len(q)); w=.34
    ax.bar(x-w/2,q.window_drop,w,label='Observed onset-to-250 cm drop',color='#9aa7b2')
    ax.bar(x+w/2,q.pure,w,label='After stationary-hover correction',color=[COL[f] for f in q.index])
    ax.set_xticks(x,q.index); ax.set_ylabel('Battery use (% points / 250 cm)'); ax.set_title('Removing waiting and stationary hover from the 250 cm flight window',loc='left',weight='bold',fontsize=15)
    ax.legend(frameon=False); ax.grid(axis='y',color='#e1e5e8'); ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(); fig.savefig(CH/'waiting_correction_by_formation.png',bbox_inches='tight'); plt.close(fig)
    q.reset_index().to_csv(OUT/'waiting_correction_summary.csv',index=False)

def main():
    d=extract(); plot_conditions(d); plot_positions(d); plot_wait_correction(d)
    print('complete five-drone pure-forward rows',len(d),'runs',d[['experiment_id','run_id']].drop_duplicates().shape[0],'charts',len(list(CH.glob('*.png'))))
    print(d.groupby('formation').agg(runs=('run_id','nunique'),median_wait=('stationary_wait_sec','median'),median_pure=('pure_forward_drop_250cm','median')).to_string())
if __name__=='__main__': main()
