"""Recover per-drone movement onset/arrival from coordination trajectories."""
from pathlib import Path
import glob
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]; DB=ROOT/'db_copy_for_cleaning'; OUT=ROOT/'swarm_analysis'/'strategy_diagnostics'
OUT.mkdir(parents=True,exist_ok=True)

def first_sustained(t,progress,threshold,n=4):
    hit=np.asarray(progress)>=threshold
    ok=np.convolve(hit.astype(int),np.ones(n,dtype=int),mode='same')>=n
    return float(np.asarray(t)[np.argmax(ok)]) if ok.any() else np.nan

def main():
    clean=pd.read_csv(ROOT/'swarm_analysis'/'clean_swarm_drone_rows.csv',low_memory=False)
    valid=set(zip(clean.experiment_id.astype(str),clean.csv_run_id.astype(str)))
    rows=[]
    for f in glob.glob(str(DB/'**/*_all_coordination.csv'),recursive=True):
        try:
            x=pd.read_csv(f,usecols=['run_id','experiment_id','formation','wind_direction','wind_speed','inter_drone_distance_cm','drone_name','takeoff_order','node_elapsed_time','X_global','Y_global','node_forward_distance_cm'],low_memory=False)
        except Exception: continue
        if x.empty or (str(x.experiment_id.iloc[0]),str(x.run_id.iloc[0])) not in valid: continue
        for dn,g in x.groupby('drone_name'):
            g=g.sort_values('node_elapsed_time').dropna(subset=['node_elapsed_time','X_global','Y_global'])
            if len(g)<20: continue
            t=g.node_elapsed_time.to_numpy(); xy=g[['X_global','Y_global']].to_numpy(float)
            start=np.nanmedian(xy[t<=min(1,t.max())],axis=0); end=np.nanmedian(xy[t>=max(t.max()-1,0)],axis=0)
            vec=end-start; dist=np.linalg.norm(vec)
            if dist<50: continue
            progress=np.clip(((xy-start)@vec)/(dist**2),-0.2,1.2)
            onset=first_sustained(t,progress,.05); arrival=first_sustained(t,progress,.90)
            rows.append(dict(run_id=str(g.run_id.iloc[0]),experiment_id=str(g.experiment_id.iloc[0]),formation=g.formation.iloc[0],
              wind_direction=g.wind_direction.iloc[0],wind_speed=g.wind_speed.iloc[0],distance=g.inter_drone_distance_cm.iloc[0],
              commanded_forward_cm=g.node_forward_distance_cm.iloc[0],drone_name=dn,position=int(g.takeoff_order.iloc[0]),
              observed_displacement_cm=dist,movement_onset_sec=onset,arrival_90_sec=arrival,
              movement_duration_sec=arrival-onset if np.isfinite(onset+arrival) else np.nan,total_log_duration_sec=t.max()))
    d=pd.DataFrame(rows); d.to_csv(OUT/'per_drone_motion_timing.csv',index=False)
    run=(d.groupby(['experiment_id','run_id','formation','wind_direction','wind_speed','distance','commanded_forward_cm'])
       .agg(drones=('drone_name','nunique'),first_onset_sec=('movement_onset_sec','min'),last_onset_sec=('movement_onset_sec','max'),
            onset_stagger_sec=('movement_onset_sec',lambda s:s.max()-s.min()),mean_wait_before_motion_sec=('movement_onset_sec','mean'),
            mean_active_motion_sec=('movement_duration_sec','mean'),last_arrival_sec=('arrival_90_sec','max'),
            mean_observed_displacement_cm=('observed_displacement_cm','mean')).reset_index())
    run.to_csv(OUT/'run_motion_strategy_summary.csv',index=False)
    summary=(run.groupby(['formation','commanded_forward_cm']).agg(runs=('run_id','size'),median_stagger_sec=('onset_stagger_sec','median'),
       median_wait_sec=('mean_wait_before_motion_sec','median'),median_active_motion_sec=('mean_active_motion_sec','median'),
       median_last_arrival_sec=('last_arrival_sec','median')).reset_index())
    summary.to_csv(OUT/'formation_strategy_summary.csv',index=False)
    print(summary.to_string(index=False)); print('runs',len(run),'drones',len(d))
if __name__=='__main__': main()
