"""Unwrap repeated mission-pad global coordinates without changing raw logs.

Mission pad IDs 1..8 are reused along long lanes. When a detector switches to a
reused ID, the logged global origin can jump by several metres. This script adds
continuous X/Y columns to sidecar CSVs and records every correction event.
"""
from pathlib import Path
import glob
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]; DB=ROOT/'db_copy_for_cleaning'; OUT=ROOT/'swarm_analysis'/'coordinate_unwrapped'
OUT.mkdir(parents=True,exist_ok=True); THRESHOLD_CM=100.0

def unwrap(values):
    raw=pd.to_numeric(values,errors='coerce').to_numpy(float); corr=np.full(len(raw),np.nan); offsets=np.zeros(len(raw)); events=[]; off=0.0
    last_raw=np.nan; last_corr=np.nan
    for i,v in enumerate(raw):
        if not np.isfinite(v): corr[i]=np.nan; offsets[i]=off; continue
        if np.isfinite(last_raw) and abs(v-last_raw)>THRESHOLD_CM:
            before=v+off; off += last_corr-before
            events.append((i,last_raw,v,v-last_raw,off))
        corr[i]=v+off; offsets[i]=off; last_raw=v; last_corr=corr[i]
    return corr,offsets,events

def main():
    all_events=[]; validation=[]; corrected_files=0
    for f in glob.glob(str(DB/'**/*_all_coordination.csv'),recursive=True):
        try:x=pd.read_csv(f,low_memory=False)
        except Exception:continue
        if not {'drone_name','node_elapsed_time','X_global','Y_global'}.issubset(x.columns):continue
        x['X_global_unwrapped']=np.nan; x['Y_global_unwrapped']=np.nan
        x['X_unwrap_offset_cm']=0.0; x['Y_unwrap_offset_cm']=0.0; file_has=False
        for dn,idx in x.groupby('drone_name').groups.items():
            ordered=x.loc[idx].sort_values('node_elapsed_time'); oi=ordered.index
            for axis in ['X','Y']:
                c,o,ev=unwrap(ordered[f'{axis}_global'])
                x.loc[oi,f'{axis}_global_unwrapped']=c; x.loc[oi,f'{axis}_unwrap_offset_cm']=o
                for pos,rb,ra,jump,off in ev:
                    row=ordered.iloc[pos]; prev=ordered.iloc[max(0,pos-1)]; file_has=True
                    all_events.append(dict(experiment_id=row.get('experiment_id',''),run_id=row.get('run_id',''),drone_name=dn,axis=axis,
                      node_elapsed_time=row.node_elapsed_time,phase=row.get('phase',''),mid_before=prev.get('mid',np.nan),mid_after=row.get('mid',np.nan),
                      raw_before=rb,raw_after=ra,raw_jump_cm=jump,applied_offset_cm=off,source_file=str(Path(f).relative_to(ROOT))))
            first=ordered.iloc[0]; last=ordered.iloc[-1]
            validation.append(dict(experiment_id=first.get('experiment_id',''),run_id=first.get('run_id',''),drone_name=dn,
              raw_y_displacement=last.Y_global-first.Y_global,
              corrected_y_displacement=x.loc[oi[-1],'Y_global_unwrapped']-x.loc[oi[0],'Y_global_unwrapped'],
              correction_events=sum(1 for e in all_events if e['experiment_id']==first.get('experiment_id','') and e['run_id']==first.get('run_id','') and e['drone_name']==dn)))
        if file_has:
            rel=Path(f).relative_to(DB); dest=OUT/rel.parent/(rel.stem+'_unwrapped.csv'); dest.parent.mkdir(parents=True,exist_ok=True)
            x.to_csv(dest,index=False); corrected_files+=1
    pd.DataFrame(all_events).to_csv(OUT/'coordinate_correction_events.csv',index=False)
    pd.DataFrame(validation).to_csv(OUT/'coordinate_unwrap_validation.csv',index=False)
    print('corrected sidecar files',corrected_files,'events',len(all_events))
if __name__=='__main__': main()
