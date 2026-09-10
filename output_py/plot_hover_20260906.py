"""Compare six observed discharge traces; run from any working directory.
Chart contract: static step-line comparison, elapsed minutes vs actual SOC;
2026-09-06, one nonempty run per battery, all samples through first <=10%.
Five categorical hues plus neutral, distinct markers; PNG/SVG QA output.
"""
from pathlib import Path
import csv
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/hover-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output_graph'
ORDER = ['B10', 'B11', 'B12', 'B13', 'B14', 'B06']
COLORS = ['#2563A6', '#B58A16', '#D16B32', '#858B36', '#BE628A', '#40464E']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
runs = {}
for p in sorted((ROOT / 'database/baselines').glob('*/*hover*20260906*timeseries.csv')):
    rows = list(csv.DictReader(p.open()))
    if not rows or rows[0]['battery_id'] not in ORDER:
        continue
    b = rows[0]['battery_id']
    # Use the requested drone5 B12 replacement run, excluding the drone2 run.
    if b == 'B12' and (rows[0]['drone_name'] != 'drone_5' or rows[0]['run_id'] != '20260906_182131'):
        continue
    assert b not in runs, f'Multiple nonempty runs for {b}'
    assert all(r['wind_speed'] == 'lv1' for r in rows)
    end = next(i for i, r in enumerate(rows) if float(r['battery']) <= 10)
    rows = rows[:end+1]
    t0 = float(rows[0]['node_elapsed_time'])
    ts = [float(r['node_elapsed_time'])-t0 for r in rows]
    ys = [float(r['battery']) for r in rows]
    assert all(a < b for a,b in zip(ts, ts[1:]))
    assert all(0 <= y <= 100 for y in ys)
    runs[b] = (ts, ys, p, rows)
assert set(runs) == set(ORDER)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12, 'axes.labelcolor':'#252A30', 'text.color':'#252A30', 'axes.edgecolor':'#838890'})
fig, ax = plt.subplots(figsize=(12,7.5))
fig.subplots_adjust(left=.09, right=.97, bottom=.19, top=.77)
fig.text(.09,.94,'Hover battery discharge comparison',fontsize=22,weight='bold')
fig.text(.09,.895,'6 September 2026  |  Recorded wind level: lv1  |  Six battery runs',fontsize=12,color='#555C65')
summary=[]
for b,c,m in zip(ORDER,COLORS,MARKERS):
    ts,ys,p,rows = runs[b]
    duration = ts[-1]
    label = f'{b} / drone5' if b == 'B12' else b
    ax.step([t/60 for t in ts],ys,where='post',color=c,lw=1.9,marker=m,markevery=65,ms=4,mfc='white',label=f'{label}  ({duration/60:.2f} min)')
    ax.scatter([duration/60],[ys[-1]],c=c,s=24,zorder=5)
    summary.append({'battery_id':b,'start_soc_percent':ys[0],'end_soc_percent':ys[-1],'seconds_to_first_10_percent':round(duration,3),'minutes_to_first_10_percent':round(duration/60,4),'samples':len(ts),'source':str(p.relative_to(ROOT))})
ax.set(xlabel='Elapsed time from first recorded sample (min)',ylabel='Remaining battery (%)',xlim=(0,10.5),ylim=(0,103))
ax.set_xticks(range(11)); ax.set_yticks(range(0,101,10))
ax.grid(axis='y',color='#E5E7EB',lw=.7)
ax.spines[['top','right']].set_visible(False)
ax.axhline(10,color='#777D85',ls='--',lw=.8,zorder=0)
ax.legend(loc='lower left',bbox_to_anchor=(0,1.015),ncol=3,frameon=False,fontsize=11,columnspacing=2.2,borderaxespad=0)
fig.text(.09,.085,'Actual SOC, without smoothing. B13 starts at 99%; all other batteries start at 100%.',fontsize=10,color='#555C65')
fig.text(.09,.055,'Includes takeoff samples; each trace ends at its first 10% reading. Source: database/baselines, 20260906 runs.',fontsize=10,color='#555C65')
OUT.mkdir(exist_ok=True)
for ext in ['png','svg']:
    fig.savefig(OUT / f'hover_battery_comparison_20260906.{ext}',dpi=220,facecolor='white')
with (OUT / 'hover_battery_comparison_20260906_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
for r in summary: print(r['battery_id'],r['start_soc_percent'],r['seconds_to_first_10_percent'])
