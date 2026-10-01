"""Deterministic demonstration provider. No aircraft calls, no research-model claims."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import itertools
import json
import math
import time

DATA = Path(__file__).parent / 'data'
CONFIGS = json.loads((DATA/'configurations.json').read_text())
SETTINGS = json.loads((DATA/'settings.json').read_text())

# Scripted inputs for the second segment of the visual demonstration only.
# Consecutive intervals deliberately include both unchanged and changed conditions.
DEMO_SEGMENT_TWO_INPUTS = {
    1: ('Tailwind', 'Low', 4),
    2: ('Tailwind', 'Low', 4),
    3: ('Headwind', 'Low', 3),
    4: ('Sidewind', 'Low', 3),
    5: ('Headwind', 'High', 4),
    6: ('Sidewind', 'Low', 5),
    7: ('Sidewind', 'Low', 5),
    8: ('Tailwind', 'High', 3),
}

# Authored Demo-only rates for the interval-3 to interval-4 comparison.
# They illustrate a wind-driven switch from Vee to Diamond; neither factor is
# measured flight performance or a research-model estimate.
DEMO_SIDEWIND_VEE_50_MULTIPLIER = 1.747
DEMO_SIDEWIND_DIAMOND_MULTIPLIER = 1.03 / 1.12

def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()

def evaluate(config, batteries, arrival_s, wind, k, assignment):
    """Demo rate fixtures + stated charging formula; evaluate only supplied candidates."""
    fixture_rates = {'Front':[10.2,10.6,10.1,10.6,10.2], 'Column':[12.1,9.4,8.9,9.1,9.7],
      'Vee':[9.0,10.1,11.8,10.1,9.0], 'Echelon':[11.4,10.6,9.7,9.1,8.8], 'Diamond':[9.8,10.3,8.7,10.3,11.3]}
    factor={'Headwind':1.03,'Sidewind':1.12,'Tailwind':.92}[wind]
    # Scripted paper-demo scenario: the previously applied Vee is less efficient
    # under the new sidewind. This is an authored example, not measured data.
    if wind=='Sidewind' and config['id']=='vee-50':
        factor*=DEMO_SIDEWIND_VEE_50_MULTIPLIER
    if wind=='Sidewind' and config['id']=='diamond-75':
        factor*=DEMO_SIDEWIND_DIAMOND_MULTIPLIER
    # All rates and measurements here are authored DEMO values.
    consumed=[fixture_rates[config['formation']][int(assignment[f'D{i+1}'][1:])-1]*factor*arrival_s/60 for i in range(5)]
    arrivals=[max(1,b-d) for b,d in zip(batteries,consumed)]
    q=[max(0,SETTINGS['demo_charge_curve_minutes']/math.log(100)*math.log((100-a)/(100-SETTINGS['charge_target_pct'])))*60 for a in arrivals]
    if not 1<=k<=5: return None
    best=None
    for slots in itertools.product(range(k),repeat=5):
        if slots[0]!=0 or any(slots[i]>1+max(slots[:i]) for i in range(1,5)):continue
        loads=[0.0]*k
        for i,p in enumerate(slots):loads[p]+=q[i]
        if best is None or max(loads)<best[0]:best=(max(loads),slots)
    loads=[0.0]*k; schedule=[]
    for i,p in enumerate(best[1]):
        schedule.append(dict(drone_id=f'D{i+1}',group=p+1,wait_s=loads[p],charge_s=q[i],start_s=loads[p],end_s=loads[p]+q[i],status='Planned'))
        loads[p]+=q[i]
    flight_seconds=round(arrival_s);charging_seconds=round(best[0])
    return dict(arrival_s=flight_seconds,charging_s=charging_seconds,ready_s=flight_seconds+charging_seconds,schedule=schedule)

class DemoProvider:
    def __init__(self, clock=time.time):
        self.clock=clock; self.reset()
    def reset(self):
        self.started=self.clock();self.last_tick=self.started
        self.segment=1;self.interval=3;self.interval_elapsed=10.;self.intervals=SETTINGS['interval_count']
        self.period=SETTINGS['decision_period_s']
        self.phase='In transit';self.phase_at=None;self.auto=True;self.connection='Connected'
        self.wind='Headwind';self.level='Low';self.k=3;self.capacity=6
        self.batteries=[78,74,81,72,76];self.missing=set();self.delayed=False
        self.applied='front-50';self.assignment={f'D{i}':f'P{i}' for i in range(1,6)}
        self.revision=1;self.seq=0;self.events=[];self.history=[];self.pending=[]
        self.current=None;self.block_ack=False;self.inputs_changed=False;self.decision_due=None
        for interval in (1,2):
            self.interval=interval;self.interval_elapsed=0.
            self.wind,self.level,self.k=DEMO_SEGMENT_TWO_INPUTS[interval]
            at=self.started-(3-interval)*self.period
            self.create_decision('Interval boundary',at)
            prior=self.current
            prior.update(status='Applied',sent_at=iso(at+3),acknowledged_at=iso(at+6),
                         applied_at=iso(at+9),updated_at=iso(at+9),
                         feedback=[dict(drone_id=f'D{i}',status='Applied') for i in range(1,6)])
            self.applied=prior['recommended']['config_id']
            self.assignment=deepcopy(prior['recommended']['assignment'])
            self.revision+=1;self.pending=[]
        self.interval=3;self.interval_elapsed=10.
        self.wind,self.level,self.k=DEMO_SEGMENT_TWO_INPUTS[3]
        self.create_decision('Segment checkpoint',self.started,initial=True)
        self.event('Mission active','Current segment N1 → N2 · interval 3',self.started)
    def event(self,title,detail,at=None):
        self.events.insert(0,dict(id=f'E{len(self.events)+1}',title=title,detail=detail,at=iso(self.clock() if at is None else at)))
        self.events=self.events[:80]
    def remaining(self):
        if self.phase!='In transit':return 0.
        return max(0,(self.intervals-self.interval+1)*self.period-self.interval_elapsed)
    def create_decision(self,reason,now,initial=False,swapped=False):
        if self.phase!='In transit':return
        self.seq+=1;self.inputs_changed=False;self.decision_due=None
        if self.current and self.current['status'] not in ['Applied','Superseded']:
            self.current['status']='Superseded'
        identifier=f'DEC-{self.seq:04}'
        input_battery_levels={f'D{i+1}':None if f'D{i+1}' in self.missing else self.batteries[i] for i in range(5)}
        remaining_s=self.remaining()
        before_config_id=self.applied
        before_assignment=deepcopy(self.assignment)
        assignment=deepcopy(self.assignment)
        if swapped:assignment['D1'],assignment['D4']=assignment['D4'],assignment['D1']
        shapes={'Headwind':['vee-50','echelon-75','front-50'],'Sidewind':['diamond-75','front-75','vee-75'],'Tailwind':['column-75','echelon-50','front-75']}[self.wind]
        n=SETTINGS['candidate_limits_by_K'].get(str(self.k),0)
        candidates=[]
        if not self.missing:
            for rank,cid in enumerate(shapes[:n],1):
                cfg=next(x for x in CONFIGS if x['id']==cid)
                forecast=evaluate(cfg,self.batteries,remaining_s,self.wind,self.k,assignment)
                if forecast:candidates.append(dict(id=f'{identifier}-C{rank}',rank=rank,config_id=cid,assignment=deepcopy(assignment),**forecast))
        best=min(candidates,key=lambda x:x['ready_s']) if candidates else None
        comparison=None
        if best:
            before_config=next(x for x in CONFIGS if x['id']==before_config_id)
            before_forecast=evaluate(before_config,self.batteries,remaining_s,self.wind,self.k,before_assignment)
            if before_forecast:
                def side(config_id, positions, forecast):
                    return dict(config_id=config_id,assignment=deepcopy(positions),
                                arrival_s=forecast['arrival_s'],charging_s=forecast['charging_s'],ready_s=forecast['ready_s'])
                comparison=dict(before=side(before_config_id,before_assignment,before_forecast),
                                after=side(best['config_id'],best['assignment'],best),
                                saved_s=before_forecast['ready_s']-best['ready_s'],
                                input_as_of=iso(now),quality='Delayed demo inputs' if self.delayed else 'Demo estimate')
        dec=dict(id=identifier,sequence=self.seq,segment_id=f'S{self.segment+1}',interval_id=f'S{self.segment+1}-I{self.interval}',
          interval=self.interval,trigger=reason,input_as_of=iso(now),updated_at=iso(now),model_version=SETTINGS['model_version'],
          input_snapshot_id=f'INPUT-{self.seq:04}',input_battery_levels=input_battery_levels,
          rate_version='demo-rate-fixtures-v1',catalog_version='research-geometry-v1',
          wind=self.wind,level=self.level,k=self.k,candidate_limit=n,candidates=candidates,recommended=best,comparison=comparison,
          status='Applied' if initial else ('Calculating' if best else 'Unavailable'),command_id=f'CMD-{self.seq:04}' if best else None,
          sent_at=None,acknowledged_at=None,applied_at=None,ack_due_at=None,
          reason='Battery data missing' if self.missing else ('No setting for charging-pad count' if not best else None),
          feedback=[],target_node_id=f'N{self.segment+1}')
        if initial:
            # Initial demo has an observed executed front configuration, separate from the latest proposal.
            self.current=dec
            dec['status']='Recommended'
            self.pending=[(now+4,identifier,'Sent'),(now+8,identifier,'Acknowledged'),(now+11,identifier,'Applied')]
        elif best:
            self.pending.extend([(now+1,identifier,'Recommended'),(now+3,identifier,'Sent'),(now+6,identifier,'Acknowledged'),(now+9,identifier,'Applied')])
        self.current=dec;self.history.insert(0,dec)
        self.event('Decision requested',f'{identifier} · {reason}',now)
    def tick(self):
        now=self.clock();delta=max(0,now-self.last_tick);self.last_tick=now
        if self.phase=='In transit':
            self.interval_elapsed+=delta
            while self.interval_elapsed>=self.period and self.phase=='In transit':
                self.interval_elapsed-=self.period;self.interval+=1
                boundary_at=now-self.interval_elapsed
                if self.interval>self.intervals:
                    self.arrive(boundary_at)
                    break
                self.event('Next interval',f'Interval {self.interval} of {self.intervals}',boundary_at)
                if self.segment==1 and self.interval in DEMO_SEGMENT_TWO_INPUTS:
                    previous=(self.wind,self.level,self.k)
                    self.wind,self.level,self.k=DEMO_SEGMENT_TWO_INPUTS[self.interval]
                    if previous!=(self.wind,self.level,self.k):
                        self.event('Demo inputs changed',f'{self.wind} · {self.level}; {self.k} pads available',boundary_at)
                if self.auto:
                    elapsed=self.interval_elapsed
                    self.interval_elapsed=0.
                    self.create_decision('Interval boundary',boundary_at)
                    self.interval_elapsed=elapsed
            if self.auto and self.decision_due is not None and now>=self.decision_due:
                self.create_decision('Conditions updated',now)
        queued=self.pending;self.pending=[]
        for due,did,status in queued:
            if due>now:self.pending.append((due,did,status));continue
            dec=next((x for x in self.history if x['id']==did),None)
            if not dec or dec is not self.current or dec['status']=='Superseded':continue
            if self.block_ack and status in ['Acknowledged','Applied']:continue
            if self.phase!='In transit':continue
            dec['status']=status;dec['updated_at']=iso(due)
            if status=='Sent':dec['sent_at']=iso(due);dec['ack_due_at']=iso(due+8)
            if status=='Acknowledged':dec['acknowledged_at']=iso(due)
            if status=='Applied':
                dec['applied_at']=iso(due);self.applied=dec['recommended']['config_id'];self.assignment=deepcopy(dec['recommended']['assignment']);self.revision+=1
                dec['feedback']=[dict(drone_id=f'D{i}',status='Applied') for i in range(1,6)]
            self.event(status,f'{did} · {dec["command_id"]}',due)
        if self.phase=='At node' and now-self.phase_at>=3:
            self.phase='Charging';self.phase_at=now;self.event('Charging started','Demonstration execution feedback',now)
    def arrive(self,now=None):
        now=self.clock() if now is None else now
        self.phase='At node';self.phase_at=now;self.interval=self.intervals;self.interval_elapsed=self.period;self.pending=[];self.decision_due=None
        if self.current and self.current['status'] not in ['Applied','Unavailable']:self.current['status']='Superseded'
        self.event('Arrived at node',f'N{self.segment+1} · segment complete',now)
    def action(self,kind):
        self.tick();now=self.clock()
        if kind=='reset':self.reset()
        elif kind=='wind':
            self.wind={'Headwind':'Sidewind','Sidewind':'Tailwind','Tailwind':'Headwind'}[self.wind];self.level='High' if self.level=='Low' else 'Low';self.inputs_changed=True
            self.decision_due=now+2;self.event('Wind changed',f'{self.wind} · {self.level}; decision scheduled',now)
        elif kind=='pads':
            self.k={1:3,2:3,3:4,4:5,5:3}[self.k];self.inputs_changed=True;self.decision_due=now+2
            self.event('Charging availability changed',f'{self.k} available · capacity {self.capacity}',now)
        elif kind=='swap':self.create_decision('Position reassignment',now,swapped=True)
        elif kind=='interval':self.interval_elapsed=self.period;self.tick()
        elif kind=='await_ack':
            self.block_ack=True;self.create_decision('Acknowledgement scenario',now)
        elif kind=='release_ack':
            self.block_ack=False
            if self.current and self.current['status']=='Sent':self.pending.extend([(now+1,self.current['id'],'Acknowledged'),(now+3,self.current['id'],'Applied')])
        elif kind=='arrive':self.arrive(now)
        elif kind=='complete_charge':
            if self.phase not in ['Charging','At node']:raise ValueError('Reach a node first')
            self.phase='Ready at node';self.batteries=[99]*5;self.event('Charging completed','Five drones reached the 99% target',now)
        elif kind=='next_segment':
            if self.phase!='Ready at node':raise ValueError('Complete charging first')
            if self.segment==3:self.phase='Completed'
            else:
                self.segment+=1;self.interval=1;self.interval_elapsed=0;self.phase='In transit';self.create_decision('Segment activated',now)
        elif kind=='missing':
            self.missing=set() if self.missing else {'D3'};self.inputs_changed=True;self.decision_due=now+2
            self.event('Telemetry updated','D3 restored' if not self.missing else 'D3 battery unavailable',now)
        elif kind=='delay':self.delayed=not self.delayed;self.event('Data freshness changed','Delayed feed' if self.delayed else 'Feed restored',now)
        elif kind=='late':
            self.event('Late result ignored','Older decision retained in history; current recommendation unchanged',now)
        elif kind=='auto':self.auto=not self.auto;self.event('Automatic decisions', 'Running' if self.auto else 'Paused · flight continues',now)
        else:raise ValueError('Unknown demo action')
        return self.snapshot()
    def snapshot(self):
        self.tick();now=self.clock();arrival=self.remaining();segment_progress=min(1,((self.interval-1)*self.period+self.interval_elapsed)/(self.intervals*self.period))
        cfg=next(x for x in CONFIGS if x['id']==self.applied)
        forecast=evaluate(cfg,self.batteries,arrival,self.wind,self.k,self.assignment) if not self.missing else None
        if forecast and self.phase=='Charging':
            charge_elapsed=max(0,now-self.phase_at)
            forecast['charging_s']=round(max(0,forecast['charging_s']-charge_elapsed));forecast['ready_s']=forecast['charging_s']
        if forecast and self.phase in ['Ready at node','Completed']:forecast.update(arrival_s=0,charging_s=0,ready_s=0)
        observed=now-35 if self.delayed else now
        charge_states={}
        if forecast and self.phase=='Charging':
            elapsed=max(0,now-self.phase_at)
            charge_states={r['drone_id']:('Charged' if elapsed>=r['end_s'] else ('Charging' if elapsed>=r['start_s'] else 'Waiting')) for r in forecast['schedule']}
        drones=[dict(id=f'D{i+1}',battery=None if f'D{i+1}' in self.missing else self.batteries[i],source='Demo',
          connection='Unknown' if f'D{i+1}' in self.missing else 'Connected',quality='Missing' if f'D{i+1}' in self.missing else ('Delayed' if self.delayed else 'Current'),
          observed_at=iso(observed),position=self.assignment[f'D{i+1}'],position_source='Target layout',charging_state=('Charged' if self.phase=='Ready at node' else charge_states.get(f'D{i+1}'))) for i in range(5)]
        return deepcopy(dict(mode='Demo',source='Scripted demonstration',provider='Demo provider',observed_at=iso(now),connection=self.connection,
          mission=dict(id='MED-024',name='Central to RPA',status=self.phase,segment_index=self.segment,segment_id=f'S{self.segment+1}',from_node=f'N{self.segment}',target_node=f'N{self.segment+1}',
           interval=self.interval,interval_count=self.intervals,interval_progress=min(1,self.interval_elapsed/self.period),segment_progress=segment_progress,
           next_decision_s=max(0,self.period-self.interval_elapsed) if self.phase=='In transit' else None,remaining_indoor_m=round(arrival*SETTINGS['indoor_interval_distance_m']/self.period,2),remaining_map_m=None,
           auto_decisions=self.auto,mapping='No geographic-to-indoor scale applied',period_s=self.period),
          conditions=dict(wind=self.wind,level=self.level,airflow='↓' if self.wind=='Headwind' else ('↑' if self.wind=='Tailwind' else '←'),k=self.k,capacity=self.capacity,observed_at=iso(observed)),
          applied=dict(config_id=self.applied,assignment=self.assignment,revision=self.revision,source='Demo execution feedback'),drones=drones,
          forecast=dict(**forecast,as_of=iso(now),basis='Current plan',target_node=f'N{self.segment+1}',quality='Delayed inputs' if self.delayed else 'Demo estimate') if forecast else None,
          decision=self.current,history=self.history[:40],events=self.events[:15],inputs_changed=self.inputs_changed,settings=SETTINGS,configurations=CONFIGS))
