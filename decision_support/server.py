"""Standalone localhost provider UI and isolated demo/replay service."""
from pathlib import Path
from copy import deepcopy
import json
import os
import threading
import time
import urllib.request
from flask import Flask, jsonify, request, send_from_directory
from .engine import DemoProvider, SETTINGS, CONFIGS, iso

ROOT=Path(__file__).parent
app=Flask(__name__,static_folder='static',static_url_path='/static')
provider=DemoProvider();lock=threading.RLock()

def route_data():return json.loads((ROOT/'data/route.json').read_text())

def live_snapshot():
    url=os.getenv('PROVIDER_SNAPSHOT_URL')
    if not url:return dict(mode='Live',connection='Not configured',reason='Connect a provider snapshot service to receive live mission data.',settings=SETTINGS,configurations=CONFIGS)
    try:
        with urllib.request.urlopen(url,timeout=3) as response:data=json.load(response)
        if data.get('mode')!='Live':raise ValueError('Provider must explicitly identify Live data')
        if not all(k in data for k in ('mission','drones','applied','conditions','observed_at')):raise ValueError('Provider snapshot missing required fields')
        return data
    except Exception as e:return dict(mode='Live',connection='Disconnected',reason=f'Provider unavailable: {type(e).__name__}',settings=SETTINGS,configurations=CONFIGS)

@app.get('/')
def index():return send_from_directory(ROOT/'static','index.html')
@app.get('/mission-input')
def mission_input():return send_from_directory(ROOT/'static','mission-input.html')
@app.get('/api/route')
def route():return jsonify(route_data())
@app.get('/api/snapshot')
def snapshot():
    if request.args.get('mode','demo')=='live':return jsonify(live_snapshot())
    with lock:return jsonify(provider.snapshot())
@app.get('/api/demo/figure/<int:interval>')
def demo_figure(interval):
    if interval not in FIGURE_SNAPSHOTS:return jsonify(error='Demo interval not found'),404
    return jsonify(deepcopy(FIGURE_SNAPSHOTS[interval]))
@app.post('/api/demo/action')
def action():
    # An explicit namespace and mode guard isolate all scripted commands from Live.
    payload=request.get_json(silent=True) or {}
    if payload.get('mode')!='Demo':return jsonify(error='Demo actions require Demo mode'),409
    with lock:
        try:return jsonify(provider.action(payload.get('action')))
        except ValueError as e:return jsonify(error=str(e)),400
@app.get('/api/replay')
def replay():
    return jsonify(source='Recorded demo session',frames=[dict(index=i,label=f['label'],at=f['snapshot']['observed_at']) for i,f in enumerate(REPLAY)])
@app.get('/api/replay/<int:index>')
def replay_frame(index):
    if not 0<=index<len(REPLAY):return jsonify(error='Frame not found'),404
    result=deepcopy(REPLAY[index]['snapshot']);result['mode']='Replay';result['source']='Recorded demo session';result['replay_index']=index
    return jsonify(result)
@app.get('/api/health')
def health():return jsonify(status='ok',hardware_control=False,live_configured=bool(os.getenv('PROVIDER_SNAPSHOT_URL')))
@app.after_request
def headers(response):
    if request.path.startswith('/api/'):response.headers['Cache-Control']='no-store'
    response.headers['X-Content-Type-Options']='nosniff';response.headers['Referrer-Policy']='no-referrer'
    return response

def build_replay():
    current=[1790280000.];p=DemoProvider(clock=lambda:current[0]);frames=[]
    # Interval 3 starts ten seconds in. The interval-4 boundary changes wind
    # while keeping pad availability fixed; allow that decision to apply
    # before the swap, then reach interval 5 after the swap has applied.
    steps=[
      (0,None,'Initial recommendation'),(4,None,'Initial command sent'),
      (4,None,'Initial command acknowledged'),(3,None,'Initial configuration applied'),
      (4,None,'Interval 4 decision calculating'),(0,None,'Wind changes'),
      (0,None,'Wind decision calculating'),(1,None,'Wind recommendation'),
      (2,None,'Wind command sent'),(3,None,'Wind command acknowledged'),
      (3,None,'Wind configuration applied'),(0,'swap','Position reassignment requested'),
      (1,None,'Position reassignment recommended'),(2,None,'Position reassignment sent'),
      (3,None,'Position reassignment acknowledged'),(3,None,'New assignments applied'),
      (7,None,'Interval 5 decision calculating'),(0,'pads','Charging availability changes'),
      (2,None,'Pad decision calculating'),(1,None,'Pad recommendation'),
      (2,None,'Pad command sent'),(3,None,'Pad command acknowledged'),
      (3,None,'Pad configuration applied'),(0,'arrive','Arrived at rooftop node'),
      (4,None,'Charging starts'),(0,'complete_charge','Ready at node'),
    ]
    for advance,action,label in steps:
        current[0]+=advance
        if action:p.action(action)
        frames.append(dict(label=label,snapshot=p.snapshot()))
    return frames
REPLAY=build_replay()

def build_figure_snapshots():
    current=[time.time()];demo=DemoProvider(clock=lambda:current[0])
    current[0]+=11
    before=demo.snapshot()
    current[0]+=13
    after=demo.snapshot()
    return {3:before,4:after}

FIGURE_SNAPSHOTS=build_figure_snapshots()
if __name__=='__main__':
    def provider_clock():
        while True:
            with lock:provider.tick()
            time.sleep(.25)
    threading.Thread(target=provider_clock,daemon=True,name='demo-provider-clock').start()
    app.run(host='127.0.0.1',port=int(os.getenv('PLATFORM_PORT','8767')),debug=False,threaded=True)
