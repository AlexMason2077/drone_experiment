const $ = id => document.getElementById(id);
const esc = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state = {mode:'Demo',snapshot:null,route:null,drone:null,node:null,segment:null,interval:null,decision:null,tab:'history',expanded:false,view:[0,0,1100,650],replay:0,playing:false,frames:[],generation:0,request:0,accepted:0,drag:null};
const figureInterval=Number(new URLSearchParams(window.location.search).get('demo-interval'));
const figureMode=[3,4].includes(figureInterval);
let busy=false, toastTimer, realMap=null, realLayers=null, realMapReady=false, lastRealMapSignature=null;
function toast(message){$('toast').textContent=message;$('toast').classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.remove('visible'),3500);}
function clock(t){return t ? new Date(t).toLocaleTimeString('en-AU',{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}) : '—';}
function duration(seconds){if(seconds==null || !Number.isFinite(seconds))return '—';let n=Math.max(0,Math.round(seconds));const h=Math.floor(n/3600),m=Math.floor(n%3600/60),s=n%60;return h?`${h}h ${String(m).padStart(2,'0')}m ${String(s).padStart(2,'0')}s`:m?`${m}m ${String(s).padStart(2,'0')}s`:`${s}s`;}
function age(t){const n=(Date.parse(state.snapshot?.observed_at)-Date.parse(t))/1000;return Number.isFinite(n)?(n<2?'Just updated':`${Math.round(n)}s ago`):'No timestamp';}
function config(id){return state.snapshot?.configurations?.find(c=>c.id===id);}
function configName(id){const c=config(id);return c?`${c.formation} · ${c.spacing_m.toFixed(2)} m`:'Unavailable';}
function nodeName(id){return state.route?.nodes.find(n=>n.id===id)?.name??id;}
function statusTag(status){return `<span class="tag ${['Sent','Calculating','Unavailable','Superseded'].includes(status)?'amber':''}">${esc(status)}</span>`;}
function selectedDecision(){return state.snapshot?.history?.find(d=>d.id===state.decision)??state.snapshot?.decision;}

async function loadSnapshot(){
  if(busy)return;busy=true;const generation=state.generation, request=++state.request;
  try{
    const url=state.mode==='Replay'?`/api/replay/${state.replay}`:state.mode==='Demo'&&figureMode?`/api/demo/figure/${figureInterval}`:`/api/snapshot?mode=${state.mode.toLowerCase()}`;
    const response=await fetch(url);if(!response.ok)throw new Error('Provider response unavailable');
    const data=await response.json();if(generation!==state.generation||request<state.accepted)return;
    accept(data,request);
  }catch(error){if(generation===state.generation){$('connection').textContent='Connection interrupted';$('connection').classList.add('offline');$('mode-banner').hidden=false;$('mode-banner').className='mode-banner warning';$('mode-banner').textContent='Connection interrupted. Last received values remain visible; predictions are not current.';}}
  finally{busy=false;}
}
function accept(data,request){
  if(state.mode!=='Replay'&&state.snapshot?.mission && data.mission){
    if(Date.parse(data.observed_at)<Date.parse(state.snapshot.observed_at))return;
    // A late model result cannot roll the current command or execution state back.
    if(data.decision?.sequence<state.snapshot.decision?.sequence){data.decision=state.snapshot.decision;}
    if(data.applied?.revision<state.snapshot.applied?.revision){data.applied=state.snapshot.applied;}
  }
  state.accepted=request;state.snapshot=data;render();
}
async function action(kind){
  if(state.mode!=='Demo')return;const generation=state.generation;const request=++state.request;
  try{
    const response=await fetch('/api/demo/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'Demo',action:kind})});
    const data=await response.json();if(!response.ok)throw new Error(data.error);
    if(generation!==state.generation||request<state.accepted)return;
    if(kind==='reset'){state.snapshot=null;state.decision=null;state.segment=null;state.interval=null;state.node=null;state.drone=null;}
    accept(data,request);
    const messages={wind:'Wind updated. A new decision is scheduled.',pads:'Charging availability updated. A new decision is scheduled.',swap:'Position reassignment requested. Waiting for execution feedback.',await_ack:'Demo command will remain sent until acknowledgement is released.',auto:data.mission.auto_decisions?'Automatic decisions resumed.':'Automatic decisions paused. Flight continues in the demo.',late:'Late result ignored. The current decision is unchanged.',reset:'Demo reset to interval 3.'};
    toast(messages[kind]??'Demo scenario updated.');
  }catch(error){toast(error.message);}
}
function render(){
 const s=state.snapshot, m=s.mission;const live=state.mode==='Live',replay=state.mode==='Replay';
 $('demo-tools').hidden=state.mode!=='Demo'||figureMode;$('replay-tools').hidden=!replay;$('auto-button').disabled=state.mode!=='Demo'||!m||m.status!=='In transit';
 $('auto-button').textContent=m?.auto_decisions===false?'Resume auto decisions':'Pause auto decisions';
 const delayed=s.drones?.some(d=>d.quality==='Delayed'),missing=s.drones?.some(d=>d.quality==='Missing');
 $('connection').textContent=replay?'Recorded session':s.connection==='Connected'?(live?'Provider connected':'Demo provider connected'):s.connection;
 $('connection').className=`connection ${!replay&&s.connection!=='Connected'?'offline':''}`;
 const warnings=[];
 if(live&&!m)warnings.push('Provider not connected · '+(s.reason??'Waiting for provider data.'));
 if(delayed)warnings.push('Feed delayed — last known battery and wind values.');
 if(missing)warnings.push('D3 battery unavailable — a new estimate needs complete inputs.');
 $('mode-banner').hidden=warnings.length===0;
 $('mode-banner').className='mode-banner warning';
 $('mode-banner').textContent=warnings.join(' ');
 if(m){renderSummary(s);renderConditions(s);renderConfigurations(s);renderCommand(s);renderPrediction(s);renderFleet(s);renderIntervals(s);renderSegmentProgress(s);}else{renderEmpty(s);}
 if(!state.drag)renderMap();renderDetails();
 $('last-updated').textContent=s.observed_at?`${replay?'Recording time':'Last updated'} ${clock(s.observed_at)} · ${state.mode}`:`${state.mode} · waiting for provider`;
 const frame=state.frames[state.replay];$('replay-label').textContent=frame?`${state.replay+1} / ${state.frames.length} · ${frame.label}`:'';$('replay-range').value=state.replay;
}
function renderSummary(s){const m=s.mission,f=s.forecast;
 const arrived=m.status!=='In transit';
 const difference=appliedDifference(s);
 $('summary').innerHTML=`<article class="summary-card arrival"><div class="summary-label">${arrived?'At next node':'Arrival at next node'} · ${esc(m.target_node)}</div><div class="summary-value">${arrived?'Arrived':duration(f?.arrival_s)}</div><div class="summary-foot">${esc(nodeName(m.target_node))} · ${arrived?'flight complete':'remaining flight'}</div><div id="segment-progress" class="arrival-progress" aria-label="Flight progress"></div></article>
 <article class="summary-card ready"><div class="ready-content"><div class="ready-time"><div class="summary-label">Ready at next node · ${esc(m.target_node)}</div><div class="summary-value">${duration(f?.ready_s)}</div><div class="summary-foot">Includes waiting and charging for all five drones · 99% target</div></div>${difference?`<div class="ready-difference"><div class="ready-difference-label">${difference.saved>0?'Estimated saving':'Estimated time change'} from configuration change</div><strong class="${difference.saved<0?'longer':''}">${esc(difference.text)}</strong><div class="ready-difference-context">${esc(difference.decision.segment_id)} · interval ${esc(difference.decision.interval)} · ${esc(difference.decision.id)} · same inputs · ${esc(difference.source)}</div></div>`:''}</div></article>`;
}
function configurationChanged(comparison){
 if(!comparison?.before||!comparison?.after)return false;
 return comparison.before.config_id!==comparison.after.config_id||['D1','D2','D3','D4','D5'].some(id=>comparison.before.assignment?.[id]!==comparison.after.assignment?.[id]);
}
function appliedDifference(s){
 const m=s.mission;
 if(!m||m.status!=='In transit'||!Number.isFinite(s.forecast?.ready_s)||s.forecast.target_node!==m.target_node||!s.applied)return null;
 const seen=new Set();
 const records=[s.decision,...(s.history??[])].filter(d=>{
  if(!d||seen.has(d.id))return false;
  seen.add(d.id);return true;
 });
 const lastChange=records.filter(d=>d.status==='Applied'&&d.segment_id===m.segment_id&&d.target_node_id===m.target_node&&configurationChanged(d.comparison))
  .sort((a,b)=>(Date.parse(b.applied_at??b.input_as_of)||0)-(Date.parse(a.applied_at??a.input_as_of)||0))[0];
 if(!lastChange)return null;
 const comparison=lastChange.comparison,saved=comparison.saved_s;
 if(!Number.isFinite(saved)||!Number.isFinite(comparison.before.ready_s)||!Number.isFinite(comparison.after.ready_s)||Math.abs(comparison.before.ready_s-comparison.after.ready_s-saved)>.01)return null;
 if(lastChange.comparison.after.config_id!==s.applied.config_id||['D1','D2','D3','D4','D5'].some(id=>lastChange.comparison.after.assignment?.[id]!==s.applied.assignment?.[id]))return null;
 const source=state.mode==='Replay'?'Replay demo estimate':state.mode==='Demo'?(comparison.quality==='Delayed demo inputs'?'Demo estimate · delayed inputs':'Demo estimate'):(comparison.quality??'Provider estimate');
 const differenceText=saved>0?`${duration(saved)} shorter`:saved<0?`${duration(-saved)} longer`:'No estimated saving';
 return {decision:lastChange,saved,text:differenceText,source};
}
function renderConditions(s){const c=s.conditions;
 $('conditions').innerHTML=`<div class="condition"><div class="condition-label">Wind condition</div><div class="condition-value"><span class="wind-arrow" title="Airflow direction relative to flight">${esc(c.airflow)}</span>${esc(c.wind)} <small>${esc(c.level)}</small></div></div><div class="condition"><div class="condition-label">Charging pad availability · ${esc(s.mission.target_node)}</div><div class="condition-value"><strong>${esc(c.k)}</strong><small>available${c.capacity==null?'':` of ${esc(c.capacity)}`}</small></div></div>`;
}
function measuredLayout(telemetry){return telemetry?.frame==='local_forward'&&telemetry?.unit==='m'&&telemetry?.source==='Telemetry'&&telemetry.positions?.length===5&&new Set(telemetry.positions.map(p=>p.drone_id)).size===5&&telemetry.positions.every(p=>/^D[1-5]$/.test(p.drone_id)&&Number.isFinite(p.x)&&Number.isFinite(p.y));}
function formationSVG(id,assignment,batteryLevels,recommended=false,telemetry=null){const c=config(id);if(!c)return '<div class="formation-empty">Configuration unavailable</div>';
 const measured=measuredLayout(telemetry);let pts=c.positions;
 if(measured){const meanX=telemetry.positions.reduce((a,p)=>a+p.x,0)/5,meanY=telemetry.positions.reduce((a,p)=>a+p.y,0)/5;pts=telemetry.positions.map(p=>({id:assignment[p.drone_id],x:p.x-meanX,y:p.y-meanY}));}
 const maxX=Math.max(...pts.map(p=>Math.abs(p.x))),maxY=Math.max(...pts.map(p=>Math.abs(p.y)));
 // Use one metre-to-pixel scale for target layouts, limited only by the SVG bounds.
 const scale=measured?Math.min(90/Math.max(.5,maxX),100/Math.max(.5,maxY)):Math.min(100,112/Math.max(.01,maxX),112/Math.max(.01,maxY));
 const project=p=>[130+p.x*scale,130-p.y*scale];
 const ref=c.spacing_reference.map(id=>pts.find(p=>p.id===id));const [a,b]=ref.map(project);
 const column=c.formation==='Column';
 const marks=pts.map(p=>{const drone=Object.keys(assignment??{}).find(k=>assignment[k]===p.id);const [x,y]=project(p);const level=batteryLevels?.[drone],battery=Number.isFinite(level)&&level>=0&&level<=100?`${level}%`:'—';return `<g class="drone-dot ${state.drone===drone?'selected':''}" data-drone="${esc(drone)}" role="button" tabindex="0" aria-label="${esc(drone)} assigned to ${esc(p.id)}, battery ${esc(battery)}" transform="translate(${x},${y})"><circle r="15" fill="${recommended?'#eef7e8':'#f6f8f4'}" stroke="${recommended?'#89ad89':'#b5c7b8'}" stroke-width="1.3"/><text class="drone-label" text-anchor="middle" y="-3" font-family="inherit" font-size="10" font-weight="650" fill="#3d6350">${esc(drone??'—')}</text><text class="battery-label" text-anchor="middle" y="9" font-family="inherit" font-size="9" font-weight="600" fill="#3d6350">${esc(battery)}</text><text text-anchor="${column?'start':'middle'}" x="${column?21:0}" y="${column?3:29}" font-size="10" fill="#80917d">${esc(p.id)}</text></g>`;}).join('');
 const dimension=column&&!measured?(()=>{const top=Math.min(a[1],b[1]),bottom=Math.max(a[1],b[1]),lineX=82,witnessEnd=Math.min(a[0],b[0])-16;return `<g class="spacing-dimension" aria-label="${esc(ref[0].id)} to ${esc(ref[1].id)} centre distance ${c.spacing_m.toFixed(2)} metres"><path d="M${lineX} ${top}H${witnessEnd} M${lineX} ${bottom}H${witnessEnd} M${lineX} ${top}V${bottom} M${lineX-3} ${top+5}L${lineX} ${top}L${lineX+3} ${top+5} M${lineX-3} ${bottom-5}L${lineX} ${bottom}L${lineX+3} ${bottom-5}" fill="none" stroke="#6c9b7c" stroke-width="1.1"/><text x="${lineX-7}" y="${(top+bottom)/2+4}" text-anchor="end" fill="#527864" font-size="11" font-weight="600">${c.spacing_m.toFixed(2)} m</text></g>`;})():'';
 const bar=c.spacing_m*scale;
 return `<svg viewBox="0 0 260 300" aria-label="${esc(c.formation)} ${measured?'reported positions':'target formation'}"><path d="M240 64V42m-4 6 4-6 4 6" fill="none" stroke="#94a48e" stroke-width="1.1"/><text x="240" y="73" text-anchor="middle" fill="#8a9785" font-size="9">Flight</text>${measured?'':`<path d="M${a[0]},${a[1]}L${b[0]},${b[1]}" stroke="#bfceb8" stroke-width="1" stroke-dasharray="2 2"/>`}${dimension}${marks}<path d="M${130-bar/2} 287h${bar}m-${bar}-3v6m${bar}-6v6" stroke="#a2b598" stroke-width=".8"/><text x="130" y="275" text-anchor="middle" fill="#8a9a80" font-size="9">${c.spacing_m.toFixed(2)} m · ${measured?'scale':c.spacing_reference.join('–')}</text></svg>`;
}
function renderConfigurations(s){const d=s.decision,rec=d?.recommended,calculating=d?.status==='Calculating';
 const current=config(s.applied.config_id);
 const appliedDecision=[s.decision,...(s.history??[])].find(record=>record?.status==='Applied'&&record.recommended?.config_id===s.applied.config_id&&['D1','D2','D3','D4','D5'].every(id=>record.recommended.assignment?.[id]===s.applied.assignment?.[id]));
 const currentBattery=Object.fromEntries((s.drones??[]).map(drone=>[drone.id,drone.battery]));
 const batteryQuality=(s.drones??[]).some(drone=>drone.battery==null)?' · missing reading':(s.drones??[]).some(drone=>drone.quality==='Delayed')?' · delayed':'';
 const decisionBattery=d?.input_battery_levels,decisionBatteryNote=decisionBattery?'Battery at decision · '+clock(d.input_as_of):'Decision battery unavailable';
 const selected=s.drones?.find(drone=>drone.id===state.drone);
 $('configurations').innerHTML=`<div class="config-card"><div class="config-name">${appliedDecision?'Best configuration':'Current configuration'}</div><div class="configuration-fields"><div><span>Formation:</span><strong>${esc(current?.formation)}</strong></div><div><span>Inter-drone spacing:</span><strong>${current?Math.round(current.spacing_m*100)+' cm':'—'}</strong></div></div><div class="config-battery-note">Current battery${batteryQuality}</div>${formationSVG(s.applied.config_id,s.applied.assignment,currentBattery,false,s.formation_positions)}<div class="config-meta">${appliedDecision?'Selected by '+esc(appliedDecision.id)+' · Applied':'Current execution feedback'} · ${measuredLayout(s.formation_positions)?'Reported positions':'Target layout'}</div>${selected?`<div class="selected-drone-note">${esc(selected.id)} · ${selected.battery==null?'Battery unavailable':`${esc(selected.battery)}% battery`} · ${esc(selected.quality)} · updated ${clock(selected.observed_at)}</div>`:''}</div>`;
 $('recommendation-detail').innerHTML=`<h3>Recommended configuration</h3><div class="recommendation-layout">${rec&&!calculating?formationSVG(rec.config_id,rec.assignment,decisionBattery,true):`<div class="formation-empty">${calculating?'Decision in progress…':'Recommendation not returned'}</div>`}<div><strong>${rec&&!calculating?esc(configName(rec.config_id)):'—'}</strong><p>${rec&&!calculating?esc(decisionBatteryNote):'Battery at decision · —'}</p><p>${rec&&!calculating?`Target layout · ${d.candidates.length} candidates`:'Awaiting a complete decision'}</p></div></div>`;
}
function renderCommand(s){const d=s.decision;
 $('decision-badge').textContent=d?.status??'Waiting';$('decision-badge').className=`subtle-tag ${d?.status!=='Applied'?'pending':''}`;
 if(!d){$('command').innerHTML='<p class="command-note">Waiting for the next provider decision.</p>';return;}
 const steps=['Recommended','Sent','Acknowledged','Applied'];const index=steps.indexOf(d.status);
 const overdue=d.status==='Sent'&&Date.parse(s.observed_at)>Date.parse(d.ack_due_at);
 const note=s.inputs_changed?'Inputs changed · a new decision is pending.':overdue?'Acknowledgement overdue · executing configuration is unchanged.':d.status==='Sent'?'Waiting for acknowledgement.':d.status==='Acknowledged'?'Command received · waiting for execution feedback.':d.status==='Applied'?`Applied feedback received from ${d.feedback.length} drones.`:d.status==='Unavailable'?d.reason:d.status==='Superseded'?'Decision closed. Current execution is shown above.':d.status==='Calculating'?'Provider is preparing the candidate list.':'Provider will send the selected configuration.';
 $('command').innerHTML=`<div class="command-head"><strong>${esc(d.id)}</strong><span>${esc(d.command_id??'No command')} · ${clock(d.input_as_of)}</span></div><div class="steps">${steps.map((x,i)=>`<span class="step ${i<=index?'complete':''} ${i===index?'active':''}"><i></i>${x}</span>`).join('')}</div><p class="command-note ${overdue||s.inputs_changed?'warning':''}">${esc(note)}</p>`;
}
function renderPrediction(s){const f=s.forecast,rec=s.decision?.recommended;
 $('prediction').innerHTML=`<div class="time-row"><span>Arrival in</span><strong>${duration(f?.arrival_s)}</strong></div><div class="time-row"><span>Charging after arrival <small>· includes waiting</small></span><strong>${duration(f?.charging_s)}</strong></div><div class="time-row total"><span>Ready at next node</span><strong>${duration(f?.ready_s)}</strong></div><div class="forecast-note">${f?`${esc(f.basis)} · updated ${clock(f.as_of)} · ${esc(f.quality)}`:'Estimate unavailable · complete battery inputs required'}${rec&&s.decision.status!=='Applied'&&s.decision.status!=='Calculating'&&s.mission.status==='In transit'?`<br>Recommended · if applied: ${duration(rec.ready_s)} · inputs ${clock(s.decision.input_as_of)}`:''}</div>`;
}
function renderFleet(s){$('fleet-source').textContent=`${state.mode==='Live'?'Reported battery':state.mode==='Replay'?'Recorded demo battery':'Demo battery'} · select a drone to inspect`;
 $('fleet').innerHTML=s.drones.map(d=>`<button class="drone-card ${state.drone===d.id?'selected':''}" data-drone="${esc(d.id)}" aria-pressed="${state.drone===d.id}" aria-label="${esc(d.id)} battery ${d.battery??'unavailable'}"><div class="drone-top"><span class="drone-name"><img class="drone-icon" src="/static/drone-icon.svg" alt="">${esc(d.id)}</span><span class="drone-connection ${d.connection!=='Connected'?'missing':''}" title="${esc(d.connection)}"></span></div><div class="battery-row"><strong>${d.battery??'—'}</strong><small>${d.battery==null?'':'%'}</small><span>${esc(d.charging_state??d.position)}</span></div><div class="battery-bar"><i style="width:${d.battery??0}%"></i></div><div class="drone-meta ${d.quality!=='Current'?'warning':''}"><span>${esc(d.connection)} · ${esc(d.quality==='Current'?d.source:d.quality)}</span><span>${age(d.observed_at)}</span></div></button>`).join('');}
function recordedWind(d){return d?.wind&&d?.level?`${d.wind} · ${d.level}`:'Not recorded';}
function recordedPads(d){return d?.k==null?'Not recorded':`${d.k} available${d.target_node_id?` at ${d.target_node_id}`:''}`;}
function renderIntervals(s){const m=s.mission;$('timeline-caption').textContent=`${m.from_node} → ${m.target_node} · ${m.period_s}s each`;
 $('decision-countdown').textContent=m.status!=='In transit'?m.status:m.auto_decisions?`Next decision in ${Math.ceil(m.next_decision_s)}s`:'Automatic decisions paused';
 const decisions=(s.history??[]).filter(d=>d.segment_id===m.segment_id).sort((a,b)=>(Number(b.sequence)||0)-(Number(a.sequence)||0)||(Date.parse(b.input_as_of)||0)-(Date.parse(a.input_as_of)||0));
 $('intervals').innerHTML=Array.from({length:m.interval_count},(_,i)=>{
  const number=i+1,records=decisions.filter(d=>d.interval===number),latest=records[0];
  const configuration=latest?.recommended?.config_id?configName(latest.recommended.config_id):'Not available';
  const label=latest?`wind ${recordedWind(latest)}, charging pads ${recordedPads(latest)}, recommended configuration ${configuration}`:'no decision inputs recorded';
  return `<button class="interval-button ${number<m.interval?'done':number===m.interval?'current':''} ${state.interval===number?'inspected':''}" data-interval="${number}" aria-label="Interval ${number}: ${esc(label)}">${number===m.interval?`<i style="width:${m.interval_progress*100}%"></i>`:''}<span class="interval-number">${number<m.interval?'✓ ':''}${String(number).padStart(2,'0')}${number===m.interval?' · Current':''}${records.length>1?` · ${records.length} decisions`:''}</span>${latest?`<span class="interval-input">Wind: ${esc(recordedWind(latest))}</span><span class="interval-input">Pads: ${esc(recordedPads(latest))}</span><span class="interval-input">Config: ${esc(configuration)}</span>`:'<span class="interval-input unavailable">No decision inputs recorded</span>'}</button>`;
 }).join('');}
function renderSegmentProgress(s){const m=s.mission,atNode=m.status!=='In transit';$('segment-progress').innerHTML=`<span>${atNode?'At '+esc(m.target_node):esc(m.from_node)+' → '+esc(m.target_node)} · segment ${m.segment_index+1} of ${state.route.segments.length}</span><strong class="segment-status">${esc(m.status)}</strong>`;}
function renderEmpty(s){$('summary').innerHTML=['Arrival at next node','Ready at next node'].map((x,i)=>`<article class="summary-card ${i===1?'ready':''}"><div class="summary-label">${x}</div><div class="summary-value">—</div><div class="summary-foot">Awaiting live provider data</div>${i===0?'<div id="segment-progress" class="arrival-progress"></div>':''}</article>`).join('');$('conditions').innerHTML='<div class="empty">Wind and charging availability unavailable</div>';$('configurations').innerHTML='<div class="config-card"><div class="config-name">Current configuration</div><div class="formation-empty">No live configuration</div></div>';$('recommendation-detail').innerHTML='<h3>Recommended configuration</h3><div class="empty">No provider decision</div>';$('command').innerHTML='<p class="command-note">No command has been sent from this platform.</p>';$('prediction').innerHTML='<div class="empty">Connect the provider to receive estimates.</div>';$('decision-badge').textContent='Not connected';$('fleet-source').textContent='Live battery · waiting for readings';$('fleet').innerHTML=Array.from({length:5},(_,i)=>`<div class="drone-card"><div class="drone-name">D${i+1}</div><div class="battery-row"><strong>—</strong></div><div class="drone-meta">No live reading</div></div>`).join('');$('intervals').innerHTML='<div class="empty">Waiting for the active segment and interval.</div>';$('timeline-caption').textContent='';$('decision-countdown').textContent='';$('segment-progress').innerHTML='<p class="empty">No live swarm position</p>';}

function fitRealMap(){
 if(!realMap||!state.route)return;
 const bounds=window.L.latLngBounds(state.route.nodes.map(n=>[n.lat,n.lon]));
 realMap.invalidateSize();
 const sidebar=document.querySelector('.decision-panel').getBoundingClientRect(),summary=$('summary').getBoundingClientRect(),details=document.querySelector('.details-panel').getBoundingClientRect();
 const desktop=window.innerWidth>850;
 realMap.fitBounds(bounds,{paddingTopLeft:[desktop?sidebar.right+45:28,Math.max(180,summary.bottom+65)],paddingBottomRight:[desktop?110:35,Math.max(90,window.innerHeight-details.top+70)],maxZoom:window.innerWidth>=2200?16:15.5,animate:false});
}
function initRealMap(){
 if(!window.L||!state.route)return;
 realMap=window.L.map('real-map',{zoomControl:false,attributionControl:false,scrollWheelZoom:true,preferCanvas:true,zoomSnap:.25,minZoom:11,maxZoom:17});
 const attribution=window.L.control.attribution({position:'bottomleft',prefix:false}).addTo(realMap);
 attribution.addAttribution('&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap contributors</a>');
 attribution.addAttribution('Map tiles &copy; <a href="https://www.openstreetmap.de/" target="_blank" rel="noreferrer">OpenStreetMap.de</a>');
 const scale=1100/(.055*Math.cos(-33.8845*Math.PI/180));
 const imageBounds=window.L.latLngBounds([[-33.870-650/scale,151.167],[-33.870,151.222]]);
 const image=window.L.imageOverlay('/static/map-base.svg',imageBounds);
 image.on('load',()=>{realMapReady=true;$('real-map').classList.add('ready');$('route-map').style.display='none';document.querySelector('.map-credit').hidden=true;});
 image.addTo(realMap);
 const tiles=window.L.tileLayer('https://tile.openstreetmap.de/tiles/osmde/{z}/{x}/{y}.png',{minZoom:11,maxZoom:17});
 tiles.once('load',()=>{realMap.removeLayer(image);});
 tiles.addTo(realMap);
 realLayers=window.L.layerGroup().addTo(realMap);
 window.addEventListener('resize',()=>{lastRealMapSignature=null;renderRealMap();fitRealMap();});
}
function renderRealMap(){
 if(!realMap||!realLayers||!state.route)return;
 const L=window.L,nodes=state.route.nodes,s=state.snapshot,m=s?.mission;
 const markerScale=window.innerWidth>=2200?1.8:1;
 const geo=s?.geo_position;
 const signature=JSON.stringify([state.mode,m?.segment_index,m?.segment_progress,m?.status,geo?.source,geo?.lat,geo?.lon,state.node,state.segment,state.drone]);
 if(signature===lastRealMapSignature)return;
 lastRealMapSignature=signature;
 realLayers.clearLayers();
 state.route.segments.forEach((seg,i)=>{
  const points=[[nodes[i].lat,nodes[i].lon],[nodes[i+1].lat,nodes[i+1].lon]];
  const active=m&&i===m.segment_index,done=m&&i<m.segment_index;
  const color=active?'#176c5a':done?'#344b54':'#8a9f94';
  L.polyline(points,{color:'#fff',weight:10*markerScale,opacity:.95,interactive:false}).addTo(realLayers);
  L.polyline(points,{color,weight:(active?6:4)*markerScale,opacity:.98,dashArray:!active&&!done?'8 9':null,interactive:false}).addTo(realLayers);
  L.polyline(points,{color:'transparent',weight:25,opacity:0}).addTo(realLayers).on('click',()=>{state.segment=seg.id;state.node=null;renderMapSelection();renderRealMap();});
 });
 nodes.forEach((n,i)=>{
  const medical=i===0||i===nodes.length-1,selected=state.node===n.id;
  const marker=L.circleMarker([n.lat,n.lon],{radius:(medical?11:9)*markerScale,color:medical?'#344b54':'#69865b',weight:2*markerScale,fillColor:selected?'#d3e7bd':medical?'#344b54':'#fff',fillOpacity:1}).addTo(realLayers);
  marker.bindTooltip(`${n.id} · ${n.name}`,{permanent:true,direction:i===0?'left':i===nodes.length-1?'right':'top',offset:i===0?[-14,0]:i===nodes.length-1?[14,0]:[0,-10],className:'route-node-name',opacity:1});
  marker.on('click',()=>{state.node=n.id;state.segment=null;renderMapSelection();renderRealMap();});
 });
 if(m){
  const actual=geo?.source==='GPS'&&Number.isFinite(geo.lon)&&Number.isFinite(geo.lat);
  if(state.mode!=='Live'||actual){
   const a=nodes[m.segment_index],b=nodes[m.segment_index+1],p=m.segment_progress;
   const position=actual?[geo.lat,geo.lon]:[a.lat+(b.lat-a.lat)*p,a.lon+(b.lon-a.lon)*p];
   const swarm=L.circleMarker(position,{radius:13*markerScale,color:'#fff',weight:2*markerScale,fillColor:'#176c5a',fillOpacity:1}).addTo(realLayers);
   swarm.bindTooltip(state.drone?`${state.drone} · in swarm`:'Swarm · 5 drones',{permanent:true,direction:'bottom',offset:[0,11],className:'route-swarm-label'});
   swarm.on('click',()=>{state.segment=m.segment_id;state.node=null;renderMapSelection();});
  }
 }
}

function renderMap(){
 if(!state.route)return;renderRealMap();if(realMapReady){renderMapSelection();return;}const s=state.snapshot,m=s?.mission,nodes=state.route.nodes;const live=state.mode==='Live';
 let html='<image href="/static/map-base.svg" x="0" y="0" width="1100" height="650"/>';
 // No unrelated point-of-interest labels. All scenario segment geometry follows node pairs.
 state.route.segments.forEach((seg,i)=>{const a=nodes[i].xy,b=nodes[i+1].xy,active=m&&i===m.segment_index,done=m&&i<m.segment_index,color=active?'#176c5a':done?'#344b54':'#9dafa3';
  html+=`<g class="map-route" data-segment="${seg.id}" tabindex="0" role="button" aria-label="Inspect segment ${seg.from_node} to ${seg.to_node}"><path d="M${a}L${b}" stroke="transparent" stroke-width="30"/><path d="M${a}L${b}" fill="none" stroke="#fff" stroke-width="8"/><path d="M${a}L${b}" fill="none" stroke="${color}" stroke-width="${active?5:3}" ${!active&&!done?'stroke-dasharray="7 7"':''}/></g>`;
 });
 nodes.forEach((n,i)=>{const [x,y]=n.xy,isMedical=i===0||i===4;const selected=state.node===n.id;
  if(n.footprint)html+=`<path d="M${n.footprint.map(p=>p.join(',')).join(' L')}Z" fill="#dce9d3" stroke="#75946f" stroke-width="1.3"/>`;
  const label=i===0?{x:x-318,y:y-52,w:308}:i===4?{x:x-63,y:y+30,w:318}:{x:x-48,y:y-48,w:128};
  html+=`<g class="map-node" data-node="${n.id}" tabindex="0" role="button" aria-label="Inspect ${esc(n.name)}"><circle class="node-circle" cx="${x}" cy="${y}" r="${isMedical?11:9}" fill="${selected?'#d3e7bd':isMedical?'#344b54':'#fff'}" stroke="${isMedical?'#344b54':'#69865b'}" stroke-width="2"/><text x="${x}" y="${y+3}" text-anchor="middle" font-size="9" font-weight="650" fill="${isMedical&&!selected?'white':'#4d6d41'}">${n.id}</text><rect x="${label.x}" y="${label.y}" width="${label.w}" height="30" rx="5" fill="#ffffffed" stroke="#dce6d7"/><text x="${label.x+10}" y="${label.y+20}" font-size="16" font-weight="600" fill="#3d5848">${esc(n.name)}</text></g>`;
 });
 if(m){
  const a=nodes[m.segment_index].xy,b=nodes[m.segment_index+1].xy;const p=m.segment_progress;
  // Geographical marker is a scenario progress illustration; indoor telemetry is never GPS.
  const geo=s.geo_position;
  const actualGeo=geo?.source==='GPS'&&Number.isFinite(geo.lon)&&Number.isFinite(geo.lat)?[(geo.lon-151.167)/.055*1100,(-33.870-geo.lat)*1100/(.055*Math.cos(-33.8845*Math.PI/180))]:null;
  if(!live||actualGeo){const [x,y]=actualGeo??[a[0]+(b[0]-a[0])*p,a[1]+(b[1]-a[1])*p];const angle=Math.atan2(b[1]-a[1],b[0]-a[0])*180/Math.PI;
    html+=`<g data-swarm="true" role="button" tabindex="0" aria-label="Inspect current swarm" style="cursor:pointer"><circle cx="${x}" cy="${y}" r="19" fill="#fff" opacity=".9"/><circle cx="${x}" cy="${y}" r="14" fill="#176c5a" stroke="#fff" stroke-width="2"/><path transform="translate(${x},${y}) rotate(${angle}) scale(.9)" d="M9 0 -6-6 -3 0 -6 6Z" fill="#fff"/><rect x="${x-68}" y="${y+24}" width="136" height="26" rx="5" fill="#176c5a"/><text x="${x}" y="${y+41}" font-size="13" font-weight="600" text-anchor="middle" fill="white">${state.drone?`${state.drone} · in swarm`:'Swarm · 5 drones'}</text></g>`;
  }
 }
 $('route-map').innerHTML=html;$('route-map').setAttribute('viewBox',state.view.join(' '));renderMapSelection();
}
function renderMapSelection(){const box=$('map-selection');if(!state.node&&!state.segment){box.hidden=true;return;}box.hidden=false;const m=state.snapshot?.mission;
 if(state.node){const n=state.route.nodes.find(x=>x.id===state.node),current=n.id===m?.target_node;const live=state.mode==='Live';const k=current?state.snapshot.conditions.k:live?null:n.demo_available,cap=current?state.snapshot.conditions.capacity:live?null:n.demo_capacity;
 box.innerHTML=`<button data-close-map="true" aria-label="Close node information">×</button><strong>${esc(n.id)} · ${esc(n.name)}</strong><p>${esc(n.kind==='Scenario rooftop'?'Rooftop node':n.kind)}${current?' · subsequent node':''}</p><p>Charging: <b>${k==null?'Not supplied':`${k} available`}</b>${cap==null?'':` · capacity ${cap}`}</p>`;
 }else{const seg=state.route.segments.find(x=>x.id===state.segment),active=seg.id===m?.segment_id;
 box.innerHTML=`<button data-close-map="true" aria-label="Close segment information">×</button><strong>${esc(seg.from_node)} → ${esc(seg.to_node)}</strong><p>${active?'Current segment · swarm shown on map':'Route context · current mission unchanged'}</p>${active?`<div>${state.snapshot.drones.map(d=>`<button class="map-drone-button ${state.drone===d.id?'active':''}" data-drone="${d.id}">${d.id}</button>`).join('')}</div>`:''}`;
 }}
function renderDetails(){const s=state.snapshot,d=selectedDecision(),m=s?.mission;$('details-body').hidden=!state.expanded;$('detail-content').hidden=!state.expanded;$('details-toggle').textContent=state.expanded?'Hide details':'View details';$('details-toggle').setAttribute('aria-expanded',String(state.expanded));
 document.querySelectorAll('[data-tab]').forEach(b=>{b.classList.toggle('active',b.dataset.tab===state.tab);b.setAttribute('aria-selected',String(b.dataset.tab===state.tab));});
 if(!state.expanded)return;if(!m){$('detail-content').innerHTML='<div class="empty">No provider records available.</div>';return;}
 let content='';const past=state.decision&&d?.id!==s.decision?.id;
 const context=`<div class="detail-description"><span>${d?`${esc(d.id)} · ${esc(d.segment_id)} · interval ${d.interval} · inputs ${clock(d.input_as_of)} · ${esc(d.model_version)}`:'No decision returned'}${past?' · Historical view':''}</span>${past||state.interval?'<button data-current-details="true">Return to current decision</button>':''}</div>`;
 if(state.tab==='history'){
  const rows=(s.history??[]).filter(x=>!state.interval||(x.interval===state.interval&&x.segment_id===m.segment_id));
  content=`<div class="detail-description"><span>${state.interval?`Inspecting interval ${state.interval} · `:''}Wind and pad availability show the inputs recorded for each decision.</span>${state.interval?'<button data-current-details="true">Show all decisions</button>':''}</div><table><thead><tr><th>Decision / input time</th><th>Segment / interval</th><th>Trigger</th><th>Wind condition</th><th>Pads at next node</th><th>Configuration</th><th>Ready at next node¹</th><th>Command status</th></tr></thead><tbody>${rows.map(r=>`<tr><td><button class="table-button" data-decision="${esc(r.id)}">${esc(r.id)}</button><br><small>${clock(r.input_as_of)}</small></td><td>${esc(r.segment_id)} / ${esc(r.interval)}</td><td>${esc(r.trigger)}</td><td>${esc(recordedWind(r))}</td><td>${esc(recordedPads(r))}</td><td>${esc(configName(r.recommended?.config_id))}</td><td>${duration(r.recommended?.ready_s)}</td><td>${statusTag(r.status)}</td></tr>`).join('')}</tbody></table>${!rows.length?'<div class="empty">No recorded decisions in this interval.</div>':''}<p class="forecast-note">¹ Estimate from the recorded inputs, for the selected recommendation if applied.</p>`;
 }else if(state.tab==='candidates'){
  content=context+`<div class="detail-description">${d?.candidates?.length??0} returned candidates · provider list setting for ${d?.k??'—'} available pads: ${d?.candidate_limit??'—'}. Times use the recorded decision inputs.</div><table><thead><tr><th>Rank</th><th>Configuration</th><th>Position assignment</th><th>Arrival in</th><th>Charging after arrival</th><th>Ready at next node</th></tr></thead><tbody>${(d?.candidates??[]).map(c=>`<tr class="${c.id===d.recommended?.id?'table-highlight':''}"><td>${c.rank}${c.id===d.recommended?.id?' · Selected':''}</td><td>${esc(configName(c.config_id))}</td><td>${Object.entries(c.assignment).map(([dr,p])=>`<button class="table-button ${state.drone===dr?'selected-row':''}" data-drone="${dr}">${dr}→${p}</button>`).join(' · ')}</td><td>${duration(c.arrival_s)}</td><td>${duration(c.charging_s)}</td><td>${duration(c.ready_s)}</td></tr>`).join('')}</tbody></table>`;
 }else if(state.tab==='charging'){
  const f=past?d?.recommended:s.forecast,rows=f?.schedule??[],charging=['Charging','Ready at node'].includes(m.status)&&!past;
  content=context+`<div class="detail-description"><span>${past?'Recommended plan · historical':charging?'Current node · execution feedback shown separately':'Current executing plan · expected arrangement at '+esc(m.target_node)}. Charge target 99%. Planned groups are not reservations.</span></div><table><thead><tr><th>Drone</th><th>Planned group</th><th>Expected wait</th><th>Charging duration</th><th>Completion after arrival</th><th>Execution feedback</th></tr></thead><tbody>${rows.map(r=>`<tr class="${state.drone===r.drone_id?'selected-row':''}"><td><button class="table-button" data-drone="${r.drone_id}">${r.drone_id}</button></td><td>Group ${r.group}</td><td>${duration(r.wait_s)}</td><td>${duration(r.charge_s)}</td><td>${duration(r.end_s)}</td><td>${charging?statusTag(s.drones.find(x=>x.id===r.drone_id)?.charging_state??'Unavailable'):'Not started'}</td></tr>`).join('')}</tbody></table><p class="forecast-note">Whole-swarm charging time includes waiting. It is not the sum of the five completion times.</p>`;
 }else content=`<div class="activity-list">${s.events.map(e=>`<div class="activity-row"><time>${clock(e.at)}</time><b>${esc(e.title)}</b><span>${esc(e.detail)}</span></div>`).join('')}</div>`;
 $('detail-content').innerHTML=content;
}
function sourceContent(){return `<h3>Geographic background</h3><p>Street map: <a href="https://www.openstreetmap.de/" target="_blank" rel="noreferrer">OpenStreetMap.de</a> tiles. A local rendering of the ${esc(state.route.map_retrieved)} extract from <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap contributors</a> is used if tiles cannot load. Map credit appears on the page.</p><p>Endpoint references: <a href="https://internationalstudents.health.nsw.gov.au/healthcareservice/sydney-central-medical-centre/" target="_blank" rel="noreferrer">Sydney Central Medical Centre</a> and <a href="https://www.nsw.gov.au/health-and-wellbeing/health-infrastructure-projects/royal-prince-alfred-hospital-redevelopment" target="_blank" rel="noreferrer">Royal Prince Alfred Hospital</a>. These locate the facilities, not surveyed landing points.</p><h3>Rooftop nodes</h3><p>${state.route.nodes.filter(n=>n.osm_way).map(n=>`<a href="${n.location_source}" target="_blank" rel="noreferrer">${n.name} · OSM ${n.osm_way}</a>`).join('<br>')}<br>Rooftop use, charging facilities and route approval are unverified. Capacities in Demo are example settings.</p><h3>Demo and replay</h3><p>Battery, wind, pad availability and execution feedback are scripted. Candidate lists and discharge rates are demo fixtures. Charging times use an illustrative ${esc(state.snapshot?.settings?.demo_charge_curve_minutes)}-minute curve scale; this is not a measured charging result. Estimates are calculated by the demo provider; the trained research model is not connected. Replay contains a recorded demo session.</p><h3>Distance and formation</h3><p>The demo uses 25-second intervals and 2.5 metres per indoor interval. No geographic-to-indoor scale is applied. Geographic swarm progress is illustrative. Formation positions use the project's existing layout definitions; D1–D5 identities remain fixed. The views show target layouts, not measured flight positions.</p><h3>Live connection</h3><p>Live reads a separately configured provider snapshot service. With no service connected, readings and estimates remain unavailable. This local prototype sends no aircraft control commands.</p>`;}

document.addEventListener('click',event=>{
 const drone=event.target.closest('[data-drone]');if(drone){state.drone=state.drone===drone.dataset.drone?null:drone.dataset.drone;render();return;}
 const act=event.target.closest('[data-action]');if(act){action(act.dataset.action);return;}
 const tab=event.target.closest('[data-tab]');if(tab){state.tab=tab.dataset.tab;state.expanded=true;renderDetails();return;}
 const dec=event.target.closest('[data-decision]');if(dec){state.decision=dec.dataset.decision;state.tab='candidates';renderDetails();return;}
 const interval=event.target.closest('[data-interval]');if(interval){state.interval=Number(interval.dataset.interval);state.expanded=true;state.tab='history';render();return;}
 if(event.target.closest('[data-current-details]')){state.decision=null;state.interval=null;render();}
 if(event.target.closest('[data-close-map]')){state.node=null;state.segment=null;renderMapSelection();}
});
$('route-map').addEventListener('click',e=>{if(state.justDragged)return;const node=e.target.closest('[data-node]'),segment=e.target.closest('[data-segment]'),swarm=e.target.closest('[data-swarm]');if(node){state.node=node.dataset.node;state.segment=null;}else if(segment||swarm){state.segment=segment?.dataset.segment??state.snapshot?.mission?.segment_id;state.node=null;}else{return;}renderMap();});
document.addEventListener('keydown',e=>{const button=e.target.closest('svg [role="button"]');if(button&&['Enter',' '].includes(e.key)){e.preventDefault();button.dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
$('mode').addEventListener('change',()=>{state.mode=$('mode').value;state.generation++;state.snapshot={mode:state.mode,connection:'Connecting',reason:'Waiting for provider data.'};state.decision=null;state.interval=null;state.node=null;state.segment=null;state.playing=false;$('replay-play').textContent='Play recording';busy=false;render();loadSnapshot();});
$('auto-button').addEventListener('click',()=>action('auto'));
$('details-toggle').addEventListener('click',()=>{state.expanded=!state.expanded;renderDetails();});
$('sources-btn').addEventListener('click',()=>{$('source-content').innerHTML=sourceContent();$('source-dialog').showModal();});
$('close-sources').addEventListener('click',()=>$('source-dialog').close());
$('replay-play').addEventListener('click',()=>{state.playing=!state.playing;if(state.replay===state.frames.length-1)state.replay=0;$('replay-play').textContent=state.playing?'Pause recording':'Play recording';});
$('replay-range').addEventListener('input',()=>{state.replay=Number($('replay-range').value);state.generation++;busy=false;loadSnapshot();});
function zoom(f){if(realMapReady){realMap.setZoom(realMap.getZoom()+(f<1?1:-1));return;}let [x,y,w,h]=state.view;const nw=Math.min(1650,Math.max(350,w*f)),nh=nw*650/1100;state.view=[x+(w-nw)/2,y+(h-nh)/2,nw,nh];$('route-map').setAttribute('viewBox',state.view.join(' '));}
$('zoom-in').addEventListener('click',()=>zoom(.8));$('zoom-out').addEventListener('click',()=>zoom(1.25));$('map-fit').addEventListener('click',()=>{if(realMapReady){fitRealMap();return;}state.view=[0,0,1100,650];renderMap();});
$('route-map').addEventListener('pointerdown',e=>{state.drag={x:e.clientX,y:e.clientY,view:[...state.view]};state.justDragged=false;});
window.addEventListener('pointermove',e=>{if(!state.drag)return;const d=state.drag,rect=$('route-map').getBoundingClientRect(),dx=e.clientX-d.x,dy=e.clientY-d.y;if(Math.abs(dx)+Math.abs(dy)>5)state.justDragged=true;if(!state.justDragged)return;state.view=[d.view[0]-dx*d.view[2]/rect.width,d.view[1]-dy*d.view[3]/rect.height,d.view[2],d.view[3]];$('route-map').classList.add('dragging');$('route-map').setAttribute('viewBox',state.view.join(' '));});
window.addEventListener('pointerup',()=>{state.drag=null;$('route-map').classList.remove('dragging');setTimeout(()=>state.justDragged=false,50);});
async function init(){try{const [route,frames]=await Promise.all([fetch('/api/route').then(r=>r.json()),fetch('/api/replay').then(r=>r.json())]);state.route=route;state.frames=frames.frames;$('replay-range').max=state.frames.length-1;initRealMap();await loadSnapshot();fitRealMap();setInterval(()=>{if(state.mode!=='Replay'&&!(state.mode==='Demo'&&figureMode))loadSnapshot();},1000);setInterval(()=>{if(state.mode==='Replay'&&state.playing){if(state.replay<state.frames.length-1){state.replay++;loadSnapshot();}else{state.playing=false;$('replay-play').textContent='Play recording';}}},1600);}catch(error){$('mode-banner').hidden=false;$('mode-banner').className='mode-banner warning';$('mode-banner').textContent='Unable to load the local provider. Start the platform service and reload.';}}
init();
