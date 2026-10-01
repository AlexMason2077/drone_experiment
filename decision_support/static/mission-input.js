const $ = id => document.getElementById(id);
const STORAGE_KEY = 'swarm-mission-input-v1';
const blank = () => ({origin:null,stops:[],destination:null,batteries:[null,null,null,null,null]});
let draft = blank();
let selected = 'origin';
let saved = false;
let map, routeLayer, referenceLayer, routeData;

function validPoint(point){
  return point && Number.isFinite(point.lat) && Number.isFinite(point.lon) && Math.abs(point.lat)<=90 && Math.abs(point.lon)<=180;
}
function validBattery(value){return Number.isInteger(value) && value>=0 && value<=100;}
function validDraft(){return validPoint(draft.origin) && validPoint(draft.destination) && draft.batteries.every(validBattery);}
function pointName(point){
  if(!point)return 'Choose on map';
  if(!routeData)return 'Selected map point';
  const nearest=routeData.nodes.map(node=>({node,metres:map.distance([point.lat,point.lon],[node.lat,node.lon])})).sort((a,b)=>a.metres-b.metres)[0];
  return nearest.metres<90?nearest.node.name:'Selected map point';
}
function coordinates(point){return `${point.lat.toFixed(5)}, ${point.lon.toFixed(5)}`;}
function routePoints(){return [draft.origin,...draft.stops,draft.destination].filter(validPoint);}

function makeRouteRow(kind,point,index){
  const li=document.createElement('li');
  if(!point)li.className='empty';
  const badge=document.createElement('span');badge.className='point-badge';badge.textContent=kind==='origin'?'A':kind==='destination'?'B':String(index+1);
  const labels=document.createElement('div');
  const title=document.createElement('strong');title.textContent=kind==='origin'?'Start':kind==='destination'?'Finish':`Stop ${index+1}`;
  const detail=document.createElement('small');detail.textContent=point?pointName(point):'Choose on map';
  if(point){const location=document.createElement('span');location.className='point-coordinates';location.textContent=coordinates(point);detail.append(location);}
  labels.append(title,detail);li.append(badge,labels);
  if(point){
    const remove=document.createElement('button');remove.className='remove-point';remove.type='button';remove.textContent='×';remove.setAttribute('aria-label',`Remove ${title.textContent}`);
    remove.addEventListener('click',()=>{if(kind==='origin'||kind==='destination')draft[kind]=null;else draft.stops.splice(index,1);saved=false;render();});
    li.append(remove);
  }
  return li;
}

function renderRouteList(){
  const list=$('route-list');list.replaceChildren();
  list.append(makeRouteRow('origin',draft.origin,0));
  draft.stops.forEach((point,i)=>list.append(makeRouteRow('stop',point,i)));
  list.append(makeRouteRow('destination',draft.destination,0));
  const count=Number(Boolean(draft.origin))+Number(Boolean(draft.destination));
  $('route-status').textContent=`${count} of 2 required points${draft.stops.length?` · ${draft.stops.length} stop${draft.stops.length===1?'':'s'}`:''}`;
}

function renderMap(){
  if(!map||!routeLayer)return;
  routeLayer.clearLayers();
  const points=routePoints();
  if(points.length>1){
    const latlngs=points.map(p=>[p.lat,p.lon]);
    window.L.polyline(latlngs,{color:'#fff',weight:9,opacity:.94,interactive:false}).addTo(routeLayer);
    window.L.polyline(latlngs,{color:'#176c5a',weight:5,opacity:.98,interactive:false}).addTo(routeLayer);
  }
  const marked=[['A',draft.origin,'Start'],...draft.stops.map((p,i)=>[String(i+1),p,`Stop ${i+1}`]),['B',draft.destination,'Finish']];
  marked.filter(([,point])=>validPoint(point)).forEach(([badge,point,name])=>{
    const size=window.innerWidth>=2200?64:32;
    const icon=window.L.divIcon({className:'',html:`<span class="chosen-point">${badge}</span>`,iconSize:[size,size],iconAnchor:[size/2,size/2]});
    window.L.marker([point.lat,point.lon],{icon,interactive:false}).bindTooltip(name,{permanent:true,direction:'top',offset:[0,size>=64?-36:-14],className:'input-node-tooltip'}).addTo(routeLayer);
  });
}

function setMode(mode){
  selected=mode;
  for(const key of ['origin','stop','destination']){
    const button=$(`pick-${key}`);button.classList.toggle('active',key===mode);button.setAttribute('aria-pressed',String(key===mode));
  }
  const instructions={origin:['Choose the starting point','Click anywhere on the map to set the start.'],stop:['Add a stop','Click the map to add a point along the route.'],destination:['Choose the destination','Click anywhere on the map to set the finish.']};
  const [title,detail]=instructions[mode];
  $('map-hint-title').textContent=title;$('map-hint-detail').textContent=detail;
  $('map-instruction').textContent=detail;
}

function placePoint(latlng){
  const point={lat:latlng.lat,lon:latlng.lng};
  if(selected==='stop')draft.stops.push(point);else draft[selected]=point;
  saved=false;render();
  if(selected==='origin'&&!draft.destination)setMode('destination');
}

function renderBatteryInputs(){
  const container=$('battery-inputs');
  for(let i=0;i<5;i++){
    const row=document.createElement('label');row.className='battery-row-input';
    const name=document.createElement('span');name.className='drone-id';name.textContent=`D${i+1}`;
    const track=document.createElement('span');track.className='battery-track';const fill=document.createElement('span');track.append(fill);
    const entry=document.createElement('span');entry.className='battery-entry';
    const input=document.createElement('input');input.type='number';input.min='0';input.max='100';input.step='1';input.inputMode='numeric';input.placeholder='—';input.value=validBattery(draft.batteries[i])?String(draft.batteries[i]):'';input.setAttribute('aria-label',`D${i+1} initial battery percentage`);
    const percent=document.createElement('span');percent.textContent='%';entry.append(input,percent);row.append(name,track,entry);container.append(row);
    fill.style.width=validBattery(draft.batteries[i])?`${draft.batteries[i]}%`:'0%';
    input.addEventListener('input',()=>{
      const value=input.value.trim();const number=Number(value);
      draft.batteries[i]=/^\d{1,3}$/.test(value)&&validBattery(number)?number:null;
      input.setCustomValidity(value && draft.batteries[i]===null?'Enter a whole percentage from 0 to 100.':'');
      fill.style.width=validBattery(draft.batteries[i])?`${draft.batteries[i]}%`:'0%';
      saved=false;renderSaveState();
    });
  }
}

function renderSaveState(){
  const complete=validDraft();$('save-inputs').disabled=!complete;
  $('save-status').textContent=saved?'Saved as an editable mission input draft.':complete?'Ready to save · route and all five battery levels are set.':!draft.origin||!draft.destination?'Choose a start and finish on the map.':'Enter 0–100% for each of the five drones.';
}
function render(){renderRouteList();renderMap();renderSaveState();}

function loadDraft(){
  try{
    const stored=JSON.parse(localStorage.getItem(STORAGE_KEY));
    if(!stored||!Array.isArray(stored.batteries)||stored.batteries.length!==5||!Array.isArray(stored.stops))return;
    const point=p=>validPoint(p)?{lat:p.lat,lon:p.lon}:null;
    draft={origin:point(stored.origin),destination:point(stored.destination),stops:stored.stops.map(point).filter(Boolean),batteries:stored.batteries.map(v=>validBattery(v)?v:null)};
    saved=validDraft();
  }catch{draft=blank();}
}

function fitMap(){
  const points=routePoints();
  const locations=points.length>1?points:routeData.nodes;
  const bounds=window.L.latLngBounds(locations.map(p=>[p.lat,p.lon]));
  map.invalidateSize();
  const wide=window.innerWidth>850;
  const sidebar=document.querySelector('.input-sidebar').getBoundingClientRect();
  const header=document.querySelector('.input-header').getBoundingClientRect();
  map.fitBounds(bounds,{paddingTopLeft:[wide?sidebar.right+38:35,wide?header.bottom+14:200],paddingBottomRight:[60,85],maxZoom:window.innerWidth>=2200?16.6:15.6});
}

function initMap(){
  const L=window.L;
  map=L.map('input-map',{zoomControl:false,attributionControl:false,scrollWheelZoom:true,preferCanvas:true,zoomSnap:.25,minZoom:11,maxZoom:18});
  const scale=1100/(.055*Math.cos(-33.8845*Math.PI/180));
  const imageBounds=L.latLngBounds([[-33.870-650/scale,151.167],[-33.870,151.222]]);
  const fallback=L.imageOverlay('/static/map-base.svg',imageBounds).addTo(map);
  const tiles=L.tileLayer('https://tile.openstreetmap.de/tiles/osmde/{z}/{x}/{y}.png',{minZoom:11,maxZoom:18});
  tiles.once('load',()=>map.removeLayer(fallback));tiles.addTo(map);
  referenceLayer=L.layerGroup().addTo(map);routeLayer=L.layerGroup().addTo(map);
  routeData.nodes.forEach((node,i)=>{
    const medical=i===0||i===routeData.nodes.length-1;
    const marker=L.circleMarker([node.lat,node.lon],{radius:medical?7:6,color:'#fff',weight:2,fillColor:medical?'#344b54':'#779386',fillOpacity:.95,bubblingMouseEvents:false}).addTo(referenceLayer);
    const labelOffset=window.innerWidth>=2200?38:10;
    marker.bindTooltip(`${node.id} · ${node.name}`,{permanent:true,direction:i===0?'left':'right',offset:i===0?[-labelOffset,0]:[labelOffset,0],className:'input-node-tooltip'});
    marker.on('click',()=>placePoint(L.latLng(node.lat,node.lon)));
  });
  map.on('click',event=>placePoint(event.latlng));
  fitMap();
  window.addEventListener('resize',()=>map.invalidateSize());
}

async function init(){
  loadDraft();renderBatteryInputs();setMode(draft.origin?'destination':'origin');
  try{const response=await fetch('/api/route');if(!response.ok)throw Error();routeData=await response.json();initMap();render();}
  catch{$('map-hint-title').textContent='Map unavailable';$('map-hint-detail').textContent='Reload the page to select a route.';}
}

for(const mode of ['origin','stop','destination'])$(`pick-${mode}`).addEventListener('click',()=>setMode(mode));
$('clear-route').addEventListener('click',()=>{draft.origin=null;draft.stops=[];draft.destination=null;saved=false;setMode('origin');render();});
$('save-inputs').addEventListener('click',()=>{
  if(!validDraft())return;
  try{localStorage.setItem(STORAGE_KEY,JSON.stringify({...draft,saved_at:new Date().toISOString()}));saved=true;renderSaveState();}
  catch{$('save-status').textContent='Could not save in this browser. Check storage permissions.';}
});
$('zoom-in').addEventListener('click',()=>map?.zoomIn());
$('zoom-out').addEventListener('click',()=>map?.zoomOut());
$('map-fit').addEventListener('click',()=>{if(map&&routeData)fitMap();});
init();
