"""Create the uncluttered, attributed SVG from a cached OpenStreetMap extract."""
from pathlib import Path
import xml.etree.ElementTree as ET
import gzip
import json
import math

ROOT = Path(__file__).resolve().parents[1]
with gzip.open(ROOT / 'data/sydney_osm_expanded.xml.gz', 'rb') as stream:
    root = ET.parse(stream).getroot()
nodes = {n.attrib['id']: (float(n.attrib['lon']), float(n.attrib['lat'])) for n in root.findall('node')}
ways = {}
for w in root.findall('way'):
    points = [nodes[n.attrib['ref']] for n in w.findall('nd') if n.attrib['ref'] in nodes]
    ways[w.attrib['id']] = {'points': points, 'tags': {t.attrib['k']: t.attrib['v'] for t in w.findall('tag')}}

# A local geographic projection, unrelated to the indoor experiment scale.
coslat = math.cos(math.radians(-33.8845))
scale = 1100 / (.055 * coslat)
def project(lon, lat):
    return [round((lon - 151.167) * coslat * scale, 2), round((-33.870 - lat) * scale, 2)]
def path(points):
    return 'M' + ' L'.join(','.join(map(str, project(*p))) for p in points)
layers = {'green': [], 'buildings': [], 'roads': [], 'minor': [], 'rail': []}
widths = {'motorway': 9, 'trunk': 8, 'primary': 8, 'secondary': 6, 'tertiary': 5, 'residential': 3.5, 'unclassified': 3.5, 'service': 2, 'pedestrian': 3}
for wid, w in ways.items():
    t, p = w['tags'], w['points']
    if len(p) < 2: continue
    d = path(p)
    if t.get('leisure') in ('park', 'garden', 'pitch') or t.get('landuse') in ('grass', 'recreation_ground'):
        layers['green'].append(f'<path d="{d}Z" fill="#dfebdf"/>')
    elif 'building' in t:
        layers['buildings'].append(f'<path d="{d}Z" fill="#e4e8e7" stroke="#d3dad8" stroke-width=".65"/>')
    elif t.get('highway') in widths:
        width = widths[t['highway']]
        layers['roads'].append(f'<path d="{d}" stroke="#d5dcda" stroke-width="{width+1.6}"/><path d="{d}" stroke="#fff" stroke-width="{width}"/>')
    elif t.get('highway') in ('footway', 'path', 'cycleway'):
        layers['minor'].append(f'<path d="{d}" stroke="#fdfefd" stroke-width="1.2"/>')
    elif t.get('railway') == 'rail':
        layers['rail'].append(f'<path d="{d}" stroke="#cbd4d1" stroke-width=".8"/>')
svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 650"><rect width="1100" height="650" fill="#f0f3f0"/>']
for name in ['green', 'buildings', 'rail', 'minor', 'roads']:
    svg += ['<g fill="none" stroke-linejoin="round" stroke-linecap="round">', *layers[name], '</g>']
svg.append('</svg>')
(ROOT/'static/map-base.svg').write_text(''.join(svg))

def roof(node_id, name, way_id, capacity):
    w = ways[way_id]; p = w['points'][:-1]
    # Centroid of the polygon area (not a surveyed landing point).
    cross = [p[i][0]*p[(i+1)%len(p)][1]-p[(i+1)%len(p)][0]*p[i][1] for i in range(len(p))]
    area = sum(cross)
    lon = sum((p[i][0]+p[(i+1)%len(p)][0])*cross[i] for i in range(len(p)))/(3*area)
    lat = sum((p[i][1]+p[(i+1)%len(p)][1])*cross[i] for i in range(len(p)))/(3*area)
    return dict(id=node_id, name=name, kind='Scenario rooftop', lon=lon, lat=lat, xy=project(lon,lat), footprint=[project(*q) for q in w['points']], osm_way=way_id,
        location_source=f'https://www.openstreetmap.org/way/{way_id}', facility_status='Rooftop use unverified', demo_capacity=capacity, demo_available=max(1,capacity-2))
route_nodes = [dict(id='N0', name='Sydney Central Medical Centre', kind='Origin', lon=151.20654296875, lat=-33.87984085083,
    xy=project(151.20654296875,-33.87984085083), location_source='https://internationalstudents.health.nsw.gov.au/healthcareservice/sydney-central-medical-centre/', facility_status='Public address reference', demo_capacity=None, demo_available=None),
    roof('N1','Rooftop A','548918549',4), roof('N2','Rooftop B','369353658',6), roof('N3','Rooftop C','205131017',3),
    dict(id='N4', name='Royal Prince Alfred Hospital',kind='Destination',lon=151.183123,lat=-33.8893087,
    xy=project(151.183123,-33.8893087),location_source='https://www.nsw.gov.au/health-and-wellbeing/health-infrastructure-projects/royal-prince-alfred-hospital-redevelopment',facility_status='Public address reference',demo_capacity=6,demo_available=4)]
data = dict(id='SCENARIO-CENTRAL-RPA',name='Central to RPA',scenario=True,nodes=route_nodes,
    segments=[dict(id=f'S{i+1}',from_node=f'N{i}',to_node=f'N{i+1}',interval_count=8,geographic_distance_m=None) for i in range(4)],
    map_source='OpenStreetMap contributors',map_source_url='https://www.openstreetmap.org/copyright',map_retrieved='2026-09-27',
    geographic_scale_to_indoor=None,description='Illustrative predefined route. Endpoint map references and OSM building footprints are geographic context. Rooftop suitability, charging facilities and route approval are unverified. Demo capacities are scenario fixtures.')
(ROOT/'data/route.json').write_text(json.dumps(data,indent=2))
print('Built map and five-node scenario route')
