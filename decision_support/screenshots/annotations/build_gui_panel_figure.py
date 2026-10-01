from pathlib import Path
from base64 import b64encode
from html import escape

root=Path(__file__).resolve().parent.parent
source=root/'gui-interval-03-2989x1720.png'
encoded=b64encode(source.read_bytes()).decode('ascii')
width,height=3889,1840
sx,sy=450,60
panels=[
 ('Mission overview panel','#111111',(38,20,2913,149.5),[(488+13,148),(360,148)],(330,96),['Mission','overview','panel'],'end'),
 ('Time estimation panel','#df202d',(38,187.5,2913,223.7),[(3401-13,360),(3525,360)],(3560,308),['Time','estimation','panel'],'start'),
 ('Swarm status panel','#009d58',(38,431.2,780,1139.3),[(488+13,1030),(360,1030)],(330,978),['Swarm','status','panel'],'end'),
 ('Route visualization panel','#1259c7',(840,431.2,2111,1139.3),[(3401-13,970),(3525,970)],(3560,918),['Route','visualization','panel'],'start'),
 ('Decision details panel','#7547a2',(38,1592.54,2913,108),[(3401-13,1706),(3525,1706)],(3560,1654),['Decision','details','panel'],'start'),
]
defs=[];layers=[]
for i,(name,color,box,points,pos,lines,anchor) in enumerate(panels):
 x,y,w,h=box
 defs.append(f'<marker id="arrow-{i}" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="3.1" markerHeight="3.1" orient="auto"><path d="M0 0 L12 6 L0 12 Z" fill="{color}"/></marker>')
 path='M'+' L'.join(f'{x},{y}' for x,y in points)
 tx,ty=pos
 texts=''.join(f'<tspan x="{tx}" dy="{0 if j==0 else 57}">{escape(line)}</tspan>' for j,line in enumerate(lines))
 layers.append(f'''<g aria-label="{escape(name)}">
 <rect x="{sx+x}" y="{sy+y}" width="{w}" height="{h}" fill="none" stroke="{color}" stroke-width="5"/>
 <circle cx="{points[0][0]}" cy="{points[0][1]}" r="11" fill="{color}"/>
 <path d="{path}" fill="none" stroke="{color}" stroke-width="8" marker-end="url(#arrow-{i})"/>
 <text x="{tx}" y="{ty}" text-anchor="{anchor}" font-family="Arial,Helvetica,sans-serif" font-size="49" font-weight="700" fill="#111111">{texts}</text>
 </g>''')
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
 <title id="title">Decision-support platform panels</title>
 <desc id="desc">Interval 3: five named panels outlined in colour, with outward arrows and external labels. The original GUI screenshot is embedded at its native resolution.</desc>
 <defs>{''.join(defs)}</defs>
 <rect width="{width}" height="{height}" fill="white"/>
 <image x="{sx}" y="{sy}" width="2989" height="1720" href="data:image/png;base64,{encoded}"/>
 {''.join(layers)}
</svg>'''
folder=root/'annotations'
(folder/'gui-interval-03-panel-annotations.svg').write_text(svg)
(folder/'gui-interval-03-panel-annotations.html').write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Decision-support platform panel annotations</title><style>html,body{{margin:0;background:#fff;width:{width}px;height:{height}px;overflow:hidden}}svg{{display:block}}</style></head><body>{svg}</body></html>')
print(width,height)
