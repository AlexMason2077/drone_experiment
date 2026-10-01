"""Compose the two GUI states without resampling their original screenshots."""
from base64 import b64decode, b64encode
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parents[2]
SHOTS = ROOT / 'decision_support/screenshots'
OUTPUT = ROOT / 'output/pdf/gui-two-interval-comparison.pdf'
NS = {'s': 'http://www.w3.org/2000/svg'}
WIDTH, HEIGHT = 3660, 3700
SOURCE_X = (WIDTH - 2989) / 2
UNIT = 1 / 6


def main():
    annotated = SHOTS / 'annotations/gui-interval-03-panel-annotations.svg'
    source_a = SHOTS / 'gui-interval-03-2989x1720.png'
    source_b = SHOTS / 'gui-interval-04-2989x1720.png'
    svg_text = annotated.read_text()
    svg = ET.fromstring(svg_text)
    image_a = svg.find('s:image', NS)
    assert b64decode(image_a.attrib['href'].split(',', 1)[1]) == source_a.read_bytes()

    # Tighten the annotation gutters so the GUI occupies more of one paper column.
    image_a.set('x', str(SOURCE_X))
    svg.set('width', str(WIDTH))
    svg.find('s:rect', NS).set('width', str(WIDTH))
    for panel in svg.findall('s:g', NS):
        box = panel.find('s:rect', NS)
        box.set('x', str(float(box.attrib['x']) + SOURCE_X - 450))
        dot = panel.find('s:circle', NS)
        dot.set('cx', str(float(dot.attrib['cx']) + SOURCE_X - 450))
        arrow = panel.find('s:path', NS)
        x1, y1, x2, y2 = [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?', arrow.attrib['d'])]
        left = x2 < x1
        arrow.set('d', f'M{x1 + SOURCE_X - 450},{y1} L{270 if left else 3325},{y2}')
        label = panel.find('s:text', NS)
        label.set('x', str(240 if left else 3350))
        for line in label.findall('s:tspan', NS):
            line.set('x', label.attrib['x'])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont('GUIArialBold', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
    canvas = Canvas(str(OUTPUT), pagesize=(WIDTH * UNIT, HEIGHT * UNIT), pageCompression=1)
    canvas.setTitle('Decision-support platform: consecutive intervals')
    canvas.scale(UNIT, UNIT)
    canvas.setFillColor(HexColor('#ffffff'))
    canvas.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    canvas.drawImage(ImageReader(str(source_a)), SOURCE_X, HEIGHT - 60 - 1720, 2989, 1720)
    canvas.drawImage(ImageReader(str(source_b)), SOURCE_X, HEIGHT - 1900 - 1720, 2989, 1720)

    for panel in svg.findall('s:g', NS):
        box = panel.find('s:rect', NS).attrib
        color = HexColor(box['stroke'])
        canvas.setStrokeColor(color)
        canvas.setLineWidth(float(box['stroke-width']))
        x, y, w, h = (float(box[k]) for k in ('x', 'y', 'width', 'height'))
        canvas.rect(x, HEIGHT - y - h, w, h, fill=0, stroke=1)

        dot = panel.find('s:circle', NS).attrib
        canvas.setFillColor(color)
        canvas.circle(float(dot['cx']), HEIGHT - float(dot['cy']), float(dot['r']), fill=1, stroke=0)
        arrow = panel.find('s:path', NS).attrib
        coords = [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?', arrow['d'])]
        x1, y1, x2, y2 = coords
        canvas.setLineWidth(float(arrow['stroke-width']))
        canvas.line(x1, HEIGHT - y1, x2, HEIGHT - y2)
        direction = 1 if x2 > x1 else -1
        head = canvas.beginPath()
        head.moveTo(x2, HEIGHT - y2)
        head.lineTo(x2 - direction * 24, HEIGHT - y2 + 12)
        head.lineTo(x2 - direction * 24, HEIGHT - y2 - 12)
        head.close()
        canvas.drawPath(head, fill=1, stroke=0)

        text = panel.find('s:text', NS)
        canvas.setFillColor(HexColor('#111111'))
        canvas.setFont('GUIArialBold', float(text.attrib['font-size']))
        tx, ty = float(text.attrib['x']), float(text.attrib['y'])
        draw_text = canvas.drawRightString if text.attrib['text-anchor'] == 'end' else canvas.drawString
        for line in text.findall('s:tspan', NS):
            ty += float(line.attrib['dy'])
            draw_text(tx, HEIGHT - ty, line.text)

    canvas.setFillColor(HexColor('#111111'))
    canvas.setFont('GUIArialBold', 46)
    canvas.drawCentredString(WIDTH / 2, HEIGHT - 1825, '(a) Interval 3')
    canvas.drawCentredString(WIDTH / 2, HEIGHT - 3680, '(b) Interval 4')
    canvas.showPage()
    canvas.save()

    # Keep the same native images and editable vector annotations in a companion SVG.
    svg.set('height', str(HEIGHT))
    svg.set('viewBox', f'0 0 {WIDTH} {HEIGHT}')
    svg.find('s:rect', NS).set('height', str(HEIGHT))
    svg.find('s:title', NS).text = 'Decision-support platform: consecutive intervals'
    svg.find('s:desc', NS).text = 'Top: annotated interval 3. Bottom: unannotated interval 4. Both screenshots retain their original dimensions and alignment.'
    ET.SubElement(svg, f'{{{NS["s"]}}}image', {
        'x': str(SOURCE_X), 'y': '1900', 'width': '2989', 'height': '1720',
        'href': 'data:image/png;base64,' + b64encode(source_b.read_bytes()).decode('ascii'),
    })
    for y, label in ((1825, '(a) Interval 3'), (3680, '(b) Interval 4')):
        ET.SubElement(svg, f'{{{NS["s"]}}}text', {
            'x': str(WIDTH / 2), 'y': str(y), 'text-anchor': 'middle',
            'font-family': 'Arial,Helvetica,sans-serif', 'font-size': '46',
            'font-weight': '700', 'fill': '#111111',
        }).text = label
    ET.register_namespace('', NS['s'])
    ET.ElementTree(svg).write(SHOTS / 'annotations/gui-two-interval-comparison.svg', encoding='utf-8')
    print(OUTPUT)


if __name__ == '__main__':
    main()
