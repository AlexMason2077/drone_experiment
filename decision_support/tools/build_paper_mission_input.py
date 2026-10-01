"""Add vector panel annotations around the unchanged mission-input screenshot."""
from base64 import b64encode
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "decision_support/screenshots/mission-input-paper-2989x1720.jpg"
OUTPUT = ROOT / "output/pdf/mission-input-panel-annotations.pdf"
SVG_OUTPUT = ROOT / "decision_support/screenshots/annotations/mission-input-panel-annotations.svg"
WIDTH, HEIGHT, SOURCE_Y = 3660, 1760, 20
SOURCE_X = (WIDTH - 2989) / 2
UNIT = 1 / 6
# Match Fig. 6's 49 px text on its 3660 px canvas at the same column width.
FONT_SIZE = 49 * WIDTH / 3660
LINE_GAP = 57 * WIDTH / 3660
LABEL_X, ARROW_END = 280, 310
PANELS = [
    {"name": "Flight route panel", "lines": ["Flight", "route", "panel"],
     "box": (28, 201, 1160, 706), "color": "#1259c7"},
    {"name": "Initial battery level panel", "lines": ["Initial", "battery", "level", "panel"],
     "box": (28, 923, 1160, 572), "color": "#009d58"},
]


def main():
    with Image.open(SOURCE) as image:
        assert image.size == (2989, 1720), image.size
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SVG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for name, filename in (("PanelName", "Arial Bold Italic.ttf"), ("PanelWord", "Arial Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, "/System/Library/Fonts/Supplemental/" + filename))
    for panel in PANELS:
        for line in panel["lines"]:
            assert pdfmetrics.stringWidth(line, "PanelName", FONT_SIZE) < LABEL_X - 10

    canvas = Canvas(str(OUTPUT), pagesize=(WIDTH * UNIT, HEIGHT * UNIT), pageCompression=1)
    canvas.setTitle("Mission input interface: Flight route and Initial battery level panels")
    canvas.scale(UNIT, UNIT)
    canvas.setFillColor(HexColor("#ffffff"))
    canvas.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    canvas.drawImage(str(SOURCE), SOURCE_X, HEIGHT - SOURCE_Y - 1720, 2989, 1720)

    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    svg = ET.Element(f"{{{ns}}}svg", {"width": str(WIDTH), "height": str(HEIGHT),
        "viewBox": f"0 0 {WIDTH} {HEIGHT}", "role": "img"})
    ET.SubElement(svg, f"{{{ns}}}title").text = "Mission input interface panels"
    ET.SubElement(svg, f"{{{ns}}}rect", {"width": str(WIDTH), "height": str(HEIGHT), "fill": "white"})
    ET.SubElement(svg, f"{{{ns}}}image", {"x": str(SOURCE_X), "y": str(SOURCE_Y),
        "width": "2989", "height": "1720", "href": "data:image/jpeg;base64," + b64encode(SOURCE.read_bytes()).decode("ascii")})

    for panel in PANELS:
        px, py, width, height = panel["box"]
        x, y = SOURCE_X + px, SOURCE_Y + py
        center = y + height / 2
        dot_x = x + 13
        color = HexColor(panel["color"])
        canvas.setStrokeColor(color)
        canvas.setFillColor(color)
        canvas.setLineWidth(5)
        canvas.rect(x, HEIGHT - y - height, width, height, fill=0, stroke=1)
        canvas.circle(dot_x, HEIGHT - center, 11, fill=1, stroke=0)
        canvas.setLineWidth(8)
        canvas.line(dot_x, HEIGHT - center, ARROW_END, HEIGHT - center)
        arrow = canvas.beginPath()
        arrow.moveTo(ARROW_END, HEIGHT - center)
        arrow.lineTo(ARROW_END + 24, HEIGHT - center + 12)
        arrow.lineTo(ARROW_END + 24, HEIGHT - center - 12)
        arrow.close()
        canvas.drawPath(arrow, fill=1, stroke=0)

        group = ET.SubElement(svg, f"{{{ns}}}g", {"aria-label": panel["name"]})
        ET.SubElement(group, f"{{{ns}}}rect", {"x": str(x), "y": str(y), "width": str(width),
            "height": str(height), "fill": "none", "stroke": panel["color"], "stroke-width": "5"})
        ET.SubElement(group, f"{{{ns}}}circle", {"cx": str(dot_x), "cy": str(center), "r": "11", "fill": panel["color"]})
        ET.SubElement(group, f"{{{ns}}}path", {"d": f"M{dot_x},{center} L{ARROW_END},{center}",
            "fill": "none", "stroke": panel["color"], "stroke-width": "8"})
        ET.SubElement(group, f"{{{ns}}}path", {"d": f"M{ARROW_END},{center} L{ARROW_END + 24},{center - 12} L{ARROW_END + 24},{center + 12} Z", "fill": panel["color"]})
        first_baseline = center - (len(panel["lines"]) - 1) * LINE_GAP / 2 + FONT_SIZE * 0.34
        canvas.setFillColor(HexColor("#111111"))
        for index, line in enumerate(panel["lines"]):
            baseline = first_baseline + index * LINE_GAP
            font = "PanelWord" if line == "panel" else "PanelName"
            canvas.setFont(font, FONT_SIZE)
            canvas.drawRightString(LABEL_X, HEIGHT - baseline, line)
            ET.SubElement(group, f"{{{ns}}}text", {"x": str(LABEL_X), "y": str(baseline),
                "text-anchor": "end", "font-family": "Arial,Helvetica,sans-serif", "font-size": str(FONT_SIZE),
                "font-weight": "700", "font-style": "normal" if line == "panel" else "italic", "fill": "#111111"}).text = line

    canvas.showPage()
    canvas.save()
    ET.ElementTree(svg).write(SVG_OUTPUT, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
