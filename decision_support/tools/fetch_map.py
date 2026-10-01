"""Fetch one bounded OpenStreetMap extract for the local route background.

The OSM map endpoint limits each request to 50,000 nodes, so the display area
is fetched in nine small cells. This is a build-time operation, not a tile or
prefetch service used by the dashboard.
"""

from pathlib import Path
from urllib.parse import urlencode
import gzip
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data/sydney_osm_expanded.xml.gz"
LONGITUDES = (151.167, 151.1853333, 151.2036667, 151.222)
LATITUDES = (-33.897, -33.888, -33.879, -33.870)


def main():
    nodes = {}
    ways = {}
    for west, east in zip(LONGITUDES, LONGITUDES[1:]):
        for south, north in zip(LATITUDES, LATITUDES[1:]):
            bbox = f"{west},{south},{east},{north}"
            url = "https://api.openstreetmap.org/api/0.6/map?" + urlencode({"bbox": bbox})
            response = subprocess.run(
                ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "60", "--user-agent", "DroneSwarmDecisionSupport/0.1 (academic local map build)", url],
                check=True,
                capture_output=True,
            )
            root = ET.fromstring(response.stdout)
            for element in root.findall("node"):
                nodes[element.attrib["id"]] = element
            for element in root.findall("way"):
                ways[element.attrib["id"]] = element
            print(f"Fetched {bbox}: {len(nodes)} unique nodes, {len(ways)} unique ways", flush=True)

    output = ET.Element("osm", {"version": "0.6", "generator": "DroneSwarmDecisionSupport", "copyright": "OpenStreetMap and contributors", "attribution": "https://www.openstreetmap.org/copyright", "license": "https://opendatacommons.org/licenses/odbl/1-0/"})
    ET.SubElement(output, "bounds", {"minlat": str(LATITUDES[0]), "minlon": str(LONGITUDES[0]), "maxlat": str(LATITUDES[-1]), "maxlon": str(LONGITUDES[-1])})
    output.extend(nodes.values())
    output.extend(ways.values())
    temporary = DESTINATION.with_suffix(DESTINATION.suffix + ".tmp")
    with gzip.open(temporary, "wb") as stream:
        ET.ElementTree(output).write(stream, encoding="utf-8", xml_declaration=True)
    temporary.replace(DESTINATION)
    print(f"Saved {DESTINATION}", flush=True)


if __name__ == "__main__":
    main()
