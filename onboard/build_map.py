"""Build a map of detections from a completed flight.

Standalone, not part of the onboard process: the operator runs it after the
card comes out of the drone. Reads GPS back out of the saved JPEGs' EXIF and
the detections out of detections.jsonl.

Run: python -m onboard.build_map --flight-dir flights
"""

import argparse
import json
from pathlib import Path

from onboard.core.geotag import read_gps

# Leaflet from a CDN rather than folium: folium's whole contribution here is
# generating these ~15 lines, and the tiles need a network either way. The
# .geojson drops out for free and opens offline in QGIS.
HTML = """<!doctype html><meta charset="utf-8"><title>Flight map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{height:100%%;margin:0}</style><div id="map"></div><script>
const data = %s;
const map = L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 19, attribution: '&copy; OpenStreetMap'}).addTo(map);
const layer = L.geoJSON(data, {
  pointToLayer: (f, ll) => L.circleMarker(ll, {radius: 6, color: '#d93025', weight: 2}),
  onEachFeature: (f, l) => l.bindPopup(
    `<b>${f.properties.image}</b><br>${f.properties.summary}<br>alt ${f.properties.alt_m} m`)
}).addTo(map);
const b = layer.getBounds();
b.isValid() ? map.fitBounds(b.pad(0.2)) : map.setView([0, 0], 2);
</script>"""


def collect(flight_dir: Path) -> dict:
    """One GeoJSON point per saved frame that has both GPS and detections."""
    records = {}
    jsonl = flight_dir / "detections.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("image"):
                    records[record["image"]] = record

    features = []
    for image in sorted(flight_dir.glob("*.jpg")):
        gps = read_gps(image)
        if gps is None:
            continue  # frame saved before the first GPS fix
        lat, lon, alt = gps
        detections = records.get(image.name, {}).get("detections", [])
        counts = {}
        for det in detections:
            counts[det["class"]] = counts.get(det["class"], 0) + 1
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]},
            "properties": {
                "image": image.name,
                "alt_m": round(alt, 1),
                "counts": counts,
                "summary": ", ".join(f"{n}x {c}" for c, n in sorted(counts.items())) or "no objects",
                "timestamp": records.get(image.name, {}).get("timestamp"),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flight-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.flight_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.flight_dir}")
    out_dir = args.out_dir or args.flight_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    collection = collect(args.flight_dir)
    geojson = json.dumps(collection, indent=2)
    (out_dir / "track.geojson").write_text(geojson + "\n")
    (out_dir / "map.html").write_text(HTML % json.dumps(collection))
    print(f"{len(collection['features'])} geotagged frames -> {out_dir / 'map.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
