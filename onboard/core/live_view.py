"""LiveView -- optional local web page: annotated video + live detection map.

    http://<host>:<port>/                  page
    http://<host>:<port>/stream            MJPEG of the newest frame, boxes drawn
    http://<host>:<port>/detections.json   GeoJSON of recent geotagged detections

stdlib http.server, no Flask: three routes do not need a framework. The
pipeline only hands over a reference to the frame (publish() is O(1)); drawing
and JPEG encoding happen in the viewer's request thread at STREAM_FPS, so a
connected browser costs the inference loop nothing and no browser costs zero.

Map points are the DRONE's position when the objects were seen, not the
objects' ground positions -- projecting a box to the ground needs camera
intrinsics and attitude.
"""

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STREAM_FPS = 5          # per viewer; the Pi encodes this many JPEGs per second
STREAM_WIDTH = 960      # downscaled for the network; detection runs on the full frame
MAX_POINTS = 1000       # map history kept in memory

# ponytail: Leaflet + OSM tiles from the internet, like build_map.py. A field
# laptop with no uplink gets the video and an empty map; ship Leaflet and a
# tile cache in the deploy if that matters.
PAGE = """<!doctype html><meta charset="utf-8"><title>UAV live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
body{margin:0;font:14px sans-serif;background:#111;color:#eee;display:flex;flex-wrap:wrap}
#video,#map{flex:1 1 480px;height:calc(100vh - 28px);min-height:300px}
#video{object-fit:contain;background:#000}#status{width:100%;padding:6px 10px}
</style>
<div id="status">connecting...</div>
<img id="video" src="/stream" alt="live detections"><div id="map"></div>
<script>
const map = L.map('map').setView([0, 0], 2);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 19, attribution: '&copy; OpenStreetMap'}).addTo(map);
const points = L.geoJSON(null, {
  pointToLayer: (f, ll) => L.circleMarker(ll, {radius: 6, color: '#d93025', weight: 2}),
  onEachFeature: (f, l) => l.bindPopup(`${f.properties.time}<br>${f.properties.summary}`)
}).addTo(map);
const drone = L.circleMarker([0, 0], {radius: 8, color: '#1a73e8', fillOpacity: 0.9});
let centered = false;
async function poll() {
  try {
    const data = await (await fetch('/detections.json')).json();
    points.clearLayers(); points.addData(data.points);
    const p = data.position;
    if (p) {
      drone.setLatLng([p.lat, p.lon]).addTo(map);
      if (!centered) { map.setView([p.lat, p.lon], 17); centered = true; }
    }
    document.getElementById('status').textContent =
      `${data.points.features.length} detection points` +
      (p ? ` | drone ${p.lat.toFixed(6)}, ${p.lon.toFixed(6)}, ${p.rel_alt_m} m AGL` : ' | no GPS');
  } catch (e) { document.getElementById('status').textContent = 'service unreachable'; }
}
poll(); setInterval(poll, 2000);
</script>"""


def annotate(image, detections, width=STREAM_WIDTH):
    """Downscaled copy of the BGR frame with boxes drawn, as JPEG bytes."""
    import cv2  # only a viewer pays for it

    scale = width / image.shape[1]
    out = cv2.resize(image, (width, round(image.shape[0] * scale)))
    for det in detections:
        x1, y1, x2, y2 = (round(v * scale) for v in det["bbox_xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, "%s %.2f" % (det["class"], det["conf"]), (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


class LiveView:
    def __init__(self, host: str, port: int, stop_event: threading.Event, logger):
        self.stop_event = stop_event
        self.logger = logger
        self._cond = threading.Condition()
        self._latest = None          # (seq, image, detections)
        self._seq = 0
        self._position = None
        self._points = deque(maxlen=MAX_POINTS)

        view = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                route = {"/": view._page, "/stream": view._stream,
                         "/detections.json": view._geojson}.get(self.path)
                if route is None:
                    self.send_error(404)
                else:
                    route(self)

            def log_message(self, *args):
                pass  # one line per MJPEG frame would flood the journal

        # Bind here, not in run(): a port clash surfaces at startup.
        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.server.timeout = 0.5    # handle_request() returns, run() sees stop
        logger.log_info(f"live view: http://{host}:{port}/")

    # --- producer side (pipeline thread) ------------------------------------

    def publish(self, image, record: dict) -> None:
        with self._cond:
            self._seq += 1
            self._latest = (self._seq, image, record["detections"])
            telemetry = record.get("telemetry")
            if telemetry:
                self._position = telemetry
                if record["detections"]:
                    self._points.append(self._feature(record, telemetry))
            self._cond.notify_all()

    @staticmethod
    def _feature(record, telemetry) -> dict:
        counts = {}
        for det in record["detections"]:
            counts[det["class"]] = counts.get(det["class"], 0) + 1
        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [telemetry["lon"], telemetry["lat"]]},
            "properties": {
                "time": record["timestamp"],
                "counts": counts,
                "summary": ", ".join(f"{n}x {c}" for c, n in sorted(counts.items())),
            },
        }

    # --- routes (request threads) -------------------------------------------

    def _page(self, handler) -> None:
        body = PAGE.encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _geojson(self, handler) -> None:
        with self._cond:
            data = {"points": {"type": "FeatureCollection", "features": list(self._points)},
                    "position": self._position}
        body = json.dumps(data).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _stream(self, handler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.end_headers()
        seen = 0
        while not self.stop_event.is_set():
            with self._cond:
                self._cond.wait_for(lambda: self._latest and self._latest[0] != seen,
                                    timeout=1.0)
                latest = self._latest
            if latest is None or latest[0] == seen:
                continue
            seen, image, detections = latest
            jpeg = annotate(image, detections)
            try:
                handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                    b"Content-Length: %d\r\n\r\n" % len(jpeg))
                handler.wfile.write(jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return  # viewer closed the tab
            self.stop_event.wait(1.0 / STREAM_FPS)

    # --- thread -------------------------------------------------------------

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.server.handle_request()
        self.server.server_close()
        self.logger.log_info("live view: stopped")
