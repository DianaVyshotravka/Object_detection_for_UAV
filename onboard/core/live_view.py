"""LiveView -- optional local web page: annotated video + live detection map.

    http://<host>:<port>/                  page
    http://<host>:<port>/stream            MJPEG of the newest frame, boxes drawn
    http://<host>:<port>/detections.json   GeoJSON of recent geotagged detections
    http://<host>:<port>/images.json       newest saved images of this run
    http://<host>:<port>/images/<name>     one saved image (view / download)
    http://<host>:<port>/thumbs/<name>     its thumbnail
    http://<host>:<port>/download.zip      the whole run folder (images + detections.jsonl)

stdlib http.server, no Flask: a handful of routes do not need a framework. The
pipeline only hands over a reference to the frame (publish() is O(1)); drawing
and JPEG encoding happen in the viewer's request thread at STREAM_FPS, so a
connected browser costs the inference loop nothing and no browser costs zero.

Map points are the DRONE's position when the objects were seen, not the
objects' ground positions -- projecting a box to the ground needs camera
intrinsics and attitude.
"""

import json
import re
import zipfile
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STREAM_FPS = 5          # per viewer; the Pi encodes this many JPEGs per second
STREAM_WIDTH = 960      # downscaled for the network; detection runs on the full frame
MAX_POINTS = 1000       # map history kept in memory
CPU_TEMP_PATH = "/sys/class/thermal/thermal_zone0/temp"   # millidegrees C
GALLERY_SIZE = 60       # newest saved images listed on the page
IMAGE_NAME = re.compile(r"\d{6,}\.jpg")  # what ImageSaver writes; nothing else is served

# ponytail: Leaflet + OSM tiles from the internet, like build_map.py. A field
# laptop with no uplink gets the video and an empty map; ship Leaflet and a
# tile cache in the deploy if that matters.
PAGE = """<!doctype html><meta charset="utf-8"><title>UAV live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
body{margin:0;font:14px sans-serif;background:#111;color:#eee;display:flex;flex-wrap:wrap}
#video,#mapbox{flex:1 1 480px;height:calc(100vh - 28px);min-height:300px}
#video{object-fit:contain;background:#000}#status{width:100%;padding:6px 10px}
#mapbox{position:relative}#map{height:100%}
#telem{position:absolute;top:10px;right:10px;z-index:1000;background:rgba(0,0,0,.75);
  padding:8px 10px;border-radius:6px;font:13px monospace;line-height:1.5;white-space:pre}
#galleryhead{width:100%;padding:10px 10px 0}
#gallery{width:100%;padding:10px;display:grid;gap:8px;
  grid-template-columns:repeat(auto-fill,minmax(240px,1fr))}
#gallery figure{margin:0;background:#222;border-radius:4px;overflow:hidden}
#gallery img{width:100%;display:block}
#gallery figcaption{display:flex;justify-content:space-between;padding:4px 8px;font-size:12px}
a{color:#8ab4f8}
</style>
<div id="status">connecting...</div>
<img id="video" src="/stream" alt="live detections">
<div id="mapbox"><div id="map"></div><div id="telem">no telemetry</div></div>
<div id="galleryhead">Saved images (newest first) |
  <a href="/download.zip" download>download whole run (zip)</a></div>
<div id="gallery"></div>
<script>
const map = L.map('map').setView([0, 0], 2);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 19, attribution: '&copy; OpenStreetMap'}).addTo(map);
const points = L.geoJSON(null, {
  pointToLayer: (f, ll) => L.circleMarker(ll, {radius: 6, color: '#d93025', weight: 2}),
  onEachFeature: (f, l) => l.bindPopup(`${f.properties.time}<br>${f.properties.summary}` +
    (f.properties.image ? `<br><a href="/images/${f.properties.image}" target="_blank">` +
      `<img src="/thumbs/${f.properties.image}" width="240"></a>` : ''))
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
    const fps = data.fps == null ? '-' : data.fps.toFixed(1);
    const temp = data.cpu_temp_c == null ? '-' : data.cpu_temp_c.toFixed(1) + ' °C';
    document.getElementById('status').textContent =
      `detector ${fps} FPS | Pi ${temp} | ${data.points.features.length} detection points` +
      (p ? '' : ' | no GPS');
    const f = (v, d, u) => v == null ? '-' : v.toFixed(d) + u;
    // Age in browser time: the Pi and laptop clocks may differ, so the server
    // sends how old the sample is instead of when it was taken.
    document.getElementById('telem').textContent = p ?
      `Lat      ${p.lat.toFixed(7)}\nLon      ${p.lon.toFixed(7)}\n` +
      `Alt MSL  ${f(p.alt_msl_m, 1, ' m')}\nAlt rel  ${f(p.rel_alt_m, 1, ' m')}\n` +
      `Heading  ${f(p.heading_deg, 1, '°')}\nTime     ${data.position_time}\n` +
      `Age      ${f(data.position_age_s, 1, ' s')}\nFPS      ${fps}\nPi temp  ${temp}` :
      `no telemetry\nFPS      ${fps}\nPi temp  ${temp}`;
  } catch (e) { document.getElementById('status').textContent = 'service unreachable'; }
}
let shown = '';
async function gallery() {
  try {
    const names = await (await fetch('/images.json')).json();
    if (names.join() === shown) return;  // unchanged: keep loaded thumbnails
    shown = names.join();
    document.getElementById('gallery').innerHTML = names.map(n =>
      `<figure><a href="/images/${n}" target="_blank"><img src="/thumbs/${n}" loading="lazy"></a>` +
      `<figcaption>${n}<a href="/images/${n}" download>download</a></figcaption></figure>`).join('');
  } catch (e) {}
}
poll(); setInterval(poll, 2000); gallery(); setInterval(gallery, 5000);
</script>"""


def cpu_temp_c():
    """SoC temperature, or None off a Pi (dev box, CI)."""
    try:
        with open(CPU_TEMP_PATH) as f:
            return int(f.read()) / 1000.0
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=GALLERY_SIZE * 2)
def thumbnail(path: Path, mtime: float):
    """Small JPEG of a saved image. Decoding at 1/4 scale is a few ms on the Pi;
    saved files never change, so (path, mtime) caches safely."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    return None if image is None else cv2.imencode(".jpg", image)[1].tobytes()


def draw_detections(image, detections) -> None:
    """Boxes plus "class conf" labels, in place on a BGR image. Line width and
    font scale with the image, so a 1920 px saved frame reads like the stream."""
    import cv2  # only a viewer or a saved frame pays for it

    k = max(1, round(image.shape[1] / STREAM_WIDTH))
    for det in detections:
        x1, y1, x2, y2 = (round(v) for v in det["bbox_xyxy"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2 * k)
        cv2.putText(image, "%s %.2f" % (det["class"], det["conf"]), (x1, max(y1 - 4 * k, 12 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, (0, 0, 255), k, cv2.LINE_AA)


def annotate(image, detections, fps=None, width=STREAM_WIDTH):
    """Downscaled copy of the BGR frame with boxes (and detector FPS), as JPEG bytes."""
    import cv2  # only a viewer pays for it

    scale = width / image.shape[1]
    out = cv2.resize(image, (width, round(image.shape[0] * scale)))
    draw_detections(out, [dict(det, bbox_xyxy=[v * scale for v in det["bbox_xyxy"]])
                          for det in detections])
    if fps is not None:
        text = "%.1f FPS" % fps
        for color, thick in (((0, 0, 0), 4), ((0, 255, 0), 2)):  # outline, then fill
            cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thick, cv2.LINE_AA)
    return cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


class LiveView:
    def __init__(self, host: str, port: int, stop_event: threading.Event, logger,
                 image_dir: Path = None):
        self.stop_event = stop_event
        self.image_dir = Path(image_dir) if image_dir else None  # this run's saved images
        self.logger = logger
        self._cond = threading.Condition()
        self._latest = None          # (seq, image, detections)
        self._seq = 0
        self._position = None
        self._position_time = None   # record timestamp (UTC string) of the position
        self._position_mono = None   # monotonic time it arrived, for the age
        self._fps = None             # frames the pipeline publishes per second, smoothed
        self._last_publish = None
        self._points = deque(maxlen=MAX_POINTS)

        view = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?")[0]
                route = {"/": view._page, "/stream": view._stream,
                         "/detections.json": view._geojson,
                         "/images.json": view._image_list,
                         "/download.zip": view._zip}.get(path)
                if route is not None:
                    route(self)
                elif path.startswith(("/images/", "/thumbs/")):
                    view._image(self, path.split("/")[2], path.startswith("/thumbs/"))
                else:
                    self.send_error(404)

            def log_message(self, *args):
                pass  # one line per MJPEG frame would flood the journal

        # Bind here, not in run(): a port clash surfaces at startup.
        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.server.timeout = 0.5    # handle_request() returns, run() sees stop
        logger.log_info(f"live view: http://{host}:{port}/")

    # --- producer side (pipeline thread) ------------------------------------

    def publish(self, image, record: dict) -> None:
        now = time.monotonic()
        with self._cond:
            if self._last_publish is not None and now > self._last_publish:
                rate = 1.0 / (now - self._last_publish)
                self._fps = rate if self._fps is None else 0.9 * self._fps + 0.1 * rate
            self._last_publish = now
            self._seq += 1
            self._latest = (self._seq, image, record["detections"], self._fps)
            telemetry = record.get("telemetry")
            if telemetry:
                self._position = telemetry
                self._position_time = record["timestamp"]
                self._position_mono = now
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
                "image": record.get("image"),
            },
        }

    # --- routes (request threads) -------------------------------------------

    @staticmethod
    def _send(handler, body: bytes, content_type: str) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _page(self, handler) -> None:
        self._send(handler, PAGE.encode(), "text/html; charset=utf-8")

    def _image_list(self, handler) -> None:
        # Names are zero-padded frame ids, so name order is capture order.
        names = (sorted((p.name for p in self.image_dir.glob("*.jpg")), reverse=True)
                 if self.image_dir else [])
        self._send(handler, json.dumps(names[:GALLERY_SIZE]).encode(), "application/json")

    def _image(self, handler, name: str, thumb: bool) -> None:
        # Strict name match: no "..", no slashes, nothing outside image_dir.
        path = self.image_dir / name if self.image_dir and IMAGE_NAME.fullmatch(name) else None
        try:
            body = (thumbnail(path, path.stat().st_mtime) if thumb else path.read_bytes()) if path else None
        except OSError:
            body = None
        if body is None:
            handler.send_error(404)
        else:
            self._send(handler, body, "image/jpeg")

    def _geojson(self, handler) -> None:
        with self._cond:
            data = {"points": {"type": "FeatureCollection", "features": list(self._points)},
                    "position": self._position, "position_time": self._position_time,
                    "position_age_s": (None if self._position_mono is None
                                       else round(time.monotonic() - self._position_mono, 1)),
                    "fps": self._fps, "cpu_temp_c": cpu_temp_c()}
        self._send(handler, json.dumps(data).encode(), "application/json")

    def _zip(self, handler) -> None:
        """Stream the run folder as a zip. ZIP_STORED: JPEGs are already
        compressed, so this is disk -> network with no CPU for the detector to
        lose. No Content-Length (size unknown up front); HTTP/1.0 ends the
        response by closing the connection."""
        if self.image_dir is None:
            handler.send_error(404)
            return
        handler.send_response(200)
        handler.send_header("Content-Type", "application/zip")
        handler.send_header("Content-Disposition",
                            f'attachment; filename="{self.image_dir.name}.zip"')
        handler.end_headers()
        try:
            with zipfile.ZipFile(handler.wfile, "w", zipfile.ZIP_STORED) as zf:
                for path in sorted(self.image_dir.iterdir()):
                    if path.is_file() and path.suffix != ".tmp":  # skip half-written images
                        zf.write(path, path.name)
        except (BrokenPipeError, ConnectionResetError):
            pass  # viewer cancelled the download

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
            seen, image, detections, fps = latest
            jpeg = annotate(image, detections, fps)
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
