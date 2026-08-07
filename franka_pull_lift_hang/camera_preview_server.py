#!/usr/bin/env python3
"""Small dependency-light HTTP preview for Isaac Lab RGB camera tensors."""

from __future__ import annotations

import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from urllib.parse import urlparse

from PIL import Image


CAMERA_LABELS = {
    "left_wrist": "Left Franka wrist",
    "right_wrist": "Right Franka wrist",
    "hanger_front": "Hanger front",
}


def _preview_html() -> bytes:
    cards = "".join(
        f'<section><h2>{label}</h2><img data-camera="{name}" alt="{label}"></section>'
        for name, label in CAMERA_LABELS.items()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Dual Franka camera views</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: #11151b; color: #edf2f7; }}
    header {{ padding: 18px 22px; border-bottom: 1px solid #313946; }}
    h1 {{ margin: 0; font-size: 20px; }}
    p {{ margin: 6px 0 0; color: #9eabbc; }}
    main {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; padding: 14px; }}
    section {{ overflow: hidden; border: 1px solid #313946; border-radius: 10px; background: #1a2029; }}
    section:last-child {{ grid-column: 1 / -1; }}
    h2 {{ margin: 0; padding: 10px 12px; font-size: 15px; font-weight: 600; }}
    img {{ display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #05070a; }}
    @media (max-width: 760px) {{ main {{ grid-template-columns: 1fr; }} section:last-child {{ grid-column: auto; }} }}
  </style>
</head>
<body>
  <header><h1>Dual Franka camera views</h1><p>Newton physics · Isaac RTX RGB sensors</p></header>
  <main>{cards}</main>
  <script>
    const refresh = () => document.querySelectorAll('img[data-camera]').forEach((img) => {{
      img.src = `/frame/${{img.dataset.camera}}?t=${{Date.now()}}`;
    }});
    refresh();
    setInterval(refresh, 100);
  </script>
</body>
</html>"""
    return html.encode("utf-8")


class _PreviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            body = _preview_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        elif path.startswith("/frame/"):
            name = path.removeprefix("/frame/")
            body = self.server.preview.get_frame(name)  # type: ignore[attr-defined]
            if body is None:
                self.send_error(503, "Camera frame is not ready")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store, max-age=0")
        else:
            self.send_error(404)
            return
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *args):
        return
class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True




class CameraPreviewServer:
    """Serve the latest RGB tensors as a browser-friendly three-view page."""

    def __init__(self, port: int, jpeg_quality: int = 82):
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self._frames: dict[str, bytes] = {}
        self._lock = Lock()
        self._server = _ReusableThreadingHTTPServer(("0.0.0.0", self.port), _PreviewHandler)
        self._server.daemon_threads = True
        self._server.preview = self
        self._thread = Thread(target=self._server.serve_forever, name="camera-preview", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)

    def get_frame(self, name: str) -> bytes | None:
        with self._lock:
            return self._frames.get(name)

    def update(self, cameras):
        encoded = {}
        for name, camera in cameras.items():
            rgb = camera.data.output.get("rgb")
            if rgb is None:
                continue
            array = rgb[0].detach().to("cpu").numpy()
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format="JPEG", quality=self.jpeg_quality)
            encoded[name] = buffer.getvalue()
        if encoded:
            with self._lock:
                self._frames.update(encoded)
