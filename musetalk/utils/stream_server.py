import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>MuseTalk Live Stream</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: sans-serif; }
    main { max-width: 960px; margin: 0 auto; padding: 24px; }
    img { width: 100%; border-radius: 8px; background: #000; }
    p { opacity: 0.8; }
  </style>
</head>
<body>
  <main>
    <h1>MuseTalk Live Stream</h1>
    <p>Real-time lip-sync preview. Refresh if the stream stops.</p>
    <img src="/stream" alt="MuseTalk stream">
  </main>
</body>
</html>
"""


class MJPEGStreamServer:
    """Serve the latest generated frame as an MJPEG HTTP stream."""

    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._latest_frame = None
        self._frame_event = threading.Event()
        self._running = False
        self._httpd = None
        self._thread = None

    def push_frame(self, frame_rgb: np.ndarray):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        with self._lock:
            self._latest_frame = frame_bgr
        self._frame_event.set()

    def _get_latest_jpeg(self, timeout=1.0):
        if not self._frame_event.wait(timeout=timeout):
            return None
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return encoded.tobytes() if ok else None

    def start(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                if self.path == "/":
                    payload = INDEX_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                if self.path != "/stream":
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.end_headers()

                while server._running:
                    jpeg = server._get_latest_jpeg(timeout=1.0)
                    if jpeg is None:
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break

        self._running = True
        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        print(f"MJPEG stream: http://{self.host}:{self.port}/")
        print(f"Viewer page:  http://127.0.0.1:{self.port}/")

    def stop(self):
        self._running = False
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
