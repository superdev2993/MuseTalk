"""
Web server: upload video + audio, stream lip-sync output to the browser (MJPEG).

Face analysis results are cached per video file. Re-uploading the same video
with a different audio skips avatar preparation and starts streaming faster.

Open http://127.0.0.1:8080/ after starting this script.
"""

import argparse
import cgi
import hashlib
import mimetypes
import json
import os
import shutil
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from transformers import WhisperModel

from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import load_all_model
from scripts.realtime_inference import Avatar, fast_check_ffmpeg

UPLOAD_ROOT = "./results/stream_web/uploads"
CACHE_INDEX_PATH = "./results/stream_web/cache_index.json"


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MuseTalk Streaming</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; font-family: system-ui, sans-serif; background: #0f1115; color: #eef2f7; }
    main { max-width: 980px; margin: 0 auto; padding: 24px; }
    h1 { margin-bottom: 8px; font-size: 1.6rem; }
    .sub { color: #9aa4b2; margin-bottom: 24px; line-height: 1.5; }
    .panel { background: #171b22; border: 1px solid #2a3140; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
    label { display: block; margin: 12px 0 6px; font-weight: 600; }
    input[type=file] { width: 100%; }
    button { margin-top: 16px; background: #3b82f6; color: white; border: 0; border-radius: 8px; padding: 12px 18px; font-size: 1rem; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    #status { min-height: 1.4em; color: #93c5fd; margin-top: 12px; white-space: pre-wrap; }
    .hint { color: #64748b; font-size: 0.9rem; margin-top: 8px; }
    .stream-wrap { background: #000; border-radius: 12px; overflow: hidden; min-height: 360px; display: flex; align-items: center; justify-content: center; position: relative; }
    #stream { width: 100%; display: block; background: #000; min-height: 360px; object-fit: contain; }
    .placeholder { color: #64748b; padding: 48px; text-align: center; position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; }
    #player { width: 100%; margin-top: 12px; }
  </style>
</head>
<body>
  <main>
    <h1>MuseTalk Live Streaming</h1>
    <p class="sub">Upload a reference video and a driving audio clip to preview lip-synced output in your browser.<br>
    Reusing the same video skips face analysis and only processes the new audio.</p>

    <section class="panel">
      <form id="upload-form">
        <label for="video">Reference video (mp4, etc.)</label>
        <input id="video" name="video" type="file" accept="video/*" required>

        <label for="audio">Driving audio (wav, mp3, etc.)</label>
        <input id="audio" name="audio" type="file" accept="audio/*" required>

        <button id="submit-btn" type="submit">Start streaming</button>
        <div id="status">Connecting to server...</div>
        <div class="hint">First upload: face analysis takes 1–3 min · Same video again: only audio is processed</div>
      </form>
      <audio id="player" controls style="display:none"></audio>
    </section>

    <section class="panel stream-wrap">
      <img id="stream" alt="stream preview" style="display:none">
      <div id="placeholder" class="placeholder">The stream will appear here after upload.</div>
    </section>
  </main>

  <script>
    const form = document.getElementById('upload-form');
    const statusEl = document.getElementById('status');
    const submitBtn = document.getElementById('submit-btn');
    const streamImg = document.getElementById('stream');
    const placeholder = document.getElementById('placeholder');
    const audioEl = document.getElementById('player');
    let pollTimer = null;
    let streamToken = 0;
    let audioStarted = false;
    let activeJobId = null;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function startStreamView() {
      streamToken += 1;
      placeholder.style.display = 'none';
      streamImg.style.display = 'block';
      streamImg.src = '/stream?t=' + streamToken;
    }

    function resetAudio() {
      audioStarted = false;
      activeJobId = null;
      audioEl.pause();
      audioEl.removeAttribute('src');
      audioEl.load();
      audioEl.style.display = 'none';
    }

    function tryStartAudio(data) {
      if (audioStarted || !data.audio_url) return;
      if (!['preparing', 'streaming'].includes(data.state)) return;
      audioEl.style.display = 'block';
      if (!audioEl.src || !audioEl.src.endsWith(data.audio_url.split('?')[0])) {
        audioEl.src = data.audio_url + '?t=' + Date.now();
        audioEl.load();
      }
      if (data.state === 'streaming' && audioEl.paused) {
        audioEl.play().catch(() => {
          setStatus((data.message || 'Streaming') + ' — click Play on the audio bar below to hear sound.');
        });
        audioStarted = true;
      }
    }

    async function pollStatus() {
      try {
        const res = await fetch('/api/status', { cache: 'no-store' });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        setStatus(data.message || data.state);
        submitBtn.disabled = ['preparing', 'streaming', 'loading', 'uploading'].includes(data.state);
        if (data.job_id && data.job_id !== activeJobId) {
          activeJobId = data.job_id;
          audioStarted = false;
        }
        if (['preparing', 'streaming'].includes(data.state)) {
          if (streamImg.style.display === 'none') startStreamView();
        }
        tryStartAudio(data);
        if (['idle', 'done', 'error', 'cancelled'].includes(data.state)) {
          submitBtn.disabled = false;
        }
      } catch (err) {
        setStatus('Server connection failed: ' + err.message);
        submitBtn.disabled = false;
      }
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const video = document.getElementById('video').files[0];
      const audio = document.getElementById('audio').files[0];
      if (!video || !audio) {
        setStatus('Please select both a video and an audio file.');
        return;
      }

      submitBtn.disabled = true;
      resetAudio();
      setStatus('Uploading files...');
      startStreamView();

      const body = new FormData();
      body.append('video', video);
      body.append('audio', audio);

      try {
        const res = await fetch('/api/start', { method: 'POST', body });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Upload failed');
        activeJobId = data.job_id;
        setStatus(data.message || 'Processing started');
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(pollStatus, 1000);
        pollStatus();
      } catch (err) {
        setStatus('Error: ' + err.message);
        submitBtn.disabled = false;
      }
    });

    pollStatus();
    setInterval(pollStatus, 3000);
  </script>
</body>
</html>
"""


def load_models(args, device):
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device,
    )
    timesteps = torch.tensor([0], device=device)
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width,
        )
    else:
        fp = FaceParsing()

    return vae, unet, pe, timesteps, audio_processor, weight_dtype, whisper, fp


def bind_realtime_globals(args, device, models):
    import scripts.realtime_inference as rt

    vae, unet, pe, timesteps, audio_processor, weight_dtype, whisper, fp = models
    rt.args = args
    rt.device = device
    rt.vae = vae
    rt.unet = unet
    rt.pe = pe
    rt.timesteps = timesteps
    rt.audio_processor = audio_processor
    rt.weight_dtype = weight_dtype
    rt.whisper = whisper
    rt.fp = fp


def compute_file_hash(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def avatar_dir(args, avatar_id):
    if args.version == "v15":
        return f"./results/v15/avatars/{avatar_id}"
    return f"./results/avatars/{avatar_id}"


def load_cache_index():
    if not os.path.exists(CACHE_INDEX_PATH):
        return {}
    with open(CACHE_INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache_index(index):
    os.makedirs(os.path.dirname(CACHE_INDEX_PATH), exist_ok=True)
    with open(CACHE_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def is_avatar_ready(args, avatar_id):
    base = avatar_dir(args, avatar_id)
    required = [
        os.path.join(base, "latents.pt"),
        os.path.join(base, "coords.pkl"),
        os.path.join(base, "mask_coords.pkl"),
        os.path.join(base, "avator_info.json"),
    ]
    return all(os.path.exists(p) for p in required)


class StreamWebService:
    def __init__(self, args):
        self.args = args
        self._lock = threading.Lock()
        self._latest_frame = None
        self._frame_event = threading.Event()
        self._status = {"state": "loading", "message": "Loading models...", "cached": False, "job_id": None, "audio_url": None}
        self._cancel_event = threading.Event()
        self._worker = None
        self._running = False
        self._models_ready = threading.Event()
        self._job_audio_paths = {}

    def set_status(self, state, message, cached=False, job_id=None, audio_url=None):
        with self._lock:
            status = {"state": state, "message": message, "cached": cached}
            if job_id is not None:
                status["job_id"] = job_id
            elif "job_id" in self._status:
                status["job_id"] = self._status.get("job_id")
            if audio_url is not None:
                status["audio_url"] = audio_url
            elif "audio_url" in self._status:
                status["audio_url"] = self._status.get("audio_url")
            self._status = status
        print(f"[{state}] {message}")

    def get_status(self):
        with self._lock:
            return dict(self._status)

    def push_frame(self, frame_bgr: np.ndarray):
        """Store a BGR frame (same format as OpenCV / get_image_blending output)."""
        with self._lock:
            self._latest_frame = frame_bgr.copy()
        self._frame_event.set()

    def push_status_frame(self, text):
        img = np.zeros((480, 854, 3), dtype=np.uint8)
        cv2.putText(
            img,
            text,
            (40, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (200, 220, 255),
            2,
            cv2.LINE_AA,
        )
        self.push_frame(img)

    def _get_latest_jpeg(self, timeout=1.0):
        if not self._frame_event.wait(timeout=timeout):
            return None
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return encoded.tobytes() if ok else None

    def _save_upload(self, form, job_id):
        upload_dir = os.path.join(UPLOAD_ROOT, job_id)
        os.makedirs(upload_dir, exist_ok=True)

        video_item = form["video"] if "video" in form else None
        audio_item = form["audio"] if "audio" in form else None
        if video_item is None or audio_item is None or not video_item.filename or not audio_item.filename:
            raise ValueError("Video and audio files are required.")

        video_ext = os.path.splitext(video_item.filename)[1] or ".mp4"
        audio_ext = os.path.splitext(audio_item.filename)[1] or ".wav"
        video_path = os.path.join(upload_dir, f"video{video_ext}")
        audio_path = os.path.join(upload_dir, f"audio{audio_ext}")

        with open(video_path, "wb") as f:
            f.write(video_item.file.read())
        with open(audio_path, "wb") as f:
            f.write(audio_item.file.read())

        return video_path, audio_path

    def _resolve_avatar(self, video_path):
        video_hash = compute_file_hash(video_path)
        avatar_id = f"cache_{video_hash[:16]}"
        cache_index = load_cache_index()
        cached = cache_index.get(video_hash) or {}
        known_id = cached.get("avatar_id", avatar_id)

        if is_avatar_ready(self.args, known_id):
            return known_id, False, video_hash

        return avatar_id, True, video_hash

    def _register_cache(self, video_hash, avatar_id, video_path):
        cache_index = load_cache_index()
        cache_index[video_hash] = {
            "avatar_id": avatar_id,
            "video_path": os.path.abspath(video_path),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_cache_index(cache_index)

    def _run_job(self, video_path, audio_path, avatar_id, need_preparation, video_hash, use_cache):
        import scripts.realtime_inference as rt

        try:
            if use_cache:
                self.set_status("preparing", "Loading cached face analysis...", cached=True)
                self.push_status_frame("Loading cached avatar...")
            else:
                self.set_status("preparing", "Preparing avatar (face analysis)...", cached=False)
                self.push_status_frame("Preparing avatar...")

            avatar = Avatar(
                avatar_id=avatar_id,
                video_path=video_path,
                bbox_shift=0,
                batch_size=self.args.batch_size,
                preparation=need_preparation,
                non_interactive=True,
            )

            if need_preparation and not self._cancel_event.is_set():
                self._register_cache(video_hash, avatar_id, video_path)

            if self._cancel_event.is_set():
                self.set_status("cancelled", "Job cancelled.", cached=use_cache)
                return

            self.set_status("streaming", "Generating lip-sync stream...", cached=use_cache)
            avatar.inference(
                audio_path,
                None,
                self.args.fps,
                skip_save_images=True,
                frame_sink=self,
                stream_fps=self.args.fps,
            )

            if self._cancel_event.is_set():
                self.set_status("cancelled", "Job cancelled.", cached=use_cache)
            else:
                msg = "Streaming finished. Upload the same video with different audio to try again."
                self.set_status("done", msg, cached=use_cache)
        except Exception as exc:
            self.set_status("error", f"Processing error: {exc}", cached=use_cache)
            self.push_status_frame("Error: " + str(exc)[:60])
            raise
        finally:
            rt.args.cancel_event = None

    def start_job(self, video_path, audio_path):
        if not self._models_ready.is_set():
            raise RuntimeError("Models are still loading.")

        if self._worker is not None and self._worker.is_alive():
            self._cancel_event.set()
            self._worker.join(timeout=3)

        avatar_id, need_preparation, video_hash = self._resolve_avatar(video_path)
        use_cache = not need_preparation

        self._cancel_event = threading.Event()
        import scripts.realtime_inference as rt

        rt.args.cancel_event = self._cancel_event

        with self._lock:
            self._latest_frame = None
        self._frame_event.clear()
        self.push_status_frame("Starting...")

        self._worker = threading.Thread(
            target=self._run_job,
            args=(video_path, audio_path, avatar_id, need_preparation, video_hash, use_cache),
            daemon=True,
        )
        self._worker.start()

    def handle_upload(self, handler):
        content_type = handler.headers.get("Content-Type", "")
        content_length = handler.headers.get("Content-Length", "0")
        if "multipart/form-data" not in content_type:
            raise ValueError("multipart/form-data is required.")

        form = cgi.FieldStorage(
            fp=handler.rfile,
            headers=handler.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": content_length,
            },
        )

        job_id = uuid.uuid4().hex
        video_path, audio_path = self._save_upload(form, job_id)
        self._job_audio_paths[job_id] = audio_path
        avatar_id, need_preparation, _ = self._resolve_avatar(video_path)
        audio_url = f"/api/audio/{job_id}"

        if need_preparation:
            message = "Upload complete. Running face analysis, then streaming will start."
        else:
            message = "Upload complete. Using cached face analysis — streaming audio now."

        self.set_status(
            "uploading",
            message,
            cached=not need_preparation,
            job_id=job_id,
            audio_url=audio_url,
        )
        self.start_job(video_path, audio_path)

        return {
            "ok": True,
            "job_id": job_id,
            "avatar_id": avatar_id,
            "cached": not need_preparation,
            "audio_url": audio_url,
            "message": message,
        }

    def serve_forever(self):
        service = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):
                return

            def _send_json(self, payload, status=200):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path

                if path == "/":
                    body = INDEX_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if path == "/api/status":
                    self._send_json(service.get_status())
                    return

                if path.startswith("/api/audio/"):
                    job_id = path.split("/")[-1]
                    audio_path = service._job_audio_paths.get(job_id)
                    if not audio_path or not os.path.isfile(audio_path):
                        self.send_error(404)
                        return
                    mime, _ = mimetypes.guess_type(audio_path)
                    if not mime:
                        mime = "application/octet-stream"
                    with open(audio_path, "rb") as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(data)
                    return

                if path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    while service._running:
                        jpeg = service._get_latest_jpeg(timeout=1.0)
                        if jpeg is None:
                            continue
                        try:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                            break
                    return

                self.send_error(404)

            def do_POST(self):
                if urlparse(self.path).path != "/api/start":
                    self.send_error(404)
                    return
                try:
                    payload = service.handle_upload(self)
                    self._send_json(payload)
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)

        self._running = True
        httpd = ThreadingHTTPServer((self.args.host, self.args.port), Handler)
        httpd.daemon_threads = True
        print(f"MuseTalk streaming web: http://127.0.0.1:{self.args.port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping web server.")
        finally:
            self._running = False
            self._cancel_event.set()
            httpd.shutdown()
            httpd.server_close()


def parse_args():
    parser = argparse.ArgumentParser(description="MuseTalk upload + streaming web server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"])
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/")
    parser.add_argument("--vae_type", type=str, default="sd-vae")
    parser.add_argument("--unet_config", type=str, default="./models/musetalkV15/musetalk.json")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper")
    parser.add_argument("--extra_margin", type=int, default=10)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--audio_padding_length_left", type=int, default=2)
    parser.add_argument("--audio_padding_length_right", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--parsing_mode", default="jaw")
    parser.add_argument("--left_cheek_width", type=int, default=90)
    parser.add_argument("--right_cheek_width", type=int, default=90)
    return parser.parse_args()


def main():
    args = parse_args()

    if not fast_check_ffmpeg():
        path_separator = ";" if sys.platform == "win32" else ":"
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"

    os.makedirs(UPLOAD_ROOT, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    service = StreamWebService(args)
    service.set_status("loading", "Loading models... please wait.")

    def load_and_start():
        models = load_models(args, device)
        bind_realtime_globals(args, device, models)
        service._models_ready.set()
        service.set_status("idle", "Ready. Upload a video and audio file.")
        service.push_status_frame("Ready - upload video and audio")

    threading.Thread(target=load_and_start, daemon=True).start()
    service.serve_forever()


if __name__ == "__main__":
    main()
