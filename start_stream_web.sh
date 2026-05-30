#!/bin/bash
# Start MuseTalk streaming web UI (default port 7860 — same as Gradio)
set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/venv/muse_talk/bin/python"
LOG="/tmp/stream_web_app.log"
PIDFILE="/tmp/stream_web_app.pid"
PORT="${PORT:-7860}"

cd "$PROJECT_DIR"
export PATH="/venv/muse_talk/bin:$PATH"
NVIDIA_LIB_DIRS="$(ls -d /venv/muse_talk/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:)"
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:/venv/muse_talk/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

for p in 7860 8080 8000; do fuser -k "${p}/tcp" 2>/dev/null || true; done
sleep 1

nohup "$PYTHON" -m scripts.stream_web_app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --fps 12 \
  --progressive_mode piped \
  --fmp4_frag_us 40000 \
  --tts_gpu \
  --stream_batch_size 8 \
  --stream_first_batch_size 4 \
  --stream_emit_frames 1 \
  --batch_size 32 \
  >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"

IP=$(hostname -I | awk '{print $1}')
echo "MuseTalk web server starting (PID $(cat "$PIDFILE"), port $PORT)"
echo "Log: $LOG"
echo ""
echo "Open ONE of these URLs in your browser:"
echo "  http://127.0.0.1:${PORT}/"
echo "  http://${IP}:${PORT}/"
echo ""
echo "If using Cursor Remote SSH:"
echo "  1. Open the Ports panel (bottom bar)"
echo "  2. Forward port ${PORT} if not listed"
echo "  3. Click the forwarded localhost link"
echo ""
echo "Waiting for models to load..."
for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    STATUS=$(curl -sf "http://127.0.0.1:${PORT}/api/status" 2>/dev/null || echo "")
    echo "Server responding: $STATUS"
    echo "$STATUS" | grep -q '"idle"' && echo "Ready!" && exit 0
  fi
  sleep 2
done
echo "Server started but models may still be loading. Check log: $LOG"
