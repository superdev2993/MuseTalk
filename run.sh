#!/bin/bash
# MuseTalk runner for muse_talk conda environment
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/administrator/miniconda3/envs/muse_talk/bin/python"
export PATH="/home/administrator/miniconda3/envs/muse_talk/bin:$PATH"

cd "$PROJECT_DIR"

usage() {
    cat <<EOF
Usage: ./run.sh <command>

Commands:
  inference       Run MuseTalk 1.5 normal inference (sample data)
  realtime        Run MuseTalk 1.5 realtime inference
  stream          Live MJPEG stream (config file, http://127.0.0.1:8080)
  stream-web      Upload web UI + live streaming (http://127.0.0.1:8080)
  gradio          Launch Gradio web UI (http://127.0.0.1:7860)
  gradio-fp16     Launch Gradio web UI with fp16 (faster, less VRAM)

Examples:
  ./run.sh inference
  ./run.sh stream-web
  ./run.sh gradio-fp16
EOF
}

case "${1:-}" in
  inference)
    "$PYTHON" -m scripts.inference \
      --inference_config configs/inference/test.yaml \
      --result_dir results/test \
      --unet_model_path models/musetalkV15/unet.pth \
      --unet_config models/musetalkV15/musetalk.json \
      --version v15 \
      --use_float16
    ;;
  realtime)
    "$PYTHON" -m scripts.realtime_inference \
      --inference_config configs/inference/realtime.yaml \
      --result_dir results/realtime \
      --unet_model_path models/musetalkV15/unet.pth \
      --unet_config models/musetalkV15/musetalk.json \
      --version v15 \
      --fps 25
    ;;
  stream)
    "$PYTHON" -m scripts.stream_inference \
      --inference_config configs/inference/stream.yaml \
      --unet_model_path models/musetalkV15/unet.pth \
      --unet_config models/musetalkV15/musetalk.json \
      --version v15 \
      --fps 25 \
      --loop
    ;;
  stream-web)
    "$PYTHON" -m scripts.stream_web_app \
      --unet_model_path models/musetalkV15/unet.pth \
      --unet_config models/musetalkV15/musetalk.json \
      --version v15 \
      --fps 25 \
      --host 0.0.0.0 \
      --port 8080
    ;;
  gradio)
    "$PYTHON" app.py --ip 0.0.0.0 --port 7860
    ;;
  gradio-fp16)
    "$PYTHON" app.py --use_float16 --ip 0.0.0.0 --port 7860
    ;;
  *)
    usage
    exit 1
    ;;
esac
