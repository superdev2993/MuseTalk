#!/bin/bash
# Download default Piper voice models for offline TTS.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_DIR="$PROJECT_DIR/models/piper/en_US-lessac-medium"
BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"

mkdir -p "$VOICE_DIR"
if [[ ! -f "$VOICE_DIR/en_US-lessac-medium.onnx" ]]; then
  echo "Downloading en_US-lessac-medium..."
  wget -q -O "$VOICE_DIR/en_US-lessac-medium.onnx" "$BASE_URL/en_US-lessac-medium.onnx"
  wget -q -O "$VOICE_DIR/en_US-lessac-medium.onnx.json" "$BASE_URL/en_US-lessac-medium.onnx.json"
fi
echo "Piper voice ready: $VOICE_DIR"
