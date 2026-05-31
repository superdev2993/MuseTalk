"""
Web server: upload video + audio, stream lip-sync output to the browser.

Video and audio are muxed with FFmpeg into a fragmented MP4 stream served
via HTTP so the browser plays them in sync from a single <video> element.

Face analysis results are cached per video file (on disk and in memory).
Re-uploading the same reference video reuses the loaded face model without
reloading or re-analyzing.

Open http://127.0.0.1:8080/ after starting this script.
"""

import argparse
import cgi
import hashlib
import mimetypes
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from transformers import WhisperModel

from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.tts import (
    TTS_VOICES,
    VOICE_LABELS,
    chunk_text_for_streaming,
    concat_wavs,
    get_tts_engine,
    prepare_tts_chunk_for_inference,
    resample_wav_for_whisper,
    synthesize_chunk_to_wav,
    synthesize_speech,
    wav_to_pcm_s16le,
)
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_utils import ensure_wav
from scripts.realtime_inference import Avatar, fast_check_ffmpeg

UPLOAD_ROOT = "./results/stream_web/uploads"
CACHE_INDEX_PATH = "./results/stream_web/cache_index.json"

PRESET_MODELS = {
    "model1": "./data/video/women_white.mp4",
    "model2": "./data/video/women_gray.mp4",
    "model3": "./data/video/women_black.mp4",
}


class StepTimer:
    """Log per-step elapsed times for a streaming job."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.t0 = time.time()
        self.last = self.t0
        self.steps = []

    def mark(self, name: str):
        now = time.time()
        step = {
            "step": name,
            "elapsed_ms": round((now - self.last) * 1000, 1),
            "total_ms": round((now - self.t0) * 1000, 1),
        }
        self.steps.append(step)
        self.last = now
        print(
            f"[timing:{self.job_id[:8]}] {name}: +{step['elapsed_ms']}ms (total {step['total_ms']}ms)",
            flush=True,
        )
        return step


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
    input[type=file], textarea, select { width: 100%; box-sizing: border-box; }
    textarea { min-height: 100px; padding: 10px; border-radius: 8px; border: 1px solid #2a3140; background: #0f1115; color: #eef2f7; font: inherit; resize: vertical; }
    select { padding: 8px; border-radius: 8px; border: 1px solid #2a3140; background: #0f1115; color: #eef2f7; }
    .mode-tabs { display: flex; gap: 8px; margin-bottom: 12px; }
    .mode-tab { flex: 1; padding: 10px; border: 1px solid #2a3140; border-radius: 8px; background: #0f1115; color: #9aa4b2; cursor: pointer; text-align: center; }
    .mode-tab.active { border-color: #3b82f6; color: #eef2f7; background: #1a2332; }
    .input-panel { display: none; }
    .input-panel.active { display: block; }
    button { margin-top: 16px; background: #3b82f6; color: white; border: 0; border-radius: 8px; padding: 12px 18px; font-size: 1rem; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    button.btn-secondary { background: #334155; margin-top: 0; padding: 8px 14px; font-size: 0.95rem; white-space: nowrap; }
    button.btn-secondary:not(:disabled):hover { background: #475569; }
    .audio-row { display: flex; gap: 10px; align-items: center; margin-top: 6px; }
    .audio-row input[type=file] { flex: 1; margin: 0; }
    .audio-upload-ok { color: #86efac; }
    #status { min-height: 1.4em; color: #93c5fd; margin-top: 12px; white-space: pre-wrap; }
    #timing-info { color: #86efac; font-size: 1rem; margin-top: 8px; min-height: 1.2em; }
    .hint { color: #64748b; font-size: 0.9rem; margin-top: 8px; }
    .stream-wrap { background: #000; border-radius: 12px; overflow: hidden; min-height: 360px; display: flex; align-items: center; justify-content: center; position: relative; }
    #preview { width: 100%; display: block; background: #000; min-height: 360px; object-fit: contain; }
    #player { width: 100%; display: block; background: #000; min-height: 360px; object-fit: contain; }
    .placeholder { color: #64748b; padding: 48px; text-align: center; position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; }
    .preset-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 8px; }
    .preset-option { padding: 12px; border: 1px solid #2a3140; border-radius: 8px; background: #0f1115; cursor: pointer; text-align: center; }
    .preset-option.selected { border-color: #3b82f6; background: #1a2332; }
    .preset-option strong { display: block; margin-bottom: 4px; }
    .preset-option span { color: #64748b; font-size: 0.85rem; }
  </style>
</head>
<body>
  <main>
    <h1>MuseTalk Live Streaming</h1>
    <p class="sub">Choose a preset model or upload your own reference video, then enter text (local TTS) or an audio file.<br>
    Text is chunked for low latency. Progressive fMP4 (video+audio) is served at <code>/api/progressive/{job_id}.mp4</code> while generating.</p>

    <section class="panel">
      <form id="upload-form">
        <label>Reference video</label>
        <div class="mode-tabs" id="video-mode-tabs">
          <button type="button" class="mode-tab active" data-video-mode="preset">Preset model</button>
          <button type="button" class="mode-tab" data-video-mode="custom">Upload video</button>
        </div>

        <div id="panel-video-preset" class="input-panel active">
          <div class="preset-grid">
            <div class="preset-option selected" data-model="model1">
              <strong>Model 1</strong><span>women_white</span>
            </div>
            <div class="preset-option" data-model="model2">
              <strong>Model 2</strong><span>women_gray</span>
            </div>
            <div class="preset-option" data-model="model3">
              <strong>Model 3</strong><span>women_black</span>
            </div>
          </div>
          <input type="hidden" id="preset_model" name="preset_model" value="model1">
        </div>

        <div id="panel-video-custom" class="input-panel">
          <label for="video">Upload video (mp4, etc.)</label>
          <input id="video" name="video" type="file" accept="video/*">
        </div>

        <label>Driving speech</label>
        <div class="mode-tabs" id="speech-mode-tabs">
          <button type="button" class="mode-tab active" data-mode="text">Text (TTS)</button>
          <button type="button" class="mode-tab" data-mode="audio">Audio file</button>
        </div>

        <div id="panel-text" class="input-panel active">
          <label for="text">Script / text</label>
          <textarea id="text" name="text" placeholder="Enter the text to speak..."></textarea>
          <label for="tts_voice">TTS voice (local Piper)</label>
          <select id="tts_voice" name="tts_voice">
            <option value="en_US-lessac-medium">English — Lessac (local)</option>
          </select>
        </div>

        <div id="panel-audio" class="input-panel">
          <label for="audio">Driving audio (wav, mp3, etc.)</label>
          <div class="audio-row">
            <input id="audio" name="audio" type="file" accept="audio/*">
            <button type="button" id="audio-upload-btn" class="btn-secondary" style="display:none" disabled>Upload</button>
          </div>
          <div id="audio-upload-status" class="hint">Select a file, click Upload, then Start streaming.</div>
        </div>

        <button id="submit-btn" type="submit">Start streaming</button>
        <div id="status">Connecting to server...</div>
        <div id="timing-info"></div>
        <div id="conn-hint" class="hint"></div>
        <div class="hint">Local offline TTS · Step timings in server log and status · MJPEG preview while generating</div>
      </form>
    </section>

    <section class="panel stream-wrap">
      <video id="player" controls playsinline muted style="display:none"></video>
      <img id="preview" alt="status preview" style="display:none">
      <div id="placeholder" class="placeholder">The stream will appear here after upload.</div>
    </section>
  </main>

  <script>
    const form = document.getElementById('upload-form');
    const statusEl = document.getElementById('status');
    const timingEl = document.getElementById('timing-info');
    const connHint = document.getElementById('conn-hint');
    const submitBtn = document.getElementById('submit-btn');
    const audioInput = document.getElementById('audio');
    const audioUploadBtn = document.getElementById('audio-upload-btn');
    const audioUploadStatus = document.getElementById('audio-upload-status');
    const previewImg = document.getElementById('preview');
    let player = document.getElementById('player');
    const placeholder = document.getElementById('placeholder');
    let pollTimer = null;
    let pollSlowTimer = null;
    let previewToken = 0;
    let avStarted = false;
    let activeJobId = null;
    let pendingNewJob = false;
    let terminalJobHandled = null;
    let msePlayer = null;
    let activeStreamUrl = null;
    let streamSession = 0;
    let jobStartMs = null;
    let clientDisplayMs = null;
    let inputMode = 'text';
    let videoMode = 'preset';
    let selectedPreset = 'model1';
    let audioUploaded = false;
    let audioWarmupReady = false;
    let audioToken = null;
    let audioUploadBusy = false;
    let serverBusy = false;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function updateTimingDisplay(data) {
      if (!timingEl) return;
      const parts = [];
      if (data && data.first_frame_ms != null) {
        parts.push('Inference→frame (UNet start): ' + data.first_frame_ms + ' ms');
      } else if (data && data.inference_elapsed_ms != null) {
        parts.push('Running inference... ' + data.inference_elapsed_ms + ' ms');
      }
      if (clientDisplayMs != null) {
        parts.push('Start→browser play: ' + clientDisplayMs + ' ms');
      }
      timingEl.textContent = parts.join('  |  ');
    }

    function resetTimingDisplay() {
      jobStartMs = null;
      clientDisplayMs = null;
      if (timingEl) timingEl.textContent = '';
    }

    function markClientFirstFrame() {
      if (jobStartMs != null && clientDisplayMs == null) {
        clientDisplayMs = Math.round(Date.now() - jobStartMs);
      }
    }

    function updateSubmitEnabled() {
      if (serverBusy || audioUploadBusy) {
        submitBtn.disabled = true;
        return;
      }
      if (inputMode === 'audio') {
        submitBtn.disabled = !audioUploaded || !audioWarmupReady;
      } else {
        submitBtn.disabled = false;
      }
    }

    function resetAudioUpload() {
      audioUploaded = false;
      audioWarmupReady = false;
      audioToken = null;
      audioUploadBusy = false;
      audioUploadStatus.textContent = audioInput.files.length
        ? 'Click Upload to prepare audio and warm up GPU.'
        : 'Select a file, click Upload, then Start streaming.';
      audioUploadStatus.classList.remove('audio-upload-ok');
      if (audioInput.files.length) {
        audioUploadBtn.style.display = 'inline-block';
        audioUploadBtn.disabled = false;
        audioUploadBtn.textContent = 'Upload';
      } else {
        audioUploadBtn.style.display = 'none';
        audioUploadBtn.disabled = true;
      }
      updateSubmitEnabled();
    }

    async function waitForAudioWarmup(token) {
      for (let attempt = 0; attempt < 600; attempt++) {
        const res = await fetch('/api/audio-warmup/' + encodeURIComponent(token), { cache: 'no-store' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Warmup status failed');
        if (data.state === 'ready') {
          audioWarmupReady = true;
          audioUploadStatus.textContent = 'Ready: ' + (data.filename || 'audio') + ' — click Start streaming';
          audioUploadStatus.classList.add('audio-upload-ok');
          updateSubmitEnabled();
          setStatus('Model and GPU ready. Click Start streaming to play.');
          return;
        }
        if (data.state === 'error') throw new Error(data.error || 'Warmup failed');
        audioUploadStatus.textContent = data.message || 'Warming up model and GPU...';
        await new Promise((resolve) => setTimeout(resolve, 400));
      }
      throw new Error('Warmup timed out — try uploading again.');
    }

    connHint.textContent = 'Page URL: ' + window.location.href;

    function bindPlayerEvents(el) {
      el.addEventListener('error', () => {
        if (el.src) setStatus('Video playback failed — click Play or try again.');
      });
    }
    bindPlayerEvents(player);

    function replaceVideoElement() {
      const wrap = player.parentNode;
      const newEl = document.createElement('video');
      newEl.id = 'player';
      newEl.controls = true;
      newEl.setAttribute('playsinline', '');
      newEl.muted = false;
      newEl.style.width = '100%';
      newEl.style.display = 'block';
      newEl.style.background = '#000';
      newEl.style.minHeight = '360px';
      newEl.style.objectFit = 'contain';
      wrap.replaceChild(newEl, player);
      player = newEl;
      bindPlayerEvents(player);
      return player;
    }

    function stopPreview() {
      previewToken += 1;
      previewImg.removeAttribute('src');
      previewImg.style.display = 'none';
    }

    function startPreview() {
      previewToken += 1;
      const token = previewToken;
      placeholder.style.display = 'none';
      player.style.display = 'none';
      previewImg.style.display = 'block';
      previewImg.src = '/stream?t=' + token;
    }

    function boxType(box) {
      return String.fromCharCode(box[4], box[5], box[6], box[7]);
    }

    function concatBoxes(boxes) {
      const total = boxes.reduce((sum, box) => sum + box.length, 0);
      const out = new Uint8Array(total);
      let offset = 0;
      for (const box of boxes) {
        out.set(box, offset);
        offset += box.length;
      }
      return out;
    }

    class FMP4BoxFramer {
      constructor(sourceBuffer, onError) {
        this.sb = sourceBuffer;
        this.pending = new Uint8Array(0);
        this.queue = [];
        this.appending = false;
        this.initDone = false;
        this.onError = onError;
      }

      push(data) {
        const merged = new Uint8Array(this.pending.length + data.length);
        merged.set(this.pending);
        merged.set(data, this.pending.length);
        this.pending = merged;
        this.collectBoxes();
        this.flush();
      }

      collectBoxes() {
        while (true) {
          const box = this.extractBox();
          if (!box) break;
          this.queue.push(box);
        }
      }

      extractBox() {
        if (this.pending.length < 8) return null;
        let size = new DataView(this.pending.buffer, this.pending.byteOffset, 4).getUint32(0);
        if (size < 8) return null;
        if (size === 1) {
          if (this.pending.length < 16) return null;
          const hi = new DataView(this.pending.buffer, this.pending.byteOffset + 8, 4).getUint32(0);
          const lo = new DataView(this.pending.buffer, this.pending.byteOffset + 12, 4).getUint32(0);
          size = hi * 4294967296 + lo;
        }
        if (this.pending.length < size) return null;
        const box = this.pending.slice(0, size);
        this.pending = this.pending.slice(size);
        if (boxType(box) === 'mfra') return this.extractBox();
        return box;
      }

      onUpdateEnd() {
        this.appending = false;
        this.collectBoxes();
        this.flush();
      }

      appendOne(buffer) {
        this.appending = true;
        try {
          this.sb.appendBuffer(buffer);
        } catch (err) {
          if (this.sb && this.sb.buffered && this.sb.buffered.length) {
            try {
              this.sb.timestampOffset = this.sb.buffered.end(this.sb.buffered.length - 1);
              this.sb.appendBuffer(buffer);
              return;
            } catch (_) {}
          }
          this.appending = false;
          if (this.onError) this.onError(err);
        }
      }

      flush() {
        if (this.appending || !this.queue.length || !this.sb || this.sb.updating) return;

        while (this.queue.length && boxType(this.queue[0]) === 'mfra') {
          this.queue.shift();
        }
        if (!this.queue.length) return;

        if (!this.initDone && this.queue.length >= 2 && boxType(this.queue[0]) === 'ftyp' && boxType(this.queue[1]) === 'moov') {
          this.appendOne(concatBoxes([this.queue.shift(), this.queue.shift()]));
          this.initDone = true;
          return;
        }

        if (this.queue.length >= 2 && boxType(this.queue[0]) === 'moof' && boxType(this.queue[1]) === 'mdat') {
          this.appendOne(concatBoxes([this.queue.shift(), this.queue.shift()]));
          return;
        }

        if (!this.initDone && boxType(this.queue[0]) === 'moov') {
          this.appendOne(this.queue.shift());
          this.initDone = true;
          return;
        }

        if (this.initDone && boxType(this.queue[0]) === 'moof') {
          if (this.queue.length >= 2 && boxType(this.queue[1]) === 'mdat') {
            this.appendOne(concatBoxes([this.queue.shift(), this.queue.shift()]));
          }
          return;
        }

        this.appendOne(this.queue.shift());
      }
    }

    class MSEStreamPlayer {
      constructor(videoEl, streamUrl, sessionId) {
        this.video = videoEl;
        this.streamUrl = streamUrl;
        this.sessionId = sessionId;
        this.mediaSource = null;
        this.sourceBuffer = null;
        this.queue = [];
        this.appending = false;
        this.objectUrl = null;
        this.aborted = false;
        this._abortController = null;
        this._sourceOpenHandler = null;
        this.framer = null;
        this._waitingForData = false;
        this._playStarted = false;
      }

      isStale() {
        return this.aborted || this.sessionId !== streamSession;
      }

      stop() {
        this.aborted = true;
        if (this._abortController) {
          try { this._abortController.abort(); } catch (_) {}
          this._abortController = null;
        }
        if (this.mediaSource && this._sourceOpenHandler) {
          try { this.mediaSource.removeEventListener('sourceopen', this._sourceOpenHandler); } catch (_) {}
        }
        try {
          if (this.sourceBuffer && this.mediaSource && this.mediaSource.readyState === 'open') {
            this.mediaSource.removeSourceBuffer(this.sourceBuffer);
          }
        } catch (_) {}
        try {
          if (this.mediaSource && this.mediaSource.readyState === 'open') {
            this.mediaSource.endOfStream();
          }
        } catch (_) {}
        if (this.objectUrl) {
          URL.revokeObjectURL(this.objectUrl);
          this.objectUrl = null;
        }
        this.sourceBuffer = null;
        this.mediaSource = null;
        this.queue = [];
        this.appending = false;
        this._playStarted = false;
        this._waitingForData = false;
      }

      start() {
        if (this.isStale()) return;
        if (!window.MediaSource) {
          setStatus('MediaSource API is not supported in this browser.');
          return;
        }
        this.mediaSource = new MediaSource();
        this.objectUrl = URL.createObjectURL(this.mediaSource);
        this._sourceOpenHandler = () => this.onSourceOpen();
        this.mediaSource.addEventListener('sourceopen', this._sourceOpenHandler);
        this.video.src = this.objectUrl;
      }

      onSourceOpen() {
        if (this.isStale()) return;
        const codecs = [
          'video/mp4; codecs="avc1.42C01F, mp4a.40.2"',
          'video/mp4; codecs="avc1.42E01E, mp4a.40.2"',
          'video/mp4; codecs="avc1.42001E, mp4a.40.2"',
          'video/mp4; codecs="avc1.4D401E, mp4a.40.2"',
        ];
        const mime = codecs.find((c) => MediaSource.isTypeSupported(c));
        if (!mime) {
          setStatus('Browser cannot play progressive MP4 (MSE unsupported).');
          return;
        }
        this.sourceBuffer = this.mediaSource.addSourceBuffer(mime);
        this.sourceBuffer.mode = 'segments';
        this.framer = new FMP4BoxFramer(this.sourceBuffer, (err) => {
          if (!this.isStale()) {
            avStarted = false;
            setStatus('Live stream decode issue — final video will play when ready.');
          }
        });
        this.sourceBuffer.addEventListener('updateend', () => {
          this.framer.onUpdateEnd();
          if (!this.sourceBuffer.buffered.length) return;
          const end = this.sourceBuffer.buffered.end(this.sourceBuffer.buffered.length - 1);
          const ahead = end - this.video.currentTime;
          if (this._waitingForData) {
            if (ahead >= 1.0) {
              this.video.play().then(() => {
                this._waitingForData = false;
              }).catch(() => {});
            }
            return;
          }
          if (!this._playStarted && ahead >= 0.45) {
            this.video.play().then(() => {
              this._playStarted = true;
            }).catch(() => {
              if (!this.isStale()) setStatus('Streaming — click Play on the video player.');
            });
          }
        });
        this.video.addEventListener('waiting', () => {
          if (!this.isStale()) this._waitingForData = true;
        });
        this.fetchStream();
      }

      async fetchStream() {
        if (this.isStale()) return;
        this._abortController = new AbortController();
        try {
          const resp = await fetch(this.streamUrl + '?t=' + Date.now(), {
            cache: 'no-store',
            signal: this._abortController.signal,
          });
          if (this.isStale()) return;
          if (!resp.ok) throw new Error('HTTP ' + resp.status);
          const reader = resp.body.getReader();
          while (!this.isStale()) {
            const { done, value } = await reader.read();
            if (done) break;
            if (value && value.length) {
              if (this.framer) this.framer.push(value);
            }
          }
          this.waitForDrain(() => {
            if (!this.isStale() && this.framer) {
              this.framer.collectBoxes();
              this.framer.flush();
            }
            if (!this.isStale() && this.mediaSource && this.mediaSource.readyState === 'open') {
              try { this.mediaSource.endOfStream(); } catch (_) {}
            }
          });
        } catch (err) {
          if (!this.isStale() && err.name !== 'AbortError') {
            setStatus('Stream fetch failed: ' + err.message);
          }
        }
      }

      waitForDrain(done) {
        if (this.isStale()) return;
        const busy = this.framer && (this.framer.appending || this.framer.queue.length || this.framer.pending.length);
        if (!busy && (!this.sourceBuffer || !this.sourceBuffer.updating)) {
          done();
          return;
        }
        setTimeout(() => this.waitForDrain(done), 50);
      }
    }

    function stopMSEPlayer() {
      if (msePlayer) {
        msePlayer.stop();
        msePlayer = null;
      }
      activeStreamUrl = null;
    }

    function startProgressiveMP4(data) {
      if (!data.stream_url || pendingNewJob) return;
      if (activeJobId && data.job_id && data.job_id !== activeJobId) return;
      if (msePlayer && activeStreamUrl === data.stream_url) return;

      stopMSEPlayer();
      streamSession += 1;
      const sessionId = streamSession;
      activeStreamUrl = data.stream_url;

      placeholder.style.display = 'none';
      previewImg.style.display = 'none';
      previewImg.removeAttribute('src');

      const videoEl = replaceVideoElement();
      videoEl.style.display = 'block';
      // Submit click counts as user gesture — allow audio during live fMP4 playback.
      videoEl.muted = false;

      msePlayer = new MSEStreamPlayer(videoEl, data.stream_url, sessionId);
      msePlayer.start();

      videoEl.addEventListener('playing', () => {
        if (sessionId !== streamSession) return;
        avStarted = true;
        markClientFirstFrame();
        updateTimingDisplay({});
        previewImg.style.display = 'none';
        videoEl.muted = false;
        videoEl.volume = 1.0;
      }, { once: true });
    }

    function hasPlayableBuffer(el) {
      try {
        return el && el.buffered && el.buffered.length > 0 && el.buffered.end(el.buffered.length - 1) > 0.3;
      } catch (_) {
        return false;
      }
    }

    async function playResultVideo(data, force) {
      if (!data.result_url) return;
      if (avStarted && !force && hasPlayableBuffer(player)) return;
      if (data.job_id && activeJobId && data.job_id !== activeJobId) return;

      stopMSEPlayer();
      streamSession += 1;

      placeholder.style.display = 'none';
      previewImg.style.display = 'none';
      previewImg.removeAttribute('src');

      const videoEl = replaceVideoElement();
      videoEl.style.display = 'block';
      videoEl.muted = false;

      for (let i = 0; i < 40; i++) {
        try {
          const head = await fetch(data.result_url, { method: 'HEAD', cache: 'no-store' });
          if (head.ok) break;
        } catch (_) {}
        await new Promise((r) => setTimeout(r, 250));
      }

      videoEl.src = data.result_url + '?t=' + Date.now();
      videoEl.load();
      videoEl.play().catch(() => {
        setStatus('Video ready — click Play on the video player.');
      });
      avStarted = true;
    }

    function resetStreamView() {
      pendingNewJob = true;
      avStarted = false;
      activeJobId = null;
      terminalJobHandled = null;
      resetTimingDisplay();
      stopMSEPlayer();
      stopPreview();
      streamSession += 1;
      placeholder.style.display = 'flex';
      try {
        player.pause();
        player.removeAttribute('src');
        player.load();
        player.style.display = 'none';
      } catch (_) {}
    }

    function beginFastPolling() {
      if (pollTimer) clearInterval(pollTimer);
      pollTimer = setInterval(pollStatus, 150);
      if (pollSlowTimer) {
        clearInterval(pollSlowTimer);
        pollSlowTimer = null;
      }
    }

    function beginSlowPolling() {
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      if (!pollSlowTimer) {
        pollSlowTimer = setInterval(pollStatus, 2000);
      }
    }

    async function pollStatus() {
      try {
        const res = await fetch('/api/status', { cache: 'no-store' });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        let msg = data.message || data.state;
        if (data.timing && data.timing.length) {
          const last = data.timing[data.timing.length - 1];
          msg += ` [${last.step}: ${last.total_ms}ms]`;
        }
        updateTimingDisplay(data);

        const busy = ['preparing', 'streaming', 'loading', 'uploading'].includes(data.state);
        serverBusy = busy || pendingNewJob;
        updateSubmitEnabled();

        if (pendingNewJob) {
          setStatus(msg);
          return;
        }

        if (activeJobId && data.job_id && data.job_id !== activeJobId) {
          return;
        }

        if (!activeJobId && data.job_id && ['uploading', 'preparing', 'streaming'].includes(data.state)) {
          activeJobId = data.job_id;
        }

        setStatus(msg);

        if (data.state === 'streaming' && data.job_id === activeJobId) {
          placeholder.style.display = 'none';
          previewImg.style.display = 'none';
          previewImg.removeAttribute('src');
          if (data.stream_url) startProgressiveMP4(data);
          let streamMsg = msg;
          if (data.frames_streamed) {
            streamMsg = 'A/V stream (video+audio) — ' + streamMsg + ` | ${data.frames_streamed} frames`;
          } else {
            streamMsg = 'A/V stream starting (video+audio)... — ' + streamMsg;
          }
          if (data.stream_bytes) {
            streamMsg += ` | mp4 ${Math.round(data.stream_bytes / 1024)}KB`;
          }
          setStatus(streamMsg);
        }

        if (['uploading', 'preparing'].includes(data.state) && data.job_id === activeJobId) {
          if (data.stream_url) startProgressiveMP4(data);
          if (previewImg.style.display === 'none' && player.style.display === 'none') startPreview();
          else previewImg.style.display = 'block';
        }

        if (data.state === 'done' && data.job_id === activeJobId) {
          beginSlowPolling();
          if (terminalJobHandled === data.job_id) {
            if (hasPlayableBuffer(player)) {
              setStatus((data.message || 'Stream complete.') + ' — progressive MP4 finished.');
            }
            return;
          }
          terminalJobHandled = data.job_id;
          if (hasPlayableBuffer(player)) {
            setStatus((data.message || 'Stream complete.') + ' — progressive MP4 finished.');
          } else if (data.result_url) {
            playResultVideo(data, true);
          } else if (data.stream_url) {
            startProgressiveMP4(data);
          }
        }

        if (['idle', 'done', 'error', 'cancelled'].includes(data.state)) {
          serverBusy = false;
          updateSubmitEnabled();
          if (['idle', 'error', 'cancelled'].includes(data.state)) {
            beginSlowPolling();
          }
        }
      } catch (err) {
        setStatus('Server connection failed: ' + err.message);
        serverBusy = false;
        pendingNewJob = false;
        updateSubmitEnabled();
      }
    }

    audioInput.addEventListener('change', () => {
      resetAudioUpload();
    });

    audioUploadBtn.addEventListener('click', async () => {
      const file = audioInput.files[0];
      if (!file) {
        setStatus('Please select an audio file first.');
        return;
      }
      audioUploadBusy = true;
      audioUploaded = false;
      audioWarmupReady = false;
      audioToken = null;
      audioUploadBtn.disabled = true;
      audioUploadBtn.textContent = 'Uploading...';
      audioUploadStatus.textContent = 'Uploading audio...';
      audioUploadStatus.classList.remove('audio-upload-ok');
      updateSubmitEnabled();

      const body = new FormData();
      body.append('audio', file);
      if (videoMode === 'preset') {
        body.append('preset_model', selectedPreset);
      }

      try {
        const res = await fetch('/api/upload-audio', { method: 'POST', body });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Audio upload failed');
        audioUploaded = true;
        audioToken = data.audio_token;
        audioUploadBtn.textContent = 'Uploaded';
        setStatus('Audio uploaded — warming up model and GPU...');
        await waitForAudioWarmup(audioToken);
      } catch (err) {
        audioUploaded = false;
        audioWarmupReady = false;
        audioToken = null;
        audioUploadStatus.textContent = 'Upload failed: ' + err.message;
        audioUploadBtn.textContent = 'Upload';
        audioUploadBtn.disabled = false;
        setStatus('Audio upload error: ' + err.message);
      } finally {
        audioUploadBusy = false;
        updateSubmitEnabled();
      }
    });

    document.querySelectorAll('#video-mode-tabs .mode-tab').forEach((btn) => {
      btn.addEventListener('click', () => {
        videoMode = btn.dataset.videoMode;
        document.querySelectorAll('#video-mode-tabs .mode-tab').forEach((b) => {
          b.classList.toggle('active', b.dataset.videoMode === videoMode);
        });
        document.getElementById('panel-video-preset').classList.toggle('active', videoMode === 'preset');
        document.getElementById('panel-video-custom').classList.toggle('active', videoMode === 'custom');
      });
    });

    document.querySelectorAll('.preset-option').forEach((el) => {
      el.addEventListener('click', () => {
        selectedPreset = el.dataset.model;
        document.querySelectorAll('.preset-option').forEach((o) => {
          o.classList.toggle('selected', o.dataset.model === selectedPreset);
        });
        document.getElementById('preset_model').value = selectedPreset;
      });
    });

    document.querySelectorAll('#speech-mode-tabs .mode-tab').forEach((btn) => {
      btn.addEventListener('click', () => {
        inputMode = btn.dataset.mode;
        document.querySelectorAll('#speech-mode-tabs .mode-tab').forEach((b) => b.classList.toggle('active', b.dataset.mode === inputMode));
        document.getElementById('panel-text').classList.toggle('active', inputMode === 'text');
        document.getElementById('panel-audio').classList.toggle('active', inputMode === 'audio');
        updateSubmitEnabled();
      });
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const video = document.getElementById('video').files[0];
      const text = document.getElementById('text').value.trim();
      if (videoMode === 'custom' && !video) {
        setStatus('Please upload a reference video.');
        return;
      }
      if (inputMode === 'text' && !text) {
        setStatus('Please enter text for TTS.');
        return;
      }
      if (inputMode === 'audio' && (!audioUploaded || !audioWarmupReady)) {
        setStatus(!audioUploaded
          ? 'Please upload an audio file first (click Upload).'
          : 'Preparing model and GPU — please wait...');
        return;
      }

      submitBtn.disabled = true;
      serverBusy = true;
      resetStreamView();
      jobStartMs = Date.now();
      clientDisplayMs = null;
      if (timingEl) timingEl.textContent = '';
      const busyMsg = inputMode === 'text'
        ? (videoMode === 'preset' ? 'Starting local TTS + progressive stream...' : 'Uploading video and starting stream...')
        : (videoMode === 'preset' ? 'Starting stream with preset model...' : 'Uploading files...');
      setStatus(busyMsg);
      startPreview();

      const body = new FormData();
      if (videoMode === 'preset') {
        body.append('preset_model', selectedPreset);
      } else {
        body.append('video', video);
      }
      if (inputMode === 'text') {
        body.append('text', text);
        body.append('tts_voice', document.getElementById('tts_voice').value);
      } else {
        body.append('audio_token', audioToken);
      }

      try {
        const res = await fetch('/api/start', { method: 'POST', body });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Upload failed');
        activeJobId = data.job_id;
        pendingNewJob = false;
        terminalJobHandled = null;
        setStatus(data.message || 'Processing started');
        beginFastPolling();
        pollStatus();
      } catch (err) {
        setStatus('Error: ' + err.message);
        serverBusy = false;
        pendingNewJob = false;
        updateSubmitEnabled();
      }
    });

    pollStatus();
    beginSlowPolling();
    updateSubmitEnabled();
  </script>
</body>
</html>
"""


def autotune_batch_sizes(args, device):
    """Raise batch sizes when VRAM headroom allows (avatar prep + streaming inference)."""
    if not torch.cuda.is_available():
        return
    idx = device.index if device.index is not None else 0
    free, total = torch.cuda.mem_get_info(idx)
    gb_free = free / (1024 ** 3)
    gb_total = total / (1024 ** 3)
    if gb_free >= 40 or gb_total >= 70:
        args.batch_size = max(args.batch_size, 64)
        args.stream_batch_size = max(args.stream_batch_size, 48)
    elif gb_free >= 20:
        args.batch_size = max(args.batch_size, 40)
        args.stream_batch_size = max(args.stream_batch_size, 32)
    elif gb_free >= 8:
        args.batch_size = max(args.batch_size, 24)
        args.stream_batch_size = max(args.stream_batch_size, 16)
    elif gb_free >= 5:
        args.batch_size = max(args.batch_size, 16)
        args.stream_batch_size = max(args.stream_batch_size, 12)
    args.stream_first_batch_size = max(1, min(args.stream_first_batch_size, 2))
    ramp = getattr(args, "stream_ramp_batch_size", 4)
    args.stream_ramp_batch_size = max(2, min(ramp, args.stream_batch_size))
    args.stream_ramp_batches = max(2, getattr(args, "stream_ramp_batches", 4))
    if getattr(args, "progressive_mode", "piped") == "piped":
        args.stream_batch_size = min(args.stream_batch_size, 24)
        args.stream_ramp_batch_size = min(args.stream_ramp_batch_size, 4)
        args.stream_ramp_batches = max(args.stream_ramp_batches, 4)
    print(
        f"VRAM {gb_free:.1f}GB free / {gb_total:.1f}GB total — "
        f"stream_batch={args.stream_batch_size}, first_batch={args.stream_first_batch_size}, "
        f"ramp={args.stream_ramp_batch_size}x{args.stream_ramp_batches}, "
        f"avatar_batch={args.batch_size}",
        flush=True,
    )


def configure_cuda(device):
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print(f"CUDA tuned: cudnn.benchmark=True, TF32 enabled on {device}")


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


def warmup_inference_models(device, models):
    """Run one GPU pass so the first user job avoids cold-start latency."""
    if not torch.cuda.is_available():
        return
    vae, unet, pe, timesteps, audio_processor, weight_dtype, whisper, fp = models
    try:
        with torch.no_grad():
            dummy_mel = torch.zeros(1, 80, 3000, device=device, dtype=weight_dtype)
            whisper.encoder(dummy_mel)
            dummy_audio = torch.zeros(4, 50, 384, device=device, dtype=weight_dtype)
            audio_feature_batch = pe(dummy_audio)
            if hasattr(vae, "encode_latents") and os.path.isfile("./results/v15/avatars"):
                pass
            latent_path = None
            for root, _, files in os.walk("./results/v15/avatars"):
                if "latents.pt" in files:
                    latent_path = os.path.join(root, "latents.pt")
                    break
            if latent_path:
                latents = torch.load(latent_path, map_location=device)
                sample = latents[0].to(device=device, dtype=unet.model.dtype)
                batch_latent = torch.cat([sample] * 4, dim=0)
                pred = unet.model(batch_latent, timesteps, encoder_hidden_states=audio_feature_batch).sample
                vae.decode_latents(pred[:1])
            torch.cuda.synchronize()
        print("GPU inference warmup complete")
    except Exception as exc:
        print(f"Warning: GPU warmup skipped: {exc}")


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


class ProgressiveMP4Buffer:
    """Thread-safe buffer for fragmented MP4 bytes; optional write-through to disk."""

    def __init__(self, disk_path: str | None = None):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._chunks = []
        self._closed = False
        self._disk_path = disk_path
        self._disk_fp = None
        if disk_path:
            os.makedirs(os.path.dirname(disk_path), exist_ok=True)
            self._disk_fp = open(disk_path, "wb")

    def append(self, data: bytes):
        if not data:
            return
        with self._cond:
            self._chunks.append(data)
            if self._disk_fp is not None:
                self._disk_fp.write(data)
                self._disk_fp.flush()
            self._cond.notify_all()

    def close(self):
        with self._cond:
            self._closed = True
            if self._disk_fp is not None:
                self._disk_fp.flush()
                self._disk_fp.close()
                self._disk_fp = None
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def byte_count(self) -> int:
        with self._lock:
            total = sum(len(c) for c in self._chunks)
        if total == 0 and self._disk_path and os.path.isfile(self._disk_path):
            return os.path.getsize(self._disk_path)
        return total

    def iter_chunks(self):
        idx = 0
        while True:
            with self._cond:
                while idx >= len(self._chunks) and not self._closed:
                    self._cond.wait(timeout=0.05)
                if idx >= len(self._chunks):
                    if self._closed:
                        break
                    continue
                chunk = self._chunks[idx]
                idx += 1
            yield chunk

    def get_all_bytes(self):
        with self._lock:
            if self._chunks:
                return b"".join(self._chunks)
        if self._disk_path and os.path.isfile(self._disk_path):
            with open(self._disk_path, "rb") as f:
                return f.read()
        return b""


def progressive_mp4_path(job_id: str) -> str:
    return os.path.join(UPLOAD_ROOT, job_id, "progressive.mp4")


def _start_stderr_drain(proc: subprocess.Popen):
    def _run():
        try:
            if proc.stderr:
                proc.stderr.read()
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


USE_FFMPEG_GPU = False


def probe_ffmpeg_nvenc() -> bool:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    try:
        encoders = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if encoders.returncode != 0 or "h264_nvenc" not in encoders.stdout:
            return False
        test = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=256x256:d=0.04",
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p1",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return test.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def ffmpeg_h264_encoder_args(fps: int, use_gpu: bool | None = None, low_latency: bool = False) -> list:
    """Return FFmpeg video encoder arguments (NVENC on GPU when available)."""
    gpu = USE_FFMPEG_GPU if use_gpu is None else use_gpu
    gop = "1" if low_latency else str(max(1, fps // 2))
    if gpu:
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-tune",
            "ll",
            "-g",
            gop,
            "-keyint_min",
            gop,
            "-pix_fmt",
            "yuv420p",
        ]
    cpu_args = [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-g",
        gop,
        "-keyint_min",
        gop,
        "-pix_fmt",
        "yuv420p",
    ]
    if low_latency:
        cpu_args.extend(["-x264-params", "keyint=1:min-keyint=1:scenecut=0"])
    return cpu_args


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


class FFmpegProgressiveStreamer:
    """Mux BGR frames + audio into fragmented MP4 (pipe) for progressive HTTP streaming."""

    def __init__(
        self,
        fps: int,
        audio_path: str,
        buffer: ProgressiveMP4Buffer,
        output_path: str,
        use_gpu: bool | None = None,
        ts_offset: float = 0.0,
        frag_discont: bool = False,
    ):
        self.fps = fps
        self.audio_path = audio_path
        self.buffer = buffer
        self.output_path = output_path
        self.use_gpu = use_gpu
        self.ts_offset = ts_offset
        self.frag_discont = frag_discont
        self._proc = None
        self._reader = None
        self._started = False
        self._lock = threading.Lock()
        self._error = None

    @staticmethod
    def _even_dim(n: int) -> int:
        return n if n % 2 == 0 else n - 1

    def _start(self, width: int, height: int):
        width = self._even_dim(width)
        height = self._even_dim(height)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-i",
            self.audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            *ffmpeg_h264_encoder_args(self.fps, use_gpu=self.use_gpu, low_latency=True),
            "-profile:v",
            "baseline",
            "-level",
            "3.1",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-shortest",
        ]
        if self.ts_offset > 0:
            cmd.extend(["-output_ts_offset", f"{self.ts_offset:.6f}"])
        if self.frag_discont:
            cmd.extend(["-force_key_frames", "expr:eq(n,0)"])
        movflags = "frag_keyframe+empty_moov+default_base_moof"
        if self.frag_discont:
            movflags += "+frag_discont"
        cmd.extend(
            [
            "-flush_packets",
            "1",
            "-f",
            "mp4",
            "-movflags",
            movflags,
            "pipe:1",
            ]
        )
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _start_stderr_drain(self._proc)
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._started = True
        enc = "h264_nvenc (GPU)" if (USE_FFMPEG_GPU if self.use_gpu is None else self.use_gpu) else "libx264 (CPU)"
        if self.ts_offset > 0:
            print(f"Progressive A/V encoder started ({width}x{height} @ {self.fps} fps, {enc}, ts={self.ts_offset:.3f}s)")
        else:
            print(f"Progressive A/V encoder started ({width}x{height} @ {self.fps} fps, {enc})")

    def _read_stdout(self):
        try:
            while True:
                chunk = self._proc.stdout.read(16384)
                if not chunk:
                    break
                self.buffer.append(chunk)
        except Exception as exc:
            self._error = exc
        finally:
            self.buffer.close()

    def write_frame(self, frame_bgr: np.ndarray):
        with self._lock:
            if self._error:
                return
            h, w = frame_bgr.shape[:2]
            w, h = self._even_dim(w), self._even_dim(h)
            if frame_bgr.shape[0] != h or frame_bgr.shape[1] != w:
                frame_bgr = frame_bgr[:h, :w]
            if not self._started:
                self._start(w, h)
            try:
                self._proc.stdin.write(frame_bgr.tobytes())
            except (BrokenPipeError, OSError) as exc:
                self._error = exc

    def finish(self):
        with self._lock:
            if self._proc is None:
                self.buffer.close()
                return
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except OSError:
                pass
            if self._reader is not None:
                self._reader.join(timeout=60)
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            if self._proc.stderr:
                err = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
                if err and self._proc.returncode not in (0, None):
                    print(f"FFmpeg progressive MP4 warning: {err[:500]}")
            if not self.buffer._closed:
                self.buffer.close()
            data = self.buffer.get_all_bytes()
            if data:
                with open(self.output_path, "wb") as f:
                    f.write(data)
                print(f"Progressive MP4 saved: {self.output_path} ({len(data)} bytes)")


class VideoStdinWriter:
    """Async queue writer so large HD frames never block the inference thread."""

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc
        self._queue = queue.Queue(maxsize=4096)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, frame_bytes: bytes):
        if frame_bytes:
            self._queue.put(frame_bytes)

    def finish(self):
        self._queue.put(None)
        self._thread.join(timeout=120)

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                if self._proc.stdin:
                    self._proc.stdin.write(item)
            except (BrokenPipeError, OSError):
                break
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except OSError:
            pass


class AudioPipeWriter:
    """Write PCM bytes into an os.pipe() fd consumed by FFmpeg."""

    def __init__(self, write_fd: int):
        self._write_fd = write_fd
        self._queue = queue.Queue()
        self._thread = None
        self._done = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while True:
                try:
                    chunk = self._queue.get(timeout=0.2)
                except queue.Empty:
                    if self._done.is_set() and self._queue.empty():
                        break
                    continue
                if chunk is None:
                    break
                os.write(self._write_fd, chunk)
        except OSError:
            pass
        finally:
            try:
                os.close(self._write_fd)
            except OSError:
                pass

    def write(self, pcm_bytes: bytes):
        if pcm_bytes:
            self._queue.put(pcm_bytes)

    def finish(self):
        self._done.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=60)


class PipedFFmpegMuxer:
    """Mux live BGR frames + streaming PCM audio into fragmented MP4 (single session)."""

    def __init__(
        self,
        fps: int,
        sample_rate: int,
        buffer: ProgressiveMP4Buffer,
        output_path: str,
        use_gpu: bool = False,
        frag_duration_us: int = 40000,
    ):
        self.fps = fps
        self.sample_rate = sample_rate
        self.buffer = buffer
        self.output_path = output_path
        self.use_gpu = use_gpu
        self.frag_duration_us = frag_duration_us
        self._audio_writer = None
        self._video_writer = None
        self._proc = None
        self._reader = None
        self._started = False
        self._lock = threading.Lock()
        self._error = None

    @staticmethod
    def _even_dim(n: int) -> int:
        return n if n % 2 == 0 else n - 1

    def prestart(self, width: int, height: int):
        with self._lock:
            if self._error or self._started:
                return
            self._start(width, height)

    def _start(self, width: int, height: int):
        width = self._even_dim(width)
        height = self._even_dim(height)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        audio_r, audio_w = os.pipe()
        self._audio_writer = AudioPipeWriter(audio_w)
        self._audio_writer.start()

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-thread_queue_size",
            "512",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-thread_queue_size",
            "512",
            "-f",
            "s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-i",
            f"pipe:{audio_r}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            *ffmpeg_h264_encoder_args(self.fps, use_gpu=self.use_gpu, low_latency=True),
            "-profile:v",
            "baseline",
            "-level",
            "3.1",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-max_muxing_queue_size",
            "512",
            "-flush_packets",
            "1",
            "-f",
            "mp4",
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof",
            "-frag_duration",
            str(self.frag_duration_us),
            "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(audio_r,),
        )
        os.close(audio_r)
        _start_stderr_drain(self._proc)
        self._video_writer = VideoStdinWriter(self._proc)
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._started = True
        enc = "h264_nvenc (GPU)" if self.use_gpu else "libx264 ultrafast (CPU, live)"
        print(f"Piped progressive A/V started ({width}x{height} @ {self.fps} fps, {enc}, audio={self.sample_rate} Hz)")

    def _read_stdout(self):
        try:
            while True:
                chunk = self._proc.stdout.read(4096)
                if not chunk:
                    break
                self.buffer.append(chunk)
        except Exception as exc:
            self._error = exc

    def write_frame(self, frame_bgr: np.ndarray):
        with self._lock:
            if self._error:
                return
            h, w = frame_bgr.shape[:2]
            w, h = self._even_dim(w), self._even_dim(h)
            if frame_bgr.shape[0] != h or frame_bgr.shape[1] != w:
                frame_bgr = frame_bgr[:h, :w]
            if not self._started:
                self._start(w, h)
            if self._video_writer is not None:
                self._video_writer.write(frame_bgr.tobytes())

    def write_audio_pcm(self, pcm_bytes: bytes):
        self._audio_writer.write(pcm_bytes)

    def finish(self):
        with self._lock:
            if self._proc is None:
                self.buffer.close()
                return
            print("[mux] Closing piped FFmpeg (video stdin + audio pipe)...", flush=True)
            if self._video_writer is not None:
                self._video_writer.finish()
                self._video_writer = None
            elif self._proc.stdin:
                try:
                    self._proc.stdin.close()
                except OSError:
                    pass
            self._audio_writer.finish()
            if self._reader is not None:
                self._reader.join(timeout=30)
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print("[mux] FFmpeg timeout — sending SIGKILL", flush=True)
                self._proc.kill()
                try:
                    self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            if self._proc.stderr:
                err = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
                if err and self._proc.returncode not in (0, None, -9):
                    print(f"[mux] FFmpeg stderr: {err[:500]}", flush=True)
            if not self.buffer._closed:
                self.buffer.close()
            data = self.buffer.get_all_bytes()
            if data:
                with open(self.output_path, "wb") as f:
                    f.write(data)
                print(f"Progressive MP4 saved: {self.output_path} ({len(data)} bytes)", flush=True)
            else:
                print("[mux] WARNING: no MP4 bytes captured from piped FFmpeg", flush=True)
            self._proc = None


class MuxWorker:
    """Pass A/V pairs to FFmpeg immediately (build browser buffer ahead of playback)."""

    _SENTINEL = ("__stop__", None)

    def __init__(self, service: "StreamWebService"):
        self.service = service
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_av_pair(self, frame_bgr: np.ndarray, pcm_bytes: bytes):
        self._queue.put(("pair", frame_bgr.copy(), pcm_bytes))

    def finish(self):
        self._queue.put(self._SENTINEL)
        self._thread.join(timeout=900)

    def _run(self):
        while True:
            item = self._queue.get()
            if item == self._SENTINEL:
                break
            try:
                if item[0] == "pair":
                    _, frame, pcm = item
                    self.service._write_mux_frame(frame)
                    self.service._write_mux_audio(pcm)
            except Exception as exc:
                print(f"[mux-worker] pair error: {exc}", flush=True)
        if self.service._av_encoder is not None:
            try:
                self.service._av_encoder.finish()
            except Exception as exc:
                print(f"[mux-worker] finish error: {exc}", flush=True)
            self.service._av_encoder = None


class StreamWebService:
    def __init__(self, args):
        self.args = args
        self._lock = threading.Lock()
        self._latest_frame = None
        self._frame_event = threading.Event()
        self._status = {"state": "loading", "message": "Loading models...", "cached": False, "job_id": None, "stream_url": None, "result_url": None, "audio_url": None}
        self._cancel_event = threading.Event()
        self._worker = None
        self._running = False
        self._models_ready = threading.Event()
        self._job_audio_paths = {}
        self._av_encoder = None
        self._av_audio_path = None
        self._av_output_path = None
        self._mp4_buffer = None
        self._job_mp4_buffers = {}
        self._current_job_id = None
        self._mux_frames = False
        self._piped_mux = False
        self._upload_dir = None
        self._fifo_path = None
        self._tts_sample_rate = 22050
        self._audio_byte_queue = bytearray()
        self._audio_bytes_per_frame = None
        self._mux_worker = None
        self._frame_archive = []
        self._job_timers = {}
        self._job_inference_t0 = {}
        self._job_first_frame_ms = {}
        self._avatar_cache = {}  # video_hash -> Avatar (kept while reference video unchanged)
        self._preset_preload = {}  # preset_id -> pending|loading|ready|error
        self._preload_lock = threading.Lock()
        self._audio_staging = {}
        self._staging_lock = threading.Lock()
        self._live_segment_mux = False
        self._live_ts_offset = 0.0
        self._chunk_pcm = None
        self._chunk_stream_archive_start = 0
        self._chunk_stream_emitted = 0
        self._chunk_bytes_per_frame = None
        self._frames_pushed = 0
        self._segment_executor = None
        self._moof_seq = 1

    @staticmethod
    def _rewrite_moof_sequences(data: bytes, start_seq: int) -> tuple[bytes, int]:
        """Assign monotonically increasing mfhd sequence numbers for MSE append."""
        out = bytearray()
        pos = 0
        seq = start_seq
        while pos + 8 <= len(data):
            size = int.from_bytes(data[pos : pos + 4], "big")
            if size < 8:
                break
            box_type = data[pos + 4 : pos + 8]
            if size == 1:
                if pos + 16 > len(data):
                    break
                size = int.from_bytes(data[pos + 8 : pos + 16], "big")
            if pos + size > len(data):
                break
            chunk = bytearray(data[pos : pos + size])
            if box_type == b"moof":
                inner = 8
                while inner + 8 <= len(chunk):
                    child_size = int.from_bytes(chunk[inner : inner + 4], "big")
                    if child_size < 8 or inner + child_size > len(chunk):
                        break
                    if chunk[inner + 4 : inner + 8] == b"mfhd" and inner + 16 <= len(chunk):
                        chunk[inner + 12 : inner + 16] = seq.to_bytes(4, "big")
                        seq += 1
                        break
                    inner += child_size
            out.extend(chunk)
            pos += size
        return bytes(out), seq

    def _preload_preset_model(self, preset_id, quiet=True):
        """Prepare preset face analysis and keep Avatar in memory."""
        if preset_id not in PRESET_MODELS:
            return False

        video_path = os.path.abspath(PRESET_MODELS[preset_id])
        if not os.path.isfile(video_path):
            print(f"Preload skip ({preset_id}): file not found: {video_path}")
            with self._preload_lock:
                self._preset_preload[preset_id] = "error"
            return False

        video_hash = compute_file_hash(video_path)
        with self._preload_lock:
            if video_hash in self._avatar_cache:
                self._preset_preload[preset_id] = "ready"
                print(f"Preload skip ({preset_id}): already in memory")
                return True
            self._preset_preload[preset_id] = "loading"

        avatar_id, need_preparation, video_hash = self._resolve_avatar(video_path)
        try:
            if not quiet:
                if need_preparation:
                    self.set_status("preparing", f"Preparing {preset_id} face analysis...", cached=False)
                    self.push_status_frame(f"Preparing {preset_id}...")
                else:
                    self.set_status("preparing", f"Loading {preset_id} face model...", cached=True)
                    self.push_status_frame(f"Loading {preset_id}...")

            avatar = Avatar(
                avatar_id=avatar_id,
                video_path=video_path,
                bbox_shift=0,
                batch_size=self.args.batch_size,
                preparation=need_preparation,
                non_interactive=True,
            )
            if need_preparation:
                self._register_cache(video_hash, avatar_id, video_path)

            with self._preload_lock:
                self._avatar_cache[video_hash] = avatar
                self._preset_preload[preset_id] = "ready"
            print(f"Preloaded {preset_id} into memory ({avatar_id}, prep={need_preparation})")
            return True
        except Exception as exc:
            print(f"Preload failed ({preset_id}): {exc}")
            with self._preload_lock:
                self._preset_preload[preset_id] = "error"
            return False

    def preload_default_presets(self):
        for preset_id in self.args.preload_presets:
            self._preload_preset_model(preset_id, quiet=True)

    def is_preset_preloaded(self, preset_id):
        with self._preload_lock:
            return self._preset_preload.get(preset_id) == "ready"

    def _get_or_create_avatar(self, video_path, avatar_id, need_preparation, video_hash, quiet=False):
        """Return cached in-memory avatar when the reference video is unchanged."""
        cached = self._avatar_cache.get(video_hash)
        if cached is not None and not need_preparation:
            cached.idx = 0
            print(f"Reusing in-memory face model: {avatar_id}")
            return cached, True

        if not need_preparation and not quiet:
            self.set_status("preparing", "Loading face model from disk...", cached=True)
            self.push_status_frame("Loading face model...")
        elif need_preparation and not quiet:
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
        self._avatar_cache[video_hash] = avatar
        return avatar, False

    def set_status(self, state, message, cached=False, job_id=None, stream_url=None, result_url=None, audio_url=None):
        with self._lock:
            if job_id is not None and self._current_job_id is not None and job_id != self._current_job_id:
                return
            status = {"state": state, "message": message, "cached": cached, "stream_url": None, "result_url": None, "audio_url": None}
            if job_id is not None:
                status["job_id"] = job_id
            elif "job_id" in self._status:
                status["job_id"] = self._status.get("job_id")
            if stream_url is not None:
                status["stream_url"] = stream_url
            elif "stream_url" in self._status:
                status["stream_url"] = self._status.get("stream_url")
            if result_url is not None:
                status["result_url"] = result_url
            elif "result_url" in self._status:
                status["result_url"] = self._status.get("result_url")
            if audio_url is not None:
                status["audio_url"] = audio_url
            elif "audio_url" in self._status:
                status["audio_url"] = self._status.get("audio_url")
            self._status = status
        print(f"[{state}] {message}")

    def mark_inference_start(self, job_id: str):
        """Mark UNet/VAE lip-sync loop start (Whisper/TTS prep excluded)."""
        with self._lock:
            if job_id in self._job_inference_t0:
                return
            self._job_inference_t0[job_id] = time.time()
            self._job_first_frame_ms.pop(job_id, None)

    def _record_first_lipsync_frame(self, job_id: str):
        with self._lock:
            t0 = self._job_inference_t0.get(job_id)
            if t0 is None or job_id in self._job_first_frame_ms:
                return None
            ms = round((time.time() - t0) * 1000, 1)
            self._job_first_frame_ms[job_id] = ms
        print(f"[timing:{job_id[:8]}] first_lipsync_frame: {ms}ms after UNet inference start", flush=True)
        with self._lock:
            if self._status.get("job_id") == job_id:
                note = f"First frame in {ms} ms"
                prev = self._status.get("message") or ""
                if note not in prev:
                    self._status["message"] = f"{note} — {prev}"
        return ms

    def _reset_av_stream(self, job_id, audio_path=None, piped=False, upload_dir=None):
        if self._mux_worker is not None:
            try:
                self._mux_worker.finish()
            except Exception:
                pass
            self._mux_worker = None
        if self._av_encoder is not None:
            self._av_encoder.finish()
        self._av_encoder = None
        if self._mp4_buffer is not None:
            self._mp4_buffer.close()
        self._mp4_buffer = None
        self._current_job_id = job_id
        self._av_audio_path = audio_path
        self._upload_dir = upload_dir or os.path.join(UPLOAD_ROOT, job_id)
        self._piped_mux = piped
        self._segment_mux = False
        self._mp4_init_sent = False
        self._moof_seq = 1
        self._tts_sample_rate = 22050
        self._audio_byte_queue = bytearray()
        self._audio_bytes_per_frame = None
        self._chunk_pcm_buf = bytearray()
        self._use_chunk_pcm = False
        self._frame_archive = []
        self._av_output_path = os.path.join(UPLOAD_ROOT, job_id, "output.mp4")
        self._progressive_path = progressive_mp4_path(job_id)
        if os.path.isfile(self._progressive_path):
            os.remove(self._progressive_path)
        self._mp4_buffer = ProgressiveMP4Buffer(disk_path=self._progressive_path)
        self._job_mp4_buffers[job_id] = self._mp4_buffer
        if os.path.isfile(self._av_output_path):
            os.remove(self._av_output_path)
        self._mux_frames = False

    def _stop_running_job(self):
        self._cancel_event.set()
        self._mux_frames = False
        if self._mux_worker is not None:
            try:
                self._mux_worker.finish()
            except Exception:
                pass
            self._mux_worker = None
        if self._av_encoder is not None:
            try:
                self._av_encoder.finish()
            except Exception:
                pass
            self._av_encoder = None
        self._shutdown_segment_executor()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=8)
        self._worker = None
        with self._lock:
            self._latest_frame = None
        self._frame_event.clear()

    def _begin_av_mux(self):
        self._mux_frames = True
        self._frame_archive = []
        self._live_ts_offset = 0.0
        self._mp4_init_sent = False
        self._moof_seq = 1

    def _write_pcm_wav_slice(self, wav_path: str, frame_offset: int, num_frames: int):
        import soundfile as sf

        bps = self._chunk_bytes_per_frame
        start = frame_offset * bps
        end = start + num_frames * bps
        pcm = np.frombuffer(self._chunk_pcm[start:end], dtype=np.int16)
        sf.write(wav_path, pcm, self._tts_sample_rate, subtype="PCM_16")

    def _prepare_chunk_stream(self, raw_wav: str, sample_rate: int, archive_start: int):
        self._chunk_pcm, self._tts_sample_rate = self._load_wav_pcm(raw_wav)
        self._chunk_stream_archive_start = archive_start
        self._chunk_stream_emitted = archive_start
        self._chunk_bytes_per_frame = max(2, int(round(sample_rate / self.args.fps) * 2))

    def _get_segment_executor(self):
        if self._segment_executor is None:
            self._segment_executor = ThreadPoolExecutor(max_workers=1)
        return self._segment_executor

    def _shutdown_segment_executor(self):
        if self._segment_executor is not None:
            self._segment_executor.shutdown(wait=True)
            self._segment_executor = None

    def _emit_live_segment(self, num_frames: int):
        if num_frames <= 0 or not self._chunk_pcm:
            return
        start_idx = self._chunk_stream_emitted
        end_idx = start_idx + num_frames
        frames = [self._frame_archive[i].copy() for i in range(start_idx, end_idx)]
        if not frames:
            return
        frame_offset = start_idx - self._chunk_stream_archive_start
        partial_wav = os.path.join(self._upload_dir, f"partial_{start_idx:05d}.wav")
        self._write_pcm_wav_slice(partial_wav, frame_offset, num_frames)
        self._chunk_stream_emitted = end_idx
        self._get_segment_executor().submit(self._append_chunk_segment, frames, partial_wav)
        print(f"[mux] Queued live segment ({num_frames} frames, idx {start_idx})", flush=True)

    def _maybe_emit_live_segment(self):
        if not self._live_segment_mux or not self._chunk_pcm:
            return
        pending = len(self._frame_archive) - self._chunk_stream_emitted
        if pending < self.args.stream_emit_frames:
            return
        self._emit_live_segment(self.args.stream_emit_frames)

    def _flush_chunk_stream(self):
        if not self._live_segment_mux:
            return
        remaining = len(self._frame_archive) - self._chunk_stream_emitted
        if remaining > 0:
            self._emit_live_segment(remaining)
        self._chunk_pcm = None

    @staticmethod
    def _strip_fmp4_init(mp4_bytes: bytes) -> bytes:
        moof = mp4_bytes.find(b"moof")
        if moof >= 4:
            return mp4_bytes[moof - 4 :]
        return mp4_bytes

    @staticmethod
    def _strip_mfra_boxes(mp4_bytes: bytes) -> bytes:
        """Remove mfra boxes — browsers reject them in MSE appendBuffer."""
        out = bytearray()
        pos = 0
        while pos + 8 <= len(mp4_bytes):
            size = int.from_bytes(mp4_bytes[pos : pos + 4], "big")
            if size < 8:
                break
            box_type = mp4_bytes[pos + 4 : pos + 8]
            if size == 1:
                if pos + 16 > len(mp4_bytes):
                    break
                size = int.from_bytes(mp4_bytes[pos + 8 : pos + 16], "big")
            if pos + size > len(mp4_bytes):
                break
            if box_type != b"mfra":
                out.extend(mp4_bytes[pos : pos + size])
            pos += size
        return bytes(out)

    def _append_chunk_segment(self, frames, wav_path: str):
        """Mux one text-chunk into fMP4 and append to the live progressive buffer."""
        if not frames or not wav_path or not os.path.isfile(wav_path):
            return
        chunk_buf = ProgressiveMP4Buffer()
        part_path = self._av_output_path + ".part.mp4"
        t0 = time.time()
        is_continuation = self._mp4_init_sent
        encoder = FFmpegProgressiveStreamer(
            self.args.fps,
            wav_path,
            chunk_buf,
            part_path,
            use_gpu=USE_FFMPEG_GPU,
            ts_offset=self._live_ts_offset,
            frag_discont=is_continuation,
        )
        for frame in frames:
            encoder.write_frame(frame)
        encoder.finish()
        data = chunk_buf.get_all_bytes()
        if not data:
            print("[mux] Chunk segment produced no MP4 bytes", flush=True)
            return
        data, self._moof_seq = self._rewrite_moof_sequences(data, self._moof_seq)
        data = self._strip_mfra_boxes(data)
        if not self._mp4_init_sent:
            self._mp4_buffer.append(data)
            self._mp4_init_sent = True
        else:
            moof = data.find(b"moof")
            if moof >= 4:
                self._mp4_buffer.append(self._strip_mfra_boxes(data[moof - 4 :]))
            else:
                self._mp4_buffer.append(data)
        self._live_ts_offset += len(frames) / self.args.fps
        print(
            f"[mux] Chunk segment appended ({len(frames)} frames, {len(data)} bytes, {time.time() - t0:.2f}s, ts={self._live_ts_offset:.3f}s, moof_seq={self._moof_seq - 1})",
            flush=True,
        )

    def _mux_archived_frames(self, audio_path: str, update_buffer: bool = True):
        if not self._frame_archive or not audio_path or not os.path.isfile(audio_path):
            print("[mux] No archived frames or audio — skip final mux", flush=True)
            return
        print(f"[mux] Muxing {len(self._frame_archive)} frames with {audio_path}", flush=True)
        t0 = time.time()
        buffer = self._mp4_buffer if update_buffer else ProgressiveMP4Buffer()
        encoder = FFmpegProgressiveStreamer(
            self.args.fps,
            audio_path,
            buffer,
            self._av_output_path,
        )
        for frame in self._frame_archive:
            encoder.write_frame(frame)
        encoder.finish()
        self._av_encoder = None
        print(f"[mux] Final mux done in {time.time() - t0:.2f}s", flush=True)

    def _finish_av_stream(self):
        self._mux_frames = False
        self._shutdown_segment_executor()
        live_bytes = 0
        if self._mux_worker is not None:
            self._mux_worker.finish()
            self._mux_worker = None
            if self._mp4_buffer is not None:
                live_bytes = len(self._mp4_buffer.get_all_bytes())
        elif self._av_encoder is not None:
            self._av_encoder.finish()
            self._av_encoder = None
            if self._mp4_buffer is not None:
                live_bytes = len(self._mp4_buffer.get_all_bytes())

        if self._frame_archive and self._av_output_path:
            audio_path = None
            if self._current_job_id:
                audio_path = self._job_audio_paths.get(self._current_job_id)
            if audio_path and os.path.isfile(audio_path):
                if self._mp4_init_sent:
                    print("[mux] Re-muxing archived frames for final playable MP4", flush=True)
                    self._mux_archived_frames(audio_path, update_buffer=False)
                elif live_bytes == 0:
                    print("[mux] Live progressive mux empty — falling back to archived frames", flush=True)
                    self._mux_archived_frames(audio_path)
        if self._mp4_buffer is not None:
            live_bytes = len(self._mp4_buffer.get_all_bytes())
            if live_bytes > 0:
                print(f"[mux] Live A/V progressive stream delivered {live_bytes} bytes", flush=True)
            self._mp4_buffer.close()
        self._frame_archive = []

    def get_status(self):
        with self._lock:
            status = dict(self._status)
        with self._preload_lock:
            status["preset_preload"] = dict(self._preset_preload)
        job_id = status.get("job_id")
        if job_id and job_id in self._job_timers:
            status["timing"] = list(self._job_timers[job_id].steps)
        if job_id and job_id in self._job_mp4_buffers:
            status["stream_bytes"] = self._job_mp4_buffers[job_id].byte_count()
        elif job_id and os.path.isfile(progressive_mp4_path(job_id)):
            status["stream_bytes"] = os.path.getsize(progressive_mp4_path(job_id))
        status["frames_streamed"] = len(self._frame_archive)
        if job_id:
            if job_id in self._job_first_frame_ms:
                status["first_frame_ms"] = self._job_first_frame_ms[job_id]
            elif job_id in self._job_inference_t0:
                status["inference_elapsed_ms"] = round(
                    (time.time() - self._job_inference_t0[job_id]) * 1000, 1
                )
        return status

    def _ensure_piped_encoder(self, width: int, height: int):
        if not (self._mux_frames and self._mp4_buffer is not None and self._piped_mux):
            return
        if self._av_encoder is None:
            self._av_encoder = PipedFFmpegMuxer(
                self.args.fps,
                self._tts_sample_rate,
                self._mp4_buffer,
                self._av_output_path,
                use_gpu=USE_FFMPEG_GPU,
                frag_duration_us=self.args.fmp4_frag_us,
            )
            self._av_encoder.prestart(width, height)

    def _prepare_prewarmed_stream(self, prewarm):
        if not prewarm or not self._piped_mux:
            return
        if prewarm.get("pcm_bytes"):
            self._enqueue_chunk_audio(prewarm["pcm_bytes"], prewarm["pcm_sr"])
        if not prewarm.get("avatar_ready"):
            return
        avatar = self._avatar_cache.get(prewarm.get("video_hash"))
        if not avatar or not avatar.frame_list_cycle:
            return
        frame = avatar.frame_list_cycle[0]
        h, w = frame.shape[:2]
        self._ensure_piped_encoder(w, h)

    def _write_mux_frame(self, frame_bgr: np.ndarray):
        if not (self._mux_frames and self._av_output_path and self._mp4_buffer is not None):
            return
        if self._av_encoder is None:
            if self._piped_mux:
                self._av_encoder = PipedFFmpegMuxer(
                    self.args.fps,
                    self._tts_sample_rate,
                    self._mp4_buffer,
                    self._av_output_path,
                    use_gpu=USE_FFMPEG_GPU,
                    frag_duration_us=self.args.fmp4_frag_us,
                )
            elif self._av_audio_path:
                self._av_encoder = FFmpegProgressiveStreamer(
                    self.args.fps,
                    self._av_audio_path,
                    self._mp4_buffer,
                    self._av_output_path,
                )
        if self._av_encoder is not None:
            self._av_encoder.write_frame(frame_bgr)

    def _write_mux_audio(self, pcm_bytes: bytes):
        if self._av_encoder is not None and pcm_bytes and hasattr(self._av_encoder, "write_audio_pcm"):
            self._av_encoder.write_audio_pcm(pcm_bytes)

    def _load_wav_pcm(self, wav_path: str):
        import soundfile as sf

        data, sample_rate = sf.read(wav_path, dtype="int16")
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.int16)
        return data.tobytes(), int(sample_rate)

    def _enqueue_chunk_audio(self, pcm_bytes: bytes, sample_rate: int):
        bpf = max(2, int(round(sample_rate / self.args.fps) * 2))
        if self._audio_bytes_per_frame is None:
            self._tts_sample_rate = sample_rate
            self._audio_bytes_per_frame = bpf
            self._audio_byte_queue = bytearray()
        elif sample_rate != self._tts_sample_rate:
            self._tts_sample_rate = sample_rate
            self._audio_bytes_per_frame = bpf
        self._audio_byte_queue.extend(pcm_bytes)

    def _set_chunk_pcm(self, pcm_bytes: bytes, sample_rate: int):
        """Bind the next lip-sync frames to this TTS chunk's PCM (avoids cross-chunk A/V drift)."""
        bpf = max(2, int(round(sample_rate / self.args.fps) * 2))
        self._tts_sample_rate = sample_rate
        self._audio_bytes_per_frame = bpf
        self._chunk_pcm_buf = bytearray(pcm_bytes)
        self._use_chunk_pcm = True

    def _take_chunk_pcm(self) -> bytes:
        n = self._audio_bytes_per_frame
        if len(self._chunk_pcm_buf) >= n:
            pcm = bytes(self._chunk_pcm_buf[:n])
            del self._chunk_pcm_buf[:n]
            return pcm
        if self._chunk_pcm_buf:
            pcm = bytes(self._chunk_pcm_buf)
            self._chunk_pcm_buf.clear()
            return pcm + b"\x00" * (n - len(pcm))
        return b"\x00" * n

    def _take_paced_pcm(self) -> bytes:
        n = self._audio_bytes_per_frame
        if len(self._audio_byte_queue) >= n:
            pcm = bytes(self._audio_byte_queue[:n])
            del self._audio_byte_queue[:n]
            return pcm
        if self._audio_byte_queue:
            pcm = bytes(self._audio_byte_queue)
            self._audio_byte_queue.clear()
            return pcm + b"\x00" * (n - len(pcm))
        return b"\x00" * n

    def _submit_piped_av_frame(self, frame_bgr: np.ndarray):
        if not self._piped_mux or not self._audio_bytes_per_frame:
            return
        if self._mux_worker is None:
            self._mux_worker = MuxWorker(self)
        if self._use_chunk_pcm:
            pcm = self._take_chunk_pcm()
        else:
            pcm = self._take_paced_pcm()
        self._mux_worker.submit_av_pair(frame_bgr, pcm)

    def push_frame(self, frame_bgr: np.ndarray, preview_only: bool = False):
        """Store a BGR frame for MJPEG preview and optionally mux into progressive MP4."""
        with self._lock:
            self._latest_frame = frame_bgr.copy()
        self._frame_event.set()

        if self._mux_frames and not preview_only:
            if self._current_job_id:
                self._record_first_lipsync_frame(self._current_job_id)
            self._frame_archive.append(frame_bgr.copy())
            if self._piped_mux:
                self._submit_piped_av_frame(frame_bgr)
            elif self._live_segment_mux:
                self._maybe_emit_live_segment()
            elif (
                self._av_output_path
                and self._mp4_buffer is not None
                and self._av_audio_path
            ):
                if self._av_encoder is None:
                    self._av_encoder = FFmpegProgressiveStreamer(
                        self.args.fps,
                        self._av_audio_path,
                        self._mp4_buffer,
                        self._av_output_path,
                    )
                if self._av_encoder is not None:
                    self._av_encoder.write_frame(frame_bgr)

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
        self.push_frame(img, preview_only=True)

    def _get_latest_jpeg(self, timeout=1.0):
        if not self._frame_event.wait(timeout=timeout):
            return None
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return encoded.tobytes() if ok else None

    def _stream_inference_kwargs(self):
        return {
            "batch_size": self.args.stream_batch_size,
            "first_batch_size": self.args.stream_first_batch_size,
            "ramp_batch_size": self.args.stream_ramp_batch_size,
            "ramp_batches": self.args.stream_ramp_batches,
            "skip_save_images": True,
            "frame_sink": self,
            "stream_fps": None,
        }

    def _resolve_video_path(self, form, upload_dir):
        preset_model = form.getvalue("preset_model", "").strip() if "preset_model" in form else ""
        if preset_model:
            if preset_model not in PRESET_MODELS:
                raise ValueError(f"Unknown preset model: {preset_model}")
            video_path = os.path.abspath(PRESET_MODELS[preset_model])
            if not os.path.isfile(video_path):
                raise ValueError(f"Preset video not found for {preset_model}: {video_path}")
            print(f"Using preset {preset_model}: {video_path}")
            return video_path, preset_model

        video_item = form["video"] if "video" in form else None
        if video_item is None or not video_item.filename:
            raise ValueError("Select a preset model or upload a reference video.")

        video_ext = os.path.splitext(video_item.filename)[1] or ".mp4"
        video_path = os.path.join(upload_dir, f"video{video_ext}")
        with open(video_path, "wb") as f:
            f.write(video_item.file.read())
        return video_path, None

    def _save_upload(self, form, job_id):
        upload_dir = os.path.join(UPLOAD_ROOT, job_id)
        os.makedirs(upload_dir, exist_ok=True)

        video_path, preset_model = self._resolve_video_path(form, upload_dir)

        text_val = form.getvalue("text", "").strip() if "text" in form else ""
        audio_token = form.getvalue("audio_token", "").strip() if "audio_token" in form else ""
        audio_item = form["audio"] if "audio" in form else None
        has_audio_file = audio_item is not None and getattr(audio_item, "filename", None)

        if text_val:
            voice = form.getvalue("tts_voice", self.args.tts_voice) if "tts_voice" in form else self.args.tts_voice
            text_path = os.path.join(upload_dir, "text.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text_val)
            print(f"TTS queued (local Piper): voice={voice}, chars={len(text_val)}")
            return video_path, None, True, preset_model, text_val, voice, None
        elif audio_token:
            audio_path, prewarm = self._consume_staged_session(audio_token, upload_dir)
            print(f"Using pre-warmed audio: token={audio_token[:8]}... -> {audio_path}")
            return video_path, audio_path, False, preset_model, None, None, prewarm
        elif has_audio_file:
            audio_ext = os.path.splitext(audio_item.filename)[1] or ".wav"
            raw_path = os.path.join(upload_dir, f"audio_upload{audio_ext}")
            with open(raw_path, "wb") as f:
                f.write(audio_item.file.read())
            audio_path = ensure_wav(raw_path, os.path.join(upload_dir, "audio_16k.wav"))
            print(f"Audio uploaded: {audio_item.filename} -> {audio_path} (16kHz mono)")
        else:
            raise ValueError("Provide text for TTS or upload an audio file.")

        return video_path, audio_path, False, preset_model, None, None, None

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

    def _prepare_tts_chunk(self, idx, chunk_text, voice, upload_dir, use_cuda):
        raw_wav = os.path.join(upload_dir, f"chunk_{idx:03d}.wav")
        whisper_wav = os.path.join(upload_dir, f"chunk_{idx:03d}_16k.wav")
        return prepare_tts_chunk_for_inference(
            chunk_text,
            raw_wav,
            whisper_wav,
            voice=voice,
            model_dir=self.args.piper_model_dir,
            use_cuda=use_cuda,
        )

    def _run_chunked_text_job(
        self,
        text,
        voice,
        avatar,
        job_id,
        upload_dir,
        use_cache,
        memory_cached,
    ):
        chunks = chunk_text_for_streaming(
            text,
            first_max_chars=self.args.tts_first_chunk_chars,
            max_chars=self.args.tts_chunk_chars,
        )
        if not chunks:
            raise ValueError("No text to synthesize.")

        timer = StepTimer(job_id)
        self._job_timers[job_id] = timer
        timer.mark("chunks_ready")

        stream_url = f"/api/progressive/{job_id}.mp4"
        result_url = f"/api/result/{job_id}.mp4"
        audio_url = f"/api/audio/{job_id}"
        self.set_status(
            "streaming",
            f"Streaming {len(chunks)} chunk(s) — live A/V (video+audio) MP4...",
            cached=use_cache or memory_cached,
            job_id=job_id,
            stream_url=stream_url,
            result_url=result_url,
            audio_url=audio_url,
        )
        self._begin_av_mux()
        timer.mark("mux_started")

        chunk_wavs = []
        use_cuda = self._tts_use_cuda()
        prefetch_ahead = max(1, self.args.tts_prefetch_chunks)
        max_workers = min(self.args.tts_prefetch_workers, len(chunks))
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = {}

        def schedule(i):
            if 0 <= i < len(chunks) and i not in futures:
                futures[i] = executor.submit(
                    self._prepare_tts_chunk, i, chunks[i], voice, upload_dir, use_cuda
                )

        try:
            for idx in range(len(chunks)):
                if self._cancel_event.is_set():
                    break
                frame_idx_before = len(self._frame_archive)
                for ahead_i in range(idx, min(idx + prefetch_ahead + 1, len(chunks))):
                    schedule(ahead_i)

                prepared = futures.pop(idx).result()
                raw_wav = prepared["raw_wav"]
                whisper_wav = prepared["whisper_wav"]
                sample_rate = prepared["sample_rate"]
                timer.mark(f"tts_ready_chunk_{idx}")
                self._tts_sample_rate = sample_rate
                chunk_wavs.append(raw_wav)

                if self._piped_mux:
                    self._set_chunk_pcm(prepared["pcm_bytes"], prepared["pcm_sr"])
                if self._live_segment_mux:
                    self._prepare_chunk_stream(raw_wav, sample_rate, frame_idx_before)

                if idx == 0:
                    self.push_status_frame(f"Streaming chunk 1/{len(chunks)}")
                    timer.mark("first_preview_frame")

                avatar.inference(
                    whisper_wav,
                    None,
                    self.args.fps,
                    **self._stream_inference_kwargs(),
                )
                timer.mark(f"inference_chunk_{idx} ({len(self._frame_archive) - frame_idx_before} frames)")

                if self._live_segment_mux:
                    self._flush_chunk_stream()
                    timer.mark(f"segment_flush_chunk_{idx}")
                elif self._segment_mux:
                    chunk_frames = self._frame_archive[frame_idx_before:]
                    if chunk_frames:
                        self._append_chunk_segment(chunk_frames, raw_wav)
                        timer.mark(f"segment_mux_chunk_{idx}")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        timer.mark("concat_audio")
        full_audio = os.path.join(upload_dir, "audio.wav")
        concat_wavs(chunk_wavs, full_audio)
        self._job_audio_paths[job_id] = full_audio
        timer.mark("job_complete")

    def _run_job(
        self,
        video_path,
        audio_path,
        avatar_id,
        need_preparation,
        video_hash,
        use_cache,
        job_id,
        text=None,
        voice=None,
        upload_dir=None,
        prewarm=None,
    ):
        import scripts.realtime_inference as rt

        try:
            avatar_ready = prewarm and prewarm.get("avatar_ready") and prewarm.get("video_hash") == video_hash
            if avatar_ready:
                avatar = self._avatar_cache.get(video_hash)
                if avatar is None:
                    avatar_ready = False
            if not avatar_ready:
                avatar, memory_cached = self._get_or_create_avatar(
                    video_path, avatar_id, need_preparation, video_hash
                )
            else:
                avatar = self._avatar_cache[video_hash]
                avatar.idx = 0
                memory_cached = prewarm.get("memory_cached", True)
                print(f"Using pre-warmed avatar: {avatar_id}")
            use_cache = use_cache or memory_cached

            if need_preparation and not avatar_ready and not self._cancel_event.is_set():
                self._register_cache(video_hash, avatar_id, video_path)

            if self._cancel_event.is_set():
                self.set_status("cancelled", "Job cancelled.", cached=use_cache, job_id=job_id)
                return

            stream_url = f"/api/progressive/{job_id}.mp4"
            result_url = f"/api/result/{job_id}.mp4"
            audio_url = f"/api/audio/{job_id}"

            if text:
                upload_dir = upload_dir or os.path.join(UPLOAD_ROOT, job_id)
                self._run_chunked_text_job(
                    text, voice, avatar, job_id, upload_dir, use_cache, memory_cached
                )
            else:
                if prewarm and prewarm.get("avatar_ready"):
                    self.set_status(
                        "streaming",
                        "Pre-warmed lip-sync stream starting...",
                        cached=True,
                        job_id=job_id,
                        stream_url=stream_url,
                        result_url=result_url,
                        audio_url=audio_url,
                    )
                elif memory_cached:
                    self.set_status(
                        "streaming",
                        "Reusing face model — progressive A/V stream starting...",
                        cached=True,
                        job_id=job_id,
                        stream_url=stream_url,
                        result_url=result_url,
                        audio_url=audio_url,
                    )
                else:
                    self.set_status(
                        "streaming",
                        "Progressive A/V stream in progress (video + audio)...",
                        cached=use_cache,
                        job_id=job_id,
                        stream_url=stream_url,
                        result_url=result_url,
                        audio_url=audio_url,
                    )
                if not prewarm:
                    self._begin_av_mux()
                whisper_chunks = prewarm.get("whisper_chunks") if prewarm else None
                if self._piped_mux:
                    if not (prewarm and prewarm.get("pcm_bytes")):
                        if audio_path and os.path.isfile(audio_path):
                            pcm_bytes, sample_rate = self._load_wav_pcm(audio_path)
                            self._enqueue_chunk_audio(pcm_bytes, sample_rate)
                elif self._live_segment_mux and audio_path and os.path.isfile(audio_path):
                    import soundfile as sf

                    info = sf.info(audio_path)
                    self._prepare_chunk_stream(audio_path, int(info.samplerate), 0)
                avatar.inference(
                    audio_path,
                    None,
                    self.args.fps,
                    whisper_chunks=whisper_chunks,
                    **self._stream_inference_kwargs(),
                )

            timer = self._job_timers.get(job_id)
            if timer:
                timer.mark("mux_finalize_start")
            self._finish_av_stream()
            if timer:
                timer.mark("mux_finalize_done")

            if self._cancel_event.is_set():
                self.set_status("cancelled", "Job cancelled.", cached=use_cache, job_id=job_id)
            else:
                msg = "Stream complete. Upload again to try another clip."
                self.set_status(
                    "done",
                    msg,
                    cached=use_cache,
                    job_id=job_id,
                    result_url=result_url,
                    stream_url=stream_url,
                    audio_url=audio_url,
                )
        except Exception as exc:
            err_msg = str(exc).strip() or f"{type(exc).__name__}"
            self.set_status("error", f"Processing error: {err_msg}", cached=use_cache, job_id=job_id)
            self.push_status_frame("Error: " + err_msg[:60])
            raise
        finally:
            rt.args.cancel_event = None

    def _tts_use_cuda(self) -> bool:
        if not self.args.tts_gpu:
            return False
        if not torch.cuda.is_available():
            return False
        try:
            import onnxruntime as ort

            return "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    def start_job(self, video_path, audio_path, job_id, text=None, voice=None, upload_dir=None, piped=False, live_segment=False, prewarm=None):
        if not self._models_ready.is_set():
            raise RuntimeError("Models are still loading.")

        self._stop_running_job()

        avatar_id, need_preparation, video_hash = self._resolve_avatar(video_path)
        use_cache = not need_preparation

        self._cancel_event = threading.Event()
        import scripts.realtime_inference as rt

        rt.args.cancel_event = self._cancel_event

        upload_dir = upload_dir or os.path.join(UPLOAD_ROOT, job_id)
        self._reset_av_stream(job_id, audio_path=audio_path, piped=piped, upload_dir=upload_dir)
        self._live_segment_mux = live_segment
        self._segment_mux = False
        self._frames_pushed = 0
        if prewarm:
            self._begin_av_mux()
            self._prepare_prewarmed_stream(prewarm)
        self.push_status_frame("Starting...")

        self._worker = threading.Thread(
            target=self._run_job,
            args=(video_path, audio_path, avatar_id, need_preparation, video_hash, use_cache, job_id),
            kwargs={"text": text, "voice": voice, "upload_dir": upload_dir, "prewarm": prewarm},
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _parse_multipart_form(handler):
        content_type = handler.headers.get("Content-Type", "")
        content_length = handler.headers.get("Content-Length", "0")
        if "multipart/form-data" not in content_type:
            raise ValueError("multipart/form-data is required.")
        return cgi.FieldStorage(
            fp=handler.rfile,
            headers=handler.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": content_length,
            },
        )

    def handle_audio_upload(self, handler):
        form = self._parse_multipart_form(handler)
        audio_item = form["audio"] if "audio" in form else None
        if audio_item is None or not getattr(audio_item, "filename", None):
            raise ValueError("No audio file in upload.")

        preset_model = form.getvalue("preset_model", "").strip() if "preset_model" in form else ""

        token = uuid.uuid4().hex
        staging_dir = os.path.join(UPLOAD_ROOT, "staging", token)
        os.makedirs(staging_dir, exist_ok=True)
        audio_ext = os.path.splitext(audio_item.filename)[1] or ".wav"
        raw_path = os.path.join(staging_dir, f"audio_upload{audio_ext}")
        with open(raw_path, "wb") as f:
            f.write(audio_item.file.read())
        audio_path = ensure_wav(raw_path, os.path.join(staging_dir, "audio_16k.wav"))
        print(f"Audio staged: {audio_item.filename} -> {audio_path} (token={token[:8]}...)")

        with self._staging_lock:
            self._audio_staging[token] = {
                "path": audio_path,
                "filename": audio_item.filename,
                "preset_model": preset_model or None,
                "warmup_state": "pending",
                "warmup_message": "Queued for warmup",
                "warmup_error": None,
            }

        threading.Thread(
            target=self._warmup_staged_audio,
            args=(token,),
            daemon=True,
        ).start()

        return {
            "ok": True,
            "audio_token": token,
            "filename": audio_item.filename,
            "warming": True,
            "message": "Audio uploaded — warming up model and GPU...",
        }

    def get_audio_warmup_status(self, token: str):
        with self._staging_lock:
            entry = self._audio_staging.get(token)
        if not entry:
            return {"ok": False, "error": "Unknown or expired audio token.", "state": "error"}
        state = entry.get("warmup_state", "pending")
        payload = {
            "ok": True,
            "state": state,
            "filename": entry.get("filename"),
            "message": entry.get("warmup_message") or "",
        }
        if state == "error":
            payload["error"] = entry.get("warmup_error") or "Warmup failed"
        return payload

    def _set_staging_warmup(self, token: str, **updates):
        with self._staging_lock:
            entry = self._audio_staging.get(token)
            if entry is not None:
                entry.update(updates)

    def _warmup_staged_audio(self, token: str):
        if not self._models_ready.is_set():
            self._models_ready.wait(timeout=600)
        with self._staging_lock:
            entry = self._audio_staging.get(token)
            if entry is None:
                return
        self._set_staging_warmup(token, warmup_state="warming", warmup_message="Extracting Whisper features on GPU...")
        try:
            import scripts.realtime_inference as rt

            audio_path = entry["path"]
            preset_model = entry.get("preset_model")

            whisper_input_features, librosa_length = rt.audio_processor.get_audio_feature(
                audio_path, weight_dtype=rt.weight_dtype
            )
            whisper_chunks = rt.audio_processor.get_whisper_chunk(
                whisper_input_features,
                rt.device,
                rt.weight_dtype,
                rt.whisper,
                librosa_length,
                fps=self.args.fps,
                audio_padding_length_left=self.args.audio_padding_length_left,
                audio_padding_length_right=self.args.audio_padding_length_right,
            )
            pcm_bytes, pcm_sr = self._load_wav_pcm(audio_path)

            prewarm = {
                "whisper_chunks": whisper_chunks,
                "pcm_bytes": pcm_bytes,
                "pcm_sr": pcm_sr,
                "avatar_ready": False,
                "memory_cached": False,
                "video_hash": None,
                "video_path": None,
                "preset_model": preset_model,
            }

            if preset_model and preset_model in PRESET_MODELS:
                self._set_staging_warmup(token, warmup_message=f"Loading preset {preset_model} face model...")
                video_path = os.path.abspath(PRESET_MODELS[preset_model])
                avatar_id, need_preparation, video_hash = self._resolve_avatar(video_path)
                avatar, memory_cached = self._get_or_create_avatar(
                    video_path, avatar_id, need_preparation, video_hash, quiet=True
                )
                if need_preparation and not self._cancel_event.is_set():
                    self._register_cache(video_hash, avatar_id, video_path)
                prewarm.update(
                    {
                        "avatar_ready": True,
                        "memory_cached": memory_cached,
                        "video_hash": video_hash,
                        "video_path": video_path,
                        "avatar_id": avatar_id,
                    }
                )
                self._set_staging_warmup(
                    token,
                    warmup_message="Running GPU inference warmup...",
                )
                if torch.cuda.is_available() and whisper_chunks.shape[0] > 0:
                    batch_n = min(self.args.stream_first_batch_size, whisper_chunks.shape[0])
                    dummy_latent = avatar.input_latent_list_cycle[:batch_n]
                    from musetalk.utils.utils import datagen

                    gen = datagen(
                        whisper_chunks[:batch_n],
                        dummy_latent,
                        batch_n,
                        first_batch_size=batch_n,
                    )
                    whisper_batch, latent_batch = next(iter(gen))
                    audio_feature_batch = rt.pe(whisper_batch.to(rt.device, non_blocking=True))
                    latent_batch = latent_batch.to(device=rt.device, dtype=rt.unet.model.dtype, non_blocking=True)
                    pred_latents = rt.unet.model(
                        latent_batch,
                        rt.timesteps,
                        encoder_hidden_states=audio_feature_batch,
                    ).sample
                    rt.vae.decode_latents(pred_latents.to(device=rt.device, dtype=rt.vae.vae.dtype))
                    torch.cuda.synchronize()

            self._set_staging_warmup(
                token,
                warmup_state="ready",
                warmup_message="Ready — click Start streaming",
                warmup_error=None,
                prewarm=prewarm,
            )
            print(f"Audio warmup ready: token={token[:8]}..., frames={whisper_chunks.shape[0]}")
        except Exception as exc:
            print(f"Audio warmup failed ({token[:8]}...): {exc}")
            self._set_staging_warmup(
                token,
                warmup_state="error",
                warmup_error=str(exc),
                warmup_message=f"Warmup failed: {exc}",
            )

    def _consume_staged_session(self, token, upload_dir):
        with self._staging_lock:
            entry = self._audio_staging.pop(token, None)
        if not entry:
            raise ValueError("Audio upload expired or invalid. Please upload the audio file again.")
        if entry.get("warmup_state") != "ready":
            raise ValueError("Audio is still warming up. Please wait until preparation completes.")
        src = entry["path"]
        if not os.path.isfile(src):
            raise ValueError("Uploaded audio file is missing. Please upload again.")
        dest = os.path.join(upload_dir, "audio_16k.wav")
        shutil.copy2(src, dest)
        prewarm = dict(entry.get("prewarm") or {})
        prewarm["staging_path"] = src
        return dest, prewarm

    def handle_upload(self, handler):
        form = self._parse_multipart_form(handler)

        job_id = uuid.uuid4().hex
        upload_dir = os.path.join(UPLOAD_ROOT, job_id)
        video_path, audio_path, used_tts, preset_model, text_val, voice, prewarm = self._save_upload(form, job_id)
        if audio_path:
            self._job_audio_paths[job_id] = audio_path
        avatar_id, need_preparation, video_hash = self._resolve_avatar(video_path)
        memory_cached = video_hash in self._avatar_cache and not need_preparation
        if preset_model and self.is_preset_preloaded(preset_model):
            memory_cached = True
        if prewarm and prewarm.get("avatar_ready"):
            memory_cached = True

        if used_tts:
            tts_note = "Local TTS streaming. "
        elif prewarm:
            tts_note = "Pre-warmed — "
        else:
            tts_note = ""
        if preset_model:
            model_note = f"Preset {preset_model}. "
        else:
            model_note = ""
        if prewarm and prewarm.get("avatar_ready"):
            message = model_note + tts_note + "Starting lip-sync stream immediately."
        elif memory_cached:
            message = model_note + tts_note + "Reusing loaded face model — starting stream."
        elif need_preparation:
            message = model_note + tts_note + "Running face analysis, then A/V streaming will start."
        else:
            message = model_note + tts_note + "Loading face model from disk — then streaming."

        self.set_status(
            "uploading",
            message,
            cached=not need_preparation,
            job_id=job_id,
            audio_url=f"/api/audio/{job_id}",
            stream_url=f"/api/progressive/{job_id}.mp4",
            result_url=f"/api/result/{job_id}.mp4",
        )
        self.start_job(
            video_path,
            audio_path,
            job_id,
            text=text_val,
            voice=voice,
            upload_dir=upload_dir,
            piped=self.args.progressive_mode == "piped",
            live_segment=self.args.progressive_mode == "segment",
            prewarm=prewarm,
        )

        return {
            "ok": True,
            "job_id": job_id,
            "avatar_id": avatar_id,
            "cached": not need_preparation,
            "memory_cached": memory_cached,
            "used_tts": used_tts,
            "preset_model": preset_model,
            "audio_url": f"/api/audio/{job_id}",
            "stream_url": f"/api/progressive/{job_id}.mp4",
            "result_url": f"/api/result/{job_id}.mp4",
            "message": message,
        }

    def serve_progressive_mp4(self, handler, job_id: str, *, send_body: bool = True):
        """Serve live fMP4 while generating; serve progressive.mp4 with Range when complete."""
        buffer = self._job_mp4_buffers.get(job_id)
        fpath = progressive_mp4_path(job_id)
        live = buffer is not None and not buffer.closed

        if buffer is None and (not os.path.isfile(fpath) or os.path.getsize(fpath) == 0):
            handler.send_error(404)
            return

        if live:
            handler.send_response(200)
            handler.send_header("Content-Type", "video/mp4")
            handler.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            handler.send_header("Pragma", "no-cache")
            handler.send_header("X-Accel-Buffering", "no")
            if not send_body:
                handler.end_headers()
                return
            handler.send_header("Connection", "close")
            handler.end_headers()
            try:
                for chunk in buffer.iter_chunks():
                    handler.wfile.write(chunk)
                    handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            return

        file_size = os.path.getsize(fpath)
        range_header = handler.headers.get("Range") if send_body else None
        if range_header:
            try:
                _, rng = range_header.split("=")
                start_s, end_s = rng.split("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else file_size - 1
                end = min(end, file_size - 1)
            except ValueError:
                handler.send_error(416)
                return
            if not send_body:
                handler.send_response(206)
                handler.send_header("Content-Type", "video/mp4")
                handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                handler.send_header("Accept-Ranges", "bytes")
                handler.send_header("Content-Length", str(end - start + 1))
                handler.end_headers()
                return
            with open(fpath, "rb") as f:
                f.seek(start)
                data = f.read(end - start + 1)
            handler.send_response(206)
            handler.send_header("Content-Type", "video/mp4")
            handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            handler.send_header("Content-Length", str(len(data)))
            handler.send_header("Accept-Ranges", "bytes")
            handler.send_header("Cache-Control", "no-cache")
            handler.end_headers()
            handler.wfile.write(data)
            return

        handler.send_response(200)
        handler.send_header("Content-Type", "video/mp4")
        handler.send_header("Content-Length", str(file_size))
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        if send_body:
            with open(fpath, "rb") as f:
                handler.wfile.write(f.read())

    def warmup_streaming_inference(self):
        """Run one short lip-sync pass on a preloaded preset to warm cuDNN kernels."""
        if not self.is_preset_preloaded("model1"):
            return
        import tempfile
        import scripts.realtime_inference as rt

        video_path = os.path.abspath(PRESET_MODELS["model1"])
        video_hash = compute_file_hash(video_path)
        avatar = self._avatar_cache.get(video_hash)
        if avatar is None:
            return
        try:
            with tempfile.TemporaryDirectory() as tmp:
                raw_wav = os.path.join(tmp, "warmup.wav")
                whisper_wav = os.path.join(tmp, "warmup_16k.wav")
                synthesize_chunk_to_wav(
                    "Hi.",
                    raw_wav,
                    voice=self.args.tts_voice,
                    model_dir=self.args.piper_model_dir,
                    use_cuda=self._tts_use_cuda(),
                )
                resample_wav_for_whisper(raw_wav, whisper_wav)
                avatar.inference(
                    whisper_wav,
                    None,
                    self.args.fps,
                    skip_save_images=True,
                    frame_sink=None,
                    batch_size=self.args.stream_batch_size,
                    first_batch_size=self.args.stream_first_batch_size,
                )
            print("Streaming inference warmup complete (model1)")
        except Exception as exc:
            print(f"Warning: streaming inference warmup skipped: {exc}")

    def serve_forever(self):
        service = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):
                return

            def do_HEAD(self):
                parsed = urlparse(self.path)
                path = parsed.path

                if path.startswith("/api/result/"):
                    job_id = path.replace("/api/result/", "").replace(".mp4", "")
                    fpath = os.path.join(UPLOAD_ROOT, job_id, "output.mp4")
                    if not os.path.isfile(fpath) or os.path.getsize(fpath) == 0:
                        self.send_error(404)
                        return
                    file_size = os.path.getsize(fpath)
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(file_size))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    return

                if path.startswith("/api/audio/"):
                    job_id = path.replace("/api/audio/", "").split("?")[0]
                    audio_path = service._job_audio_paths.get(job_id)
                    if not audio_path or not os.path.isfile(audio_path):
                        self.send_error(404)
                        return
                    mime, _ = mimetypes.guess_type(audio_path)
                    if not mime:
                        mime = "application/octet-stream"
                    file_size = os.path.getsize(audio_path)
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(file_size))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    return

                if path.startswith("/api/progressive/") or path.startswith("/api/stream/"):
                    job_id = path.replace("/api/progressive/", "").replace("/api/stream/", "").replace(".mp4", "")
                    service.serve_progressive_mp4(self, job_id, send_body=False)
                    return

                if path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    return

                self.send_error(404)

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

                if path == "/api/presets":
                    presets = {
                        k: {"label": k, "file": os.path.basename(v), "available": os.path.isfile(os.path.abspath(v))}
                        for k, v in PRESET_MODELS.items()
                    }
                    self._send_json({"presets": presets})
                    return

                if path == "/health":
                    self._send_json({"ok": True, "state": service.get_status().get("state")})
                    return

                if path == "/api/status":
                    self._send_json(service.get_status())
                    return

                if path.startswith("/api/audio-warmup/"):
                    token = path.replace("/api/audio-warmup/", "").split("?")[0]
                    self._send_json(service.get_audio_warmup_status(token))
                    return

                if path.startswith("/api/progressive/") or path.startswith("/api/stream/"):
                    job_id = path.replace("/api/progressive/", "").replace("/api/stream/", "").replace(".mp4", "")
                    service.serve_progressive_mp4(self, job_id, send_body=True)
                    return

                if path.startswith("/api/result/"):
                    job_id = path.replace("/api/result/", "").replace(".mp4", "")
                    fpath = os.path.join(UPLOAD_ROOT, job_id, "output.mp4")
                    if not os.path.isfile(fpath) or os.path.getsize(fpath) == 0:
                        self.send_error(404)
                        return
                    file_size = os.path.getsize(fpath)
                    range_header = self.headers.get("Range")
                    if range_header:
                        try:
                            units, rng = range_header.split("=")
                            start_s, end_s = rng.split("-")
                            start = int(start_s) if start_s else 0
                            end = int(end_s) if end_s else file_size - 1
                            end = min(end, file_size - 1)
                        except ValueError:
                            self.send_error(416)
                            return
                        with open(fpath, "rb") as f:
                            f.seek(start)
                            data = f.read(end - start + 1)
                        self.send_response(206)
                        self.send_header("Content-Type", "video/mp4")
                        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                        self.send_header("Content-Length", str(len(data)))
                        self.send_header("Accept-Ranges", "bytes")
                        self.end_headers()
                        self.wfile.write(data)
                        return
                    with open(fpath, "rb") as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(data)
                    return

                if path.startswith("/api/audio/"):
                    job_id = path.replace("/api/audio/", "").split("?")[0]
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
                path = urlparse(self.path).path
                if path == "/api/upload-audio":
                    try:
                        payload = service.handle_audio_upload(self)
                        self._send_json(payload)
                    except Exception as exc:
                        self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return
                if path != "/api/start":
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
    parser.add_argument("--port", type=int, default=7860)
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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for avatar preparation / file jobs")
    parser.add_argument("--parsing_mode", default="jaw")
    parser.add_argument("--left_cheek_width", type=int, default=90)
    parser.add_argument("--right_cheek_width", type=int, default=90)
    parser.add_argument("--tts_voice", type=str, default=TTS_VOICES["en"])
    parser.add_argument("--piper_model_dir", type=str, default="./models/piper")
    parser.add_argument(
        "--stream_batch_size",
        type=int,
        default=8,
        help="UNet batch size after the first streaming batch",
    )
    parser.add_argument(
        "--stream_first_batch_size",
        type=int,
        default=1,
        help="First UNet batch — 1 minimizes first-frame latency",
    )
    parser.add_argument(
        "--stream_ramp_batch_size",
        type=int,
        default=4,
        help="UNet batch size for the first few batches after the first frame (smooth 1→2 transition)",
    )
    parser.add_argument(
        "--stream_ramp_batches",
        type=int,
        default=4,
        help="Number of ramp batches before switching to --stream_batch_size",
    )
    parser.add_argument(
        "--stream_emit_frames",
        type=int,
        default=1,
        help="Emit progressive MP4 segment every N lip-sync frames (segment mode only)",
    )
    parser.add_argument(
        "--tts_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU (ONNX CUDA) for local Piper TTS when available",
    )
    parser.add_argument(
        "--progressive_mode",
        type=str,
        default="piped",
        choices=["piped", "segment"],
        help="piped: single FFmpeg fMP4 session (lowest latency); segment: batch mux per N frames",
    )
    parser.add_argument(
        "--fmp4_frag_us",
        type=int,
        default=0,
        help="fMP4 fragment duration in microseconds (0 = one video frame at --fps, fastest start)",
    )
    parser.add_argument("--tts_first_chunk_chars", type=int, default=18, help="Max chars in first TTS chunk for low latency")
    parser.add_argument("--tts_chunk_chars", type=int, default=96, help="Max chars per follow-up TTS chunk")
    parser.add_argument(
        "--tts_prefetch_workers",
        type=int,
        default=4,
        help="Parallel Piper TTS workers (overlap synthesis with lip-sync inference)",
    )
    parser.add_argument(
        "--tts_prefetch_chunks",
        type=int,
        default=4,
        help="Number of upcoming text chunks to pre-synthesize ahead of inference",
    )
    parser.add_argument(
        "--ffmpeg_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NVIDIA NVENC (h264_nvenc) for FFmpeg video encoding when available",
    )
    parser.add_argument(
        "--preload_presets",
        type=str,
        default="model1",
        help="Comma-separated preset IDs to preload at startup (e.g. model1,model2). Empty to disable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not fast_check_ffmpeg():
        path_separator = ";" if sys.platform == "win32" else ":"
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"

    os.makedirs(UPLOAD_ROOT, exist_ok=True)

    preload_list = [p.strip() for p in args.preload_presets.split(",") if p.strip()]
    args.preload_presets = preload_list

    if args.fmp4_frag_us <= 0:
        # One video frame per fragment — fastest first MSE append.
        args.fmp4_frag_us = max(10000, 1_000_000 // max(1, args.fps))

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    configure_cuda(device)
    autotune_batch_sizes(args, device)

    global USE_FFMPEG_GPU
    USE_FFMPEG_GPU = bool(args.ffmpeg_gpu and probe_ffmpeg_nvenc())
    if args.ffmpeg_gpu and not USE_FFMPEG_GPU:
        print("FFmpeg GPU (h264_nvenc) unavailable — using libx264 on CPU for mux", flush=True)
    elif USE_FFMPEG_GPU:
        print("FFmpeg video encoding: h264_nvenc (GPU). Audio AAC remains on CPU.", flush=True)
    print(
        f"Progressive MP4 mode: {args.progressive_mode} "
        f"(fMP4 frag={args.fmp4_frag_us}us, emit_every={args.stream_emit_frames} frames in segment mode)",
        flush=True,
    )
    print(
        f"TTS pipeline: prefetch_workers={args.tts_prefetch_workers}, "
        f"prefetch_chunks={args.tts_prefetch_chunks}, gpu={args.tts_gpu}",
        flush=True,
    )

    service = StreamWebService(args)
    service.set_status("loading", "Loading models... please wait.")

    def load_and_start():
        models = load_models(args, device)
        bind_realtime_globals(args, device, models)
        warmup_inference_models(device, models)
        service._models_ready.set()
        try:
            service.set_status("loading", "Loading local Piper TTS...")
            use_cuda = service._tts_use_cuda()
            engine = get_tts_engine(args.tts_voice, args.piper_model_dir, use_cuda=use_cuda)
            import tempfile
            import wave

            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                with wave.open(tmp.name, "wb") as wf:
                    engine.synthesize_wav("Hi.", wf)
            mode = "GPU (ONNX CUDA)" if use_cuda else "CPU"
            print(f"Local Piper TTS ready ({mode})")
        except Exception as exc:
            print(f"Warning: Piper TTS preload failed: {exc}")
        if args.preload_presets:
            service.set_status("loading", "Preloading Model 1 face analysis...")
            service.push_status_frame("Preloading Model 1...")
            service.preload_default_presets()
            if service.is_preset_preloaded("model1"):
                service.set_status("loading", "Warming up GPU inference pipeline...")
                service.warmup_streaming_inference()
                msg = "Ready. Model 1 preloaded — fastest start."
            else:
                msg = "Ready. Upload a video and text or audio."
        else:
            msg = "Ready. Upload a video and text or audio."
        service.set_status("idle", msg)
        service.push_status_frame("Ready - Model 1 loaded" if service.is_preset_preloaded("model1") else "Ready")

    threading.Thread(target=load_and_start, daemon=True).start()
    service.serve_forever()


if __name__ == "__main__":
    main()
