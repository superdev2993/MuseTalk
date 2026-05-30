"""Local offline TTS via Piper (ONNX) with semantic text chunking for streaming."""

from __future__ import annotations

import os
import re
import threading
import wave
from typing import Iterable, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

TTS_VOICES = {
    "en": "en_US-lessac-medium",
    "en-f": "en_US-lessac-medium",
    "en-m": "en_US-lessac-medium",
    "ko": "en_US-lessac-medium",
    "ja": "en_US-lessac-medium",
    "zh": "en_US-lessac-medium",
}

VOICE_LABELS = {
    "en_US-lessac-medium": "English — Lessac (local)",
}

DEFAULT_VOICE = TTS_VOICES["en"]
DEFAULT_MODEL_DIR = "./models/piper"
WHISPER_SAMPLE_RATE = 16000

_ENGINE_LOCK = threading.Lock()
_ENGINE: dict = {}


def _resolve_voice_id(voice: str) -> str:
    voice = (voice or DEFAULT_VOICE).strip()
    if voice in VOICE_LABELS:
        return voice
    if voice in TTS_VOICES.values():
        return voice
    if voice.startswith("en"):
        return TTS_VOICES["en"]
    if voice.startswith("ko"):
        return TTS_VOICES.get("ko", DEFAULT_VOICE)
    if voice.startswith("ja"):
        return TTS_VOICES.get("ja", DEFAULT_VOICE)
    if voice.startswith("zh"):
        return TTS_VOICES.get("zh", DEFAULT_VOICE)
    return DEFAULT_VOICE


def _model_paths(voice_id: str, model_dir: str = DEFAULT_MODEL_DIR) -> Tuple[str, str]:
    base = os.path.join(model_dir, voice_id, f"{voice_id}.onnx")
    if os.path.isfile(base):
        return base, f"{base}.json"
    alt = os.path.join(model_dir, f"{voice_id}.onnx")
    if os.path.isfile(alt):
        return alt, f"{alt}.json"
    raise FileNotFoundError(
        f"Piper voice not found: {voice_id}. "
        f"Download models into {model_dir}/ (see scripts/download_piper_voices.sh)."
    )


def get_tts_engine(voice: str = DEFAULT_VOICE, model_dir: str = DEFAULT_MODEL_DIR):
    voice_id = _resolve_voice_id(voice)
    with _ENGINE_LOCK:
        cached = _ENGINE.get((voice_id, model_dir))
        if cached is not None:
            return cached
        from piper import PiperVoice

        model_path, config_path = _model_paths(voice_id, model_dir)
        engine = PiperVoice.load(model_path, config_path=config_path)
        _ENGINE[voice_id, model_dir] = engine
        return engine


def chunk_text_for_streaming(
    text: str,
    first_max_chars: int = 28,
    max_chars: int = 96,
) -> List[str]:
    """Split text into short first chunk + semantic follow-up chunks."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    units: List[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            units.append(sentence)
            continue
        clauses = re.split(r"(?<=[,，;；:：])\s+", sentence)
        buf = ""
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            candidate = f"{buf} {clause}".strip() if buf else clause
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                if buf:
                    units.append(buf)
                if len(clause) <= max_chars:
                    buf = clause
                else:
                    words = clause.split()
                    buf = ""
                    for word in words:
                        candidate = f"{buf} {word}".strip() if buf else word
                        if len(candidate) > max_chars and buf:
                            units.append(buf)
                            buf = word
                        else:
                            buf = candidate
                    if buf:
                        units.append(buf)
                    buf = ""
        if buf:
            units.append(buf)

    if not units:
        units = [text]

    first = units[0]
    if len(first) > first_max_chars:
        words = first.split()
        head: List[str] = []
        length = 0
        for word in words:
            extra = len(word) if not head else len(word) + 1
            if head and length + extra > first_max_chars:
                break
            head.append(word)
            length += extra
        if head and len(head) < len(words):
            tail = " ".join(words[len(head) :]).strip()
            rest = ([tail] if tail else []) + units[1:]
            return [" ".join(head)] + rest

    return units


def synthesize_chunk_to_wav(
    text: str,
    output_path: str,
    voice: str = DEFAULT_VOICE,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> Tuple[int, int]:
    """Synthesize one chunk to WAV. Returns (sample_rate, num_samples)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("TTS chunk text is empty.")

    engine = get_tts_engine(voice, model_dir)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with wave.open(output_path, "wb") as wav_file:
        engine.synthesize_wav(text, wav_file)

    with wave.open(output_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_samples = wav_file.getnframes()
    return sample_rate, num_samples


def wav_to_pcm_s16le(wav_path: str) -> Tuple[bytes, int]:
    audio, sample_rate = sf.read(wav_path, dtype="int16")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.int16)
    return audio.tobytes(), int(sample_rate)


def resample_wav_for_whisper(src_wav: str, dst_wav: str, sample_rate: int = WHISPER_SAMPLE_RATE):
    import numpy as np

    audio, sr = sf.read(src_wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        duration = len(audio) / sr
        target_len = max(1, int(round(duration * sample_rate)))
        x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    sf.write(dst_wav, audio, sample_rate)


def synthesize_speech(text: str, output_path: str, voice: str = DEFAULT_VOICE, model_dir: str = DEFAULT_MODEL_DIR):
    """Full-text offline synthesis (non-streaming fallback)."""
    chunks = chunk_text_for_streaming(text, first_max_chars=10_000, max_chars=10_000)
    if not chunks:
        raise ValueError("TTS text is empty.")

    if len(chunks) == 1:
        synthesize_chunk_to_wav(chunks[0], output_path, voice=voice, model_dir=model_dir)
        return

    tmp_paths: List[str] = []
    try:
        base, ext = os.path.splitext(output_path)
        for idx, chunk in enumerate(chunks):
            chunk_path = f"{base}.part{idx:03d}{ext or '.wav'}"
            synthesize_chunk_to_wav(chunk, chunk_path, voice=voice, model_dir=model_dir)
            tmp_paths.append(chunk_path)
        concat_wavs(tmp_paths, output_path)
    finally:
        for path in tmp_paths:
            if os.path.isfile(path):
                os.remove(path)


def concat_wavs(paths: Iterable[str], output_path: str):
    paths = list(paths)
    if not paths:
        raise ValueError("No WAV files to concatenate.")
    audio_parts = []
    sample_rate = None
    for path in paths:
        audio, sr = librosa.load(path, sr=None, mono=True)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        audio_parts.append(audio)
    merged = np.concatenate(audio_parts)
    sf.write(output_path, merged, sample_rate)
