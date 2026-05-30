"""Offline-first TTS (Piper). edge-tts is no longer required."""

from musetalk.utils.local_tts import (
    DEFAULT_VOICE,
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

__all__ = [
    "DEFAULT_VOICE",
    "TTS_VOICES",
    "VOICE_LABELS",
    "chunk_text_for_streaming",
    "concat_wavs",
    "get_tts_engine",
    "prepare_tts_chunk_for_inference",
    "resample_wav_for_whisper",
    "synthesize_chunk_to_wav",
    "synthesize_speech",
    "wav_to_pcm_s16le",
]
