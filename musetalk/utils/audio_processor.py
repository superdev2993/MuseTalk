import math
import os

import numpy as np
import soundfile as sf
import torch
from einops import rearrange
from transformers import AutoFeatureExtractor


class AudioProcessor:
    def __init__(self, feature_extractor_path="openai/whisper-tiny/"):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)

    def get_audio_feature(self, wav_path, start_index=0, weight_dtype=None):
        if not os.path.exists(wav_path):
            return None
        librosa_output, sampling_rate = sf.read(wav_path, dtype="float32")
        if librosa_output.ndim > 1:
            librosa_output = librosa_output.mean(axis=1)
        assert sampling_rate == 16000
        segment_length = 30 * sampling_rate
        segments = [librosa_output[i:i + segment_length] for i in range(0, len(librosa_output), segment_length)]

        features = []
        for segment in segments:
            audio_feature = self.feature_extractor(
                segment,
                return_tensors="pt",
                sampling_rate=sampling_rate,
            ).input_features
            if weight_dtype is not None:
                audio_feature = audio_feature.to(dtype=weight_dtype)
            features.append(audio_feature)

        return features, len(librosa_output)

    def get_whisper_chunk(
        self,
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=25,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    ):
        audio_feature_length_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1)
        whisper_feature = []
        for input_feature in whisper_input_features:
            input_feature = input_feature.to(device).to(weight_dtype)
            audio_feats = whisper.encoder(input_feature, output_hidden_states=True).hidden_states
            audio_feats = torch.stack(audio_feats, dim=2)
            whisper_feature.append(audio_feats)

        whisper_feature = torch.cat(whisper_feature, dim=1)
        sr = 16000
        audio_fps = 50
        fps = int(fps)
        whisper_idx_multiplier = audio_fps / fps
        num_frames = math.floor((librosa_length / sr) * fps)
        actual_length = math.floor((librosa_length / sr) * audio_fps)
        whisper_feature = whisper_feature[:, :actual_length, ...]

        padding_nums = math.ceil(whisper_idx_multiplier)
        whisper_feature = torch.cat([
            torch.zeros_like(whisper_feature[:, :padding_nums * audio_padding_length_left]),
            whisper_feature,
            torch.zeros_like(whisper_feature[:, :padding_nums * 3 * audio_padding_length_right]),
        ], 1)

        if num_frames <= 0:
            return torch.empty(0, device=whisper_feature.device, dtype=whisper_feature.dtype)

        frame_indices = torch.arange(num_frames, device=whisper_feature.device, dtype=torch.float32)
        audio_indices = torch.floor(frame_indices * whisper_idx_multiplier).long()
        clip_offsets = torch.arange(audio_feature_length_per_frame, device=whisper_feature.device)
        col_idx = audio_indices.unsqueeze(1) + clip_offsets.unsqueeze(0)
        col_idx = col_idx.clamp(0, whisper_feature.shape[1] - 1)

        feat = whisper_feature.squeeze(0)
        clips = feat[col_idx]
        audio_prompts = rearrange(clips, "b l1 l2 h -> b (l1 l2) h")
        return audio_prompts


if __name__ == "__main__":
    audio_processor = AudioProcessor()
    wav_path = "./2.wav"
    audio_feature, librosa_feature_length = audio_processor.get_audio_feature(wav_path)
    print("Audio Feature shape:", audio_feature[0].shape)
    print("librosa_feature_length:", librosa_feature_length)
