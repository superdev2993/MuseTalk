"""
Stream MuseTalk output to a browser via MJPEG.

Requires a prepared avatar (preparation=True on first run, then False).
Open http://127.0.0.1:8080/ while inference is running.
"""

import argparse
import os
import sys
import time

import torch
from omegaconf import OmegaConf
from transformers import WhisperModel

from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.stream_server import MJPEGStreamServer
from musetalk.utils.utils import load_all_model
from scripts.realtime_inference import Avatar, fast_check_ffmpeg


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


def parse_args():
    parser = argparse.ArgumentParser(description="MuseTalk MJPEG live stream")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"])
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--vae_type", type=str, default="sd-vae")
    parser.add_argument(
        "--unet_config",
        type=str,
        default="./models/musetalkV15/musetalk.json",
    )
    parser.add_argument(
        "--unet_model_path",
        type=str,
        default="./models/musetalkV15/unet.pth",
    )
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper")
    parser.add_argument(
        "--inference_config",
        type=str,
        default="configs/inference/stream.yaml",
    )
    parser.add_argument("--extra_margin", type=int, default=10)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--audio_padding_length_left", type=int, default=2)
    parser.add_argument("--audio_padding_length_right", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--parsing_mode", default="jaw")
    parser.add_argument("--left_cheek_width", type=int, default=90)
    parser.add_argument("--right_cheek_width", type=int, default=90)
    parser.add_argument("--stream_host", type=str, default="0.0.0.0")
    parser.add_argument("--stream_port", type=int, default=8080)
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop audio clips continuously for demo streaming",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not fast_check_ffmpeg():
        path_separator = ";" if sys.platform == "win32" else ":"
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = load_models(args, device)
    bind_realtime_globals(args, device, models)

    stream_server = MJPEGStreamServer(host=args.stream_host, port=args.stream_port)
    stream_server.start()

    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)

    try:
        while True:
            for avatar_id in inference_config:
                entry = inference_config[avatar_id]
                bbox_shift = 0 if args.version == "v15" else entry["bbox_shift"]
                avatar = Avatar(
                    avatar_id=avatar_id,
                    video_path=entry["video_path"],
                    bbox_shift=bbox_shift,
                    batch_size=args.batch_size,
                    preparation=entry["preparation"],
                    non_interactive=True,
                )

                for audio_name, audio_path in entry["audio_clips"].items():
                    print(f"Streaming avatar={avatar_id} audio={audio_path}")
                    avatar.inference(
                        audio_path,
                        audio_name,
                        args.fps,
                        skip_save_images=True,
                        frame_sink=stream_server,
                        stream_fps=args.fps,
                    )

            if not args.loop:
                break
            print("Looping audio clips...")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping stream.")
    finally:
        stream_server.stop()


if __name__ == "__main__":
    main()
