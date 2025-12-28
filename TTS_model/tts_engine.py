import argparse
import importlib
import inspect
import os
import tempfile
from typing import Optional

import soundfile as sf
import torch

DEFAULT_CKPT_PATH = "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
DEFAULT_VOCAB_PATH = "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt"


def _resolve_dit_cls():
    model_mod = importlib.import_module("f5_tts.model")
    for name in ("DiT", "DiTModel", "DiT_TTS", "DiTBase"):
        if hasattr(model_mod, name):
            return getattr(model_mod, name)
    for name in dir(model_mod):
        obj = getattr(model_mod, name)
        if "DiT" in name and isinstance(obj, type):
            return obj
    raise ImportError(
        "Unable to locate the DiT class in f5_tts.model. "
        "Check f5_tts/model/__init__.py for the correct class name."
    )


def _trim_silence_wav(
    input_path: str,
    min_silence_len: int = 1000,
    silence_thresh: int = -40,
) -> str:
    try:
        from pydub import AudioSegment, silence
    except Exception as exc:
        raise RuntimeError(
            "pydub is required for trim_silence. Install it with `pip install pydub`."
        ) from exc

    audio = AudioSegment.from_file(input_path)
    chunks = silence.split_on_silence(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )
    if not chunks:
        return input_path

    # Keep the longest non-silent segment to reduce leading/trailing silence.
    chunk = max(chunks, key=len)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    chunk.export(tmp_path, format="wav")
    return tmp_path


def _resolve_hf_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    path = str(path)
    if path.startswith("hf://"):
        try:
            from cached_path import cached_path
        except Exception as exc:
            raise RuntimeError(
                "cached_path is required to download Hugging Face assets. "
                "Install it or pass a local checkpoint path."
            ) from exc
        return str(cached_path(path))
    return path


def _resolve_ref_audio_path(ref_audio_path: str) -> str:
    ref_audio_path = os.fspath(ref_audio_path)
    if os.path.isdir(ref_audio_path):
        wav_files = sorted(
            name
            for name in os.listdir(ref_audio_path)
            if os.path.isfile(os.path.join(ref_audio_path, name))
            and name.lower().endswith(".wav")
        )
        if not wav_files:
            raise FileNotFoundError(f"No .wav files found in {ref_audio_path}")
        if len(wav_files) > 1:
            print(f"Multiple .wav files found; using {wav_files[0]}")
        return os.path.join(ref_audio_path, wav_files[0])
    return ref_audio_path


class VoiceCloner:
    def __init__(
        self,
        model_name: str = "F5-TTS",
        device: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        vocab_file: Optional[str] = None,
        model_cfg: Optional[dict] = None,
        mel_spec_type: str = "vocos",
        ode_method: str = "euler",
        use_ema: bool = True,
        compile_model: bool = False,
        vocoder_local: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")

        model_cfg = model_cfg or dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
        )

        from f5_tts.infer.utils_infer import load_model, load_vocoder

        if ckpt_path is None:
            ckpt_path = DEFAULT_CKPT_PATH
        if vocab_file is None:
            vocab_file = DEFAULT_VOCAB_PATH

        ckpt_path = _resolve_hf_path(ckpt_path)
        vocab_file = _resolve_hf_path(vocab_file)

        model_cls = _resolve_dit_cls()
        self.model = load_model(
            model_cls=model_cls,
            model_cfg=model_cfg,
            ckpt_path=ckpt_path,
            mel_spec_type=mel_spec_type,
            vocab_file=vocab_file or "",
            ode_method=ode_method,
            use_ema=use_ema,
            device=self.device,
        )
        self.model.eval()

        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                print("torch.compile failed; continuing without compilation.")

        self.vocoder = load_vocoder(is_local=vocoder_local)
        self.mel_spec_type = mel_spec_type
        print("Model loaded successfully.")

    def clone_voice(
        self,
        ref_audio_path: str,
        ref_text: str,
        gen_text: str,
        output_path: str = "output.wav",
        speed: float = 1.0,
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        trim_silence: bool = False,
        min_silence_len: int = 1000,
        silence_thresh: int = -40,
    ) -> str:
        """
        ref_audio_path: Path to the 10-15s WAV file you want to clone.
        ref_text: Transcript of the reference audio (recommended for best quality).
        gen_text: The new text you want the voice to say.
        """
        from f5_tts.infer.utils_infer import infer_process

        ref_audio_path = _resolve_ref_audio_path(ref_audio_path)
        output_path = os.fspath(output_path)

        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

        tmp_path = None
        if not ref_text.strip():
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text

            ref_audio_path, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)
        elif trim_silence:
            tmp_path = _trim_silence_wav(
                ref_audio_path,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
            )
            ref_audio_path = tmp_path

        try:
            sig = inspect.signature(infer_process)
            infer_kwargs = {}
            if "mel_spec_type" in sig.parameters:
                infer_kwargs["mel_spec_type"] = self.mel_spec_type
            if "speed" in sig.parameters:
                infer_kwargs["speed"] = speed
            if "nfe_step" in sig.parameters:
                infer_kwargs["nfe_step"] = nfe_step
            if "cfg_strength" in sig.parameters:
                infer_kwargs["cfg_strength"] = cfg_strength
            if "sway_sampling_coef" in sig.parameters:
                infer_kwargs["sway_sampling_coef"] = sway_sampling_coef

            audio, sample_rate, _ = infer_process(
                ref_audio_path,
                ref_text,
                gen_text,
                self.model,
                self.vocoder,
                **infer_kwargs,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        sf.write(output_path, audio, sample_rate)
        print(f"Saved to {output_path}")
        return output_path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate speech from a reference WAV file or folder using F5-TTS."
    )
    parser.add_argument(
        "--ref",
        required=True,
        help="Path to a reference .wav file or a folder containing one .wav file.",
    )
    parser.add_argument(
        "--ref_text",
        default="",
        help="Transcript of the reference audio. Leave empty to auto-transcribe.",
    )
    parser.add_argument(
        "--gen_text",
        required=True,
        help="Text to synthesize in the reference voice.",
    )
    parser.add_argument(
        "--out",
        default="result.wav",
        help="Output wav path.",
    )
    parser.add_argument(
        "--trim_silence",
        action="store_true",
        help="Trim leading/trailing silence from reference audio.",
    )
    parser.add_argument("--nfe_step", type=int, default=32, help="Denoising steps.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed.")
    parser.add_argument("--cfg_strength", type=float, default=2.0, help="CFG strength.")
    parser.add_argument(
        "--sway_sampling_coef",
        type=float,
        default=-1.0,
        help="Sway sampling coefficient.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster inference (Linux).",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        help="Custom checkpoint path or hf:// URL.",
    )
    parser.add_argument(
        "--vocab_file",
        default=None,
        help="Custom vocab path or hf:// URL.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    engine = VoiceCloner(
        compile_model=args.compile,
        ckpt_path=args.ckpt_path,
        vocab_file=args.vocab_file,
    )
    engine.clone_voice(
        ref_audio_path=args.ref,
        ref_text=args.ref_text,
        gen_text=args.gen_text,
        output_path=args.out,
        trim_silence=args.trim_silence,
        nfe_step=args.nfe_step,
        speed=args.speed,
        cfg_strength=args.cfg_strength,
        sway_sampling_coef=args.sway_sampling_coef,
    )


if __name__ == "__main__":
    main()
