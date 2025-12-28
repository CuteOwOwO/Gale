# TTS_model (F5-TTS)

This folder contains a small wrapper around F5-TTS for zero-shot voice cloning.

## Setup
1. Create a clean environment:
   - `conda create -n f5tts python=3.10`
   - `conda activate f5tts`
2. Install ffmpeg:
   - Linux/WSL: `sudo apt update && sudo apt install ffmpeg`
   - Conda: `conda install -c conda-forge ffmpeg`
3. Install PyTorch with CUDA from https://pytorch.org.
4. Clone F5-TTS inside this folder and install in editable mode:
   - `git clone https://github.com/SWivid/F5-TTS.git TTS_model/F5-TTS`
   - `pip install -e TTS_model/F5-TTS`
5. Install local dependencies:
   - `pip install -r TTS_model/requirements.txt`

## Usage
Run from the CLI (file or folder):
```bash
python TTS_model/tts_engine.py \
  --ref TTS_model/ref_audio \
  --ref_text "The quick brown fox jumps over the lazy dog." \
  --gen_text "Hello, this is a cloned voice." \
  --out result.wav
```

Or import it:
```python
from TTS_model import VoiceCloner

engine = VoiceCloner()
engine.clone_voice(
    ref_audio_path="my_voice_sample.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Hello, this is a cloned voice.",
    output_path="result.wav",
    trim_silence=True,
)
```

## Tips
- Always provide `ref_text` for best quality.
- Lower `nfe_step` (for example, 16) to speed up generation.
- Use `trim_silence=True` if your reference audio has leading silence.
- On Linux, pass `compile_model=True` to `VoiceCloner` for a speed boost.
- The default checkpoint and vocab are pulled from Hugging Face; pass `ckpt_path` or `vocab_file` to override.
- If `ref_text` is empty, the model will auto-transcribe (downloads Whisper on first run).
