# Staccato: Stuttered Speech Recognition

Staccato is a speech recognition pipeline optimized for transcribing stuttered speech. It combines OpenAI's Whisper model with GPT-4o to produce accurate transcriptions that capture the speaker's intended message, filtering out involuntary disfluencies.

## Overview

People who stutter speak with involuntary sound repetitions, word repetitions, prolongations, and blocks. Standard ASR models like Whisper are not trained on stuttered speech data and often produce poor transcriptions. Staccato addresses this by:

1. **Whisper transcription**: Initial transcription using Whisper Large V3
2. **GPT-4o refinement**: Uses GPT-4o with audio understanding to refine the transcription, leveraging both the audio and initial transcription

## Installation

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA support (CUDA 11.8+)
- CUDA toolkit installed (`nvcc` must be in PATH)
- Linux (tested on Ubuntu)
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/staccato.git
   cd staccato
   ```

2. Install uv:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Restart your terminal** to ensure `uv` is in your PATH.

4. Install dependencies:
   ```bash
   uv sync
   ```

5. Fix flash-attn (required for Flash Attention 2 support):
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export CUDA_HOME=/usr/local/cuda
   rm -rf .venv/lib/python3.12/site-packages/flash_attn*
   uv cache clean flash-attn
   uv pip install flash-attn --no-build-isolation --no-binary flash-attn
   ```
   
   > **Note:** This compilation takes 5-10 minutes and requires the CUDA toolkit. If `nvcc --version` doesn't work, install CUDA toolkit first.

6. Download Whisper model:
   ```bash
   mkdir -p ./data/models
   uv run huggingface-cli download --local-dir ./data/models/whisper-large-v3 openai/whisper-large-v3
   ```

7. Create a `.env` file with your OpenAI API key:
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

## Usage

### Command Line

Transcribe audio files from the terminal:

```bash
# With your own audio file
uv run python src/approaches/dspy_whisper_gpt_4o/pipeline.py \
  --audio-files /path/to/your/audio.wav

# If you downloaded sample data (multiple files)
uv run python src/approaches/dspy_whisper_gpt_4o/pipeline.py \
  --audio-files data/processed/fluencybank/wav_clips/27fb_086_000.wav \
               data/processed/fluencybank/wav_clips/44m_381_000.wav

# With custom model directory
uv run python src/approaches/dspy_whisper_gpt_4o/pipeline.py \
  --model-dir ./data/models/whisper-large-v3 \
  --audio-files /path/to/your/audio.wav
```

### Python API

```python
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, 'src')

import librosa
from approaches.dspy_whisper_gpt_4o.pipeline import DspyPipeline

# Initialize pipeline
pipe = DspyPipeline(
    model_dir=Path("./data/models/whisper-large-v3"),
)
pipe.load_model()

# Load audio
audio, sr = librosa.load("path/to/audio.wav")

# Transcribe
result = pipe.transcribe([audio], sr)
print(result)
```

### Offline evaluation (no GPU or API key)

Compare saved baseline and refined transcripts using corpus word error rate,
per-example regressions, and optional paired latency/cost measurements:

```bash
uv run --script src/evaluation/compare_transcripts.py examples/synthetic_transcripts.jsonl --reference-kind intended
```

The bundled examples and measurements are **synthetic**, not model results.
See [Offline evaluation](docs/offline-evaluation.md) for the input contract,
reference-task distinctions, limitations, and lightweight test instructions.

## Project Structure

```
src/
├── approaches/
│   ├── dspy_whisper_gpt_4o/
│   │   └── pipeline.py          # Main pipeline (Whisper + GPT-4o)
│   └── vanilla_whisper/
│       └── whisper.py           # WhisperTranscriber wrapper
└── evaluation/
    └── inference_models/
        └── asr_model_base.py    # Base class for ASR models
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
