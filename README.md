# subgen

Modular subtitle generation pipeline for video files. Transcribes speech, translates across languages, and emits standards-compliant `.srt` files. Built for Egyptian Arabic film subtitling but works with any language Whisper supports.

## How it works

```
Video file
  |
  v
[Audio Extraction] ── ffmpeg ── mono 16kHz WAV
  |
  v
[Voice Activity Detection] ── silero-vad / energy-based
  |
  v
[Language Detection] ── Whisper language ID / fixed
  |
  v
[ASR Transcription] ── faster-whisper / WhisperX / HF Whisper
  |
  v
[Translation] ── llama.cpp (local LLM) / NLLB / Opus-MT / none
  |
  v
[Subtitle Formatting] ── line wrap, CPS enforcement, merge/split
  |
  v
output.srt
```

Every stage is swappable via config file or CLI flags. No code changes needed to switch models.

## Quick start

```bash
# Clone and install
git clone https://github.com/youruser/subtitles-fall.git
cd subtitles-fall
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Install ML dependencies
pip install torch torchaudio                  # or with CUDA: --index-url https://download.pytorch.org/whl/cu126
pip install faster-whisper transformers accelerate sentencepiece soundfile numpy

# Basic usage — transcribe only (no translation)
subgen generate video.mp4 --translator none

# Egyptian Arabic → English with local LLM translation
subgen generate film.mkv -s ar -t en --translator llama_cpp --translator-model /path/to/model.gguf

# Using a config file
subgen generate film.mkv -c config.example.yaml -o output.srt
```

## Requirements

- Python 3.11+
- ffmpeg (system install)
- CUDA GPU recommended (CPU works but is slow)
- For `llama_cpp` translation: [llama.cpp](https://github.com/ggerganov/llama.cpp) with `llama-server` on PATH

### Python dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchaudio` | ML runtime |
| `transformers` | HuggingFace model loading |
| `faster-whisper` | Default ASR engine (CTranslate2) |
| `ffmpeg-python` | Audio extraction from video |
| `soundfile` | Audio file loading |
| `pysrt` | SRT file writing |
| `pydantic` | Configuration validation |
| `pyyaml` | YAML config parsing |
| `typer` | CLI framework |

Optional:

| Package | Purpose |
|---------|---------|
| `whisperx` | WhisperX ASR engine with word-level alignment |
| `demucs` | Audio denoising (vocal isolation) |
| `sentencepiece`, `accelerate` | Required by some translation models |

## CLI reference

### `subgen generate`

```
subgen generate INPUT_FILE [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `{input}.srt` | Output SRT file path |
| `--config` | `-c` | — | YAML config file |
| `--source-lang` | `-s` | auto-detect | Source language code (e.g. `ar`) |
| `--target-lang` | `-t` | `eng_Latn` | Target language code |
| `--asr` | | `faster_whisper` | ASR engine: `faster_whisper`, `whisperx`, `whisper_hf` |
| `--asr-model` | | `large-v3` | ASR model name or HuggingFace ID |
| `--translator` | | `llama_cpp` | Translation engine: `llama_cpp`, `nllb`, `opus`, `local_llm`, `none` |
| `--translator-model` | | — | Translation model path or HuggingFace ID |
| `--llama-url` | | auto-launch | URL of running llama-server |
| `--initial-prompt` | | — | ASR initial prompt hint (improves dialect accuracy) |
| `--denoise` | | off | Enable Demucs vocal isolation |
| `--device` | | `auto` | `cuda`, `cpu`, or `auto` |
| `--verbose` | `-v` | off | Debug logging |

### `subgen version`

Prints the installed version.

## Configuration

All options can be set via YAML config file. CLI flags override config values. See [config.example.yaml](config.example.yaml) for a complete template.

```yaml
audio:
  sample_rate: 16000
  mono: true
  denoise: false

vad:
  engine: silero          # silero | energy
  min_speech_duration: 0.5
  min_silence_duration: 0.3

language_detection:
  engine: fixed           # whisper | fixed
  fixed_language: ar

asr:
  engine: faster_whisper  # faster_whisper | whisperx | whisper_hf
  model: large-v3
  device: auto
  compute_type: float16
  beam_size: 5
  best_of: 5
  patience: 1.0
  condition_on_previous_text: true
  initial_prompt: null

translation:
  engine: llama_cpp       # llama_cpp | nllb | opus | local_llm | none
  model: /path/to/model.gguf
  source_lang: arz_Arab
  target_lang: eng_Latn
  batch_size: 10
  llama_server_url: null  # null = auto-launch
  llama_auto_launch: true
  llama_gpu_layers: auto
  llama_ctx_size: 8192

subtitle:
  max_chars_per_line: 42
  max_cps: 17
  max_lines: 2
  min_duration: 0.8
  max_duration: 7.0
```

## Pipeline stages

### Audio extraction

Uses ffmpeg to demux video into mono 16kHz WAV. Supports any format ffmpeg can read (`.mkv`, `.mp4`, `.avi`, `.mov`, etc.).

Optional denoising via [Demucs](https://github.com/facebookresearch/demucs) — isolates vocals from music/noise. Recommended for old films with background hiss.

### Voice Activity Detection

Identifies speech regions before ASR to avoid hallucinations on silence/music.

| Engine | Description | Dependencies |
|--------|-------------|--------------|
| `silero` | Neural VAD via silero-vad. High accuracy. | `torch` |
| `energy` | RMS energy thresholding. No ML, fast. | `torch` |

### Language detection

| Engine | Description |
|--------|-------------|
| `whisper` | Uses Whisper's built-in language ID on the first 30 seconds. |
| `fixed` | Returns a hardcoded language code. Use `-s ar` on the CLI. |

### ASR (transcription)

All engines produce word-level timestamps.

| Engine | Backend | Best for |
|--------|---------|----------|
| `faster_whisper` | CTranslate2 | Default. Fast, GPU/CPU, built-in VAD filter. |
| `whisperx` | Whisper + wav2vec2 alignment | Best word-level accuracy. Requires `pip install whisperx`. |
| `whisper_hf` | HuggingFace transformers | Access to any HF Whisper model variant. |

The `faster_whisper` engine includes a hallucination filter that drops segments with `no_speech_prob > 0.6`.

**ASR tuning tips:**
- Use `beam_size=5` and `best_of=5` for best accuracy (default).
- Set `--initial-prompt` with sample text in the target dialect to bias the model.
- For Egyptian Arabic, language code `ar` works — Whisper doesn't distinguish dialects.

### Translation

| Engine | Description | Speed | Quality (EG-AR) |
|--------|-------------|-------|------------------|
| `llama_cpp` | Local LLM via llama-server. Any GGUF model. | Slow (thinking models) | Best |
| `nllb` | facebook/nllb-200-3.3B seq2seq | Fast | Good (MSA) |
| `opus` | Helsinki-NLP/opus-mt-ar-en | Fast | Fair (MSA only) |
| `local_llm` | Any HuggingFace causal LM | Varies | Good |
| `none` | Skip translation | — | — |

**`llama_cpp` features:**
- Auto-launches `llama-server` if not already running, shuts it down when done
- Batch translation — sends 10 lines per API call with numbered formatting
- Arabic detection fallback — if a translated line is still >30% Arabic script, retries individually
- Retry logic with 600s timeout per request (supports thinking models)
- Works with any GGUF model: Gemma, Qwen, Llama, Mistral, etc.

**Pre-launching the server** (recommended for repeated runs):

```bash
llama-server -m /path/to/model.gguf -ngl auto --port 8081 -c 8192
# Then:
subgen generate film.mp4 -s ar -t en --llama-url http://127.0.0.1:8081
```

### Subtitle formatting

- Line wrapping at configurable character limit (default 42)
- CPS (characters per second) enforcement (default max 17)
- Splits long segments using word-level timestamps when available
- Merges consecutive short segments
- Minimum/maximum duration enforcement

## Architecture

```
subgen/
  config.py          Pydantic models + YAML loader
  models.py          Shared dataclasses (TimedSegment, SubtitleEntry, etc.)
  orchestrator.py    Pipeline class — connects all stages
  cli.py             Typer CLI entry point

  audio/
    extractor.py     ffmpeg wrapper
    loader.py        soundfile-based audio loading
    denoiser.py      Demucs vocal isolation

  vad/
    base.py          BaseVAD ABC + factory
    silero.py        Silero VAD
    energy.py        Energy-based VAD

  langdetect/
    base.py          BaseLangDetector ABC + factory + FixedLangDetector
    whisper_lid.py   Whisper language identification

  asr/
    base.py          BaseASR ABC + factory
    faster_whisper.py
    whisperx_engine.py
    whisper_hf.py

  translation/
    base.py          BaseTranslator ABC + factory
    llama_cpp.py     llama-server integration
    nllb.py          NLLB-200 seq2seq
    opus.py          Opus-MT Marian
    local_llm.py     HuggingFace causal LMs

  subtitle/
    formatter.py     Line wrap, CPS, merge/split
    writer.py        SRT output via pysrt
```

Every pipeline component implements an abstract base class with a `from_config()` factory method. Adding a new engine means:
1. Create a new class inheriting from the stage's ABC
2. Add its engine name to the config Literal type
3. Add a branch in the factory method

## Examples

```bash
# Transcribe only (Arabic source, no translation)
subgen generate film.mp4 -s ar --translator none -o arabic_transcript.srt

# Egyptian Arabic → English with Gemma
subgen generate film.mkv -s ar -t en \
  --translator llama_cpp \
  --translator-model /path/to/gemma.gguf \
  -o english_subs.srt

# With denoising for old/noisy audio
subgen generate old_film.avi -s ar -t en --denoise -o clean_subs.srt

# Use NLLB for fast (lower quality) translation
subgen generate film.mp4 -s ar -t en \
  --translator nllb \
  --translator-model facebook/nllb-200-distilled-600M

# CPU-only mode
subgen generate film.mp4 -s ar --device cpu --translator none

# Custom ASR model
subgen generate film.mp4 --asr-model openai/whisper-large-v3 -s ar

# Pre-launched llama-server
subgen generate film.mp4 -s ar -t en --llama-url http://localhost:8080

# Full config file
subgen generate film.mp4 -c my_config.yaml
```

## Performance

Tested on a 42-minute Egyptian Arabic film (RTX 3070 Laptop GPU, 8GB VRAM):

| Stage | Time | Notes |
|-------|------|-------|
| Audio extraction | 3s | ffmpeg |
| VAD | 24s | Silero, 413 speech regions |
| ASR | 3.5 min | faster-whisper large-v3, fp16, 508 segments |
| Translation | ~55 min | Gemma-4 E4B thinking model, 51 batches |
| Formatting | instant | |
| **Total** | **~58 min** | With thinking model translation |

Translation time depends heavily on the model. A non-thinking model (e.g. Qwen-2.5-7B) would be 5-10x faster. Using `--translator none` for transcription-only takes under 4 minutes.

## License

MIT
