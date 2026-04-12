"""Pydantic configuration models and YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    mono: bool = True
    denoise: bool = False


class VADConfig(BaseModel):
    engine: Literal["silero", "energy"] = "silero"
    min_speech_duration: float = 0.5
    min_silence_duration: float = 0.3


class LanguageDetectionConfig(BaseModel):
    engine: Literal["whisper", "fixed"] = "whisper"
    fixed_language: str | None = None


class ASRConfig(BaseModel):
    engine: Literal["whisperx", "whisper_hf", "faster_whisper"] = "faster_whisper"
    model: str = "large-v3"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: str = "float16"
    beam_size: int = 10
    best_of: int = 5
    patience: float = 1.0
    # Disabling prev-text conditioning prevents hallucination cascades where one
    # bad segment poisons the next. Critical for dialectal/noisy audio.
    condition_on_previous_text: bool = False
    initial_prompt: str | None = None
    # Decoder fallback / filtering thresholds (passed to faster-whisper)
    temperature: list[float] = Field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8])
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.5
    # Post-decode hallucination filter: drop segments with no_speech_prob above this
    hallucination_no_speech_prob: float = 0.6
    # VAD filter parameters (Silero inside faster-whisper)
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 250
    vad_speech_pad_ms: int = 400
    vad_threshold: float = 0.35


class TranslationConfig(BaseModel):
    engine: Literal["nllb", "opus", "local_llm", "llama_cpp", "none"] = "llama_cpp"
    model: str = "facebook/nllb-200-3.3B"
    source_lang: str = "arz_Arab"
    target_lang: str = "eng_Latn"
    batch_size: int = 8
    # llama_cpp-specific settings
    llama_server_url: str | None = None
    llama_auto_launch: bool = True
    llama_gpu_layers: str = "auto"
    llama_ctx_size: int = 8192


class SubtitleConfig(BaseModel):
    max_chars_per_line: int = 42
    max_cps: int = 17
    max_lines: int = 2
    min_duration: float = 0.8
    max_duration: float = 7.0


class AppConfig(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    language_detection: LanguageDetectionConfig = Field(default_factory=LanguageDetectionConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    subtitle: SubtitleConfig = Field(default_factory=SubtitleConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load configuration from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(**raw)


def resolve_device(device: str) -> str:
    """Resolve 'auto' device to 'cuda' or 'cpu'."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
