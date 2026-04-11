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
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    condition_on_previous_text: bool = True
    initial_prompt: str | None = None


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
