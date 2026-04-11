"""Abstract base class for ASR engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from subgen.config import ASRConfig
from subgen.models import SpeechRegion, TimedSegment


class BaseASR(ABC):
    """ASR engine interface."""

    def __init__(self, cfg: ASRConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        regions: list[SpeechRegion] | None = None,
    ) -> list[TimedSegment]:
        """Transcribe audio and return timed segments.

        Parameters
        ----------
        audio_path:
            Path to the WAV audio file.
        language:
            Language code hint (e.g. 'ar', 'en').  None = auto-detect.
        regions:
            Optional VAD regions to constrain transcription.
        """

    @classmethod
    def from_config(cls, cfg: ASRConfig) -> BaseASR:
        """Factory: create the appropriate ASR engine from config."""
        if cfg.engine == "whisperx":
            from subgen.asr.whisperx_engine import WhisperXASR

            return WhisperXASR(cfg)
        if cfg.engine == "whisper_hf":
            from subgen.asr.whisper_hf import WhisperHFASR

            return WhisperHFASR(cfg)
        if cfg.engine == "faster_whisper":
            from subgen.asr.faster_whisper import FasterWhisperASR

            return FasterWhisperASR(cfg)
        raise ValueError(f"Unknown ASR engine: {cfg.engine}")
