"""Abstract base class for language detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from subgen.config import LanguageDetectionConfig


class BaseLangDetector(ABC):
    """Language detection interface."""

    def __init__(self, cfg: LanguageDetectionConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def detect(self, audio_path: str | Path) -> str:
        """Detect the spoken language from an audio file.

        Returns a language code (e.g. 'ar', 'en', 'fr').
        """

    @classmethod
    def from_config(cls, cfg: LanguageDetectionConfig) -> BaseLangDetector:
        """Factory: create the appropriate language detector from config."""
        if cfg.engine == "fixed":
            return FixedLangDetector(cfg)
        if cfg.engine == "whisper":
            from subgen.langdetect.whisper_lid import WhisperLangDetector

            return WhisperLangDetector(cfg)
        raise ValueError(f"Unknown language detection engine: {cfg.engine}")


class FixedLangDetector(BaseLangDetector):
    """Returns a pre-configured fixed language (skips detection)."""

    def detect(self, audio_path: str | Path) -> str:
        if self.cfg.fixed_language is None:
            raise ValueError("fixed_language must be set when engine='fixed'")
        return self.cfg.fixed_language
