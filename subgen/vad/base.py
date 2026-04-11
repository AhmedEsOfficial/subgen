"""Abstract base class for VAD engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from subgen.config import VADConfig
from subgen.models import SpeechRegion


class BaseVAD(ABC):
    """Voice Activity Detection interface."""

    def __init__(self, cfg: VADConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def detect(self, audio_path: str | Path) -> list[SpeechRegion]:
        """Return speech regions detected in the audio file."""

    @classmethod
    def from_config(cls, cfg: VADConfig) -> BaseVAD:
        """Factory: create the appropriate VAD engine from config."""
        if cfg.engine == "silero":
            from subgen.vad.silero import SileroVAD

            return SileroVAD(cfg)
        if cfg.engine == "energy":
            from subgen.vad.energy import EnergyVAD

            return EnergyVAD(cfg)
        raise ValueError(f"Unknown VAD engine: {cfg.engine}")
