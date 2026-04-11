"""Silero VAD implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from subgen.audio.loader import load_audio
from subgen.config import VADConfig
from subgen.models import SpeechRegion
from subgen.vad.base import BaseVAD

logger = logging.getLogger(__name__)

SILERO_SAMPLE_RATE = 16000


class SileroVAD(BaseVAD):
    """Voice Activity Detection using silero-vad."""

    def __init__(self, cfg: VADConfig) -> None:
        super().__init__(cfg)
        self._model, self._utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )

    def detect(self, audio_path: str | Path) -> list[SpeechRegion]:
        audio_path = Path(audio_path)
        logger.info("Running Silero VAD on %s", audio_path)

        wav, _ = load_audio(audio_path, target_sr=SILERO_SAMPLE_RATE)

        get_speech_timestamps = self._utils[0]
        timestamps = get_speech_timestamps(
            wav,
            self._model,
            sampling_rate=SILERO_SAMPLE_RATE,
            min_speech_duration_ms=int(self.cfg.min_speech_duration * 1000),
            min_silence_duration_ms=int(self.cfg.min_silence_duration * 1000),
        )

        regions = [
            SpeechRegion(
                start=ts["start"] / SILERO_SAMPLE_RATE,
                end=ts["end"] / SILERO_SAMPLE_RATE,
            )
            for ts in timestamps
        ]
        logger.info("Silero VAD found %d speech regions", len(regions))
        return regions
