"""Simple energy-based VAD fallback (no ML dependencies)."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from subgen.audio.loader import load_audio
from subgen.config import VADConfig
from subgen.models import SpeechRegion
from subgen.vad.base import BaseVAD

logger = logging.getLogger(__name__)


class EnergyVAD(BaseVAD):
    """Basic RMS energy thresholding VAD."""

    def __init__(self, cfg: VADConfig) -> None:
        super().__init__(cfg)
        self.frame_duration = 0.03  # 30ms frames
        self.energy_threshold = 0.01  # relative RMS threshold

    def detect(self, audio_path: str | Path) -> list[SpeechRegion]:
        audio_path = Path(audio_path)
        logger.info("Running energy-based VAD on %s", audio_path)

        wav, sr = load_audio(audio_path)

        frame_size = int(sr * self.frame_duration)
        num_frames = len(wav) // frame_size

        is_speech = []
        for i in range(num_frames):
            frame = wav[i * frame_size : (i + 1) * frame_size]
            rms = torch.sqrt(torch.mean(frame**2)).item()
            is_speech.append(rms > self.energy_threshold)

        regions: list[SpeechRegion] = []
        in_speech = False
        start = 0.0

        for i, speech in enumerate(is_speech):
            t = i * self.frame_duration
            if speech and not in_speech:
                start = t
                in_speech = True
            elif not speech and in_speech:
                duration = t - start
                if duration >= self.cfg.min_speech_duration:
                    regions.append(SpeechRegion(start=start, end=t))
                in_speech = False

        if in_speech:
            end_t = num_frames * self.frame_duration
            if end_t - start >= self.cfg.min_speech_duration:
                regions.append(SpeechRegion(start=start, end=end_t))

        # Merge regions that are separated by less than min_silence_duration
        merged: list[SpeechRegion] = []
        for region in regions:
            if merged and (region.start - merged[-1].end) < self.cfg.min_silence_duration:
                merged[-1] = SpeechRegion(start=merged[-1].start, end=region.end)
            else:
                merged.append(region)

        logger.info("Energy VAD found %d speech regions", len(merged))
        return merged
