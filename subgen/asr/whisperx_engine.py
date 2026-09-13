"""WhisperX ASR engine with word-level forced alignment."""

from __future__ import annotations

import logging
from pathlib import Path

from subgen.asr.base import BaseASR
from subgen.config import ASRConfig, resolve_device
from subgen.models import SpeechRegion, TimedSegment, TimedWord

logger = logging.getLogger(__name__)


class WhisperXASR(BaseASR):
    """ASR using WhisperX — Whisper + wav2vec2 forced alignment."""

    def __init__(self, cfg: ASRConfig) -> None:
        super().__init__(cfg)
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import whisperx

        device = resolve_device(self.cfg.device)
        compute_type = self.cfg.compute_type if device == "cuda" else "int8"
        logger.info(
            "Loading WhisperX model=%s device=%s compute=%s",
            self.cfg.model, device, compute_type,
        )
        self._model = whisperx.load_model(
            self.cfg.model,
            device=device,
            compute_type=compute_type,
        )
        self._device = device

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        regions: list[SpeechRegion] | None = None,
    ) -> list[TimedSegment]:
        import whisperx

        self._load()
        audio_path = str(audio_path)
        logger.info("WhisperX transcribing %s (lang=%s)", audio_path, language)

        audio = whisperx.load_audio(audio_path)
        result = self._model.transcribe(audio, language=language, batch_size=16)

        # Forced alignment for word-level timestamps
        align_model, align_metadata = whisperx.load_align_model(
            language_code=result.get("language", language or "en"),
            device=self._device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            self._device,
        )

        segments: list[TimedSegment] = []
        for seg in result.get("segments", []):
            words = [
                TimedWord(start=w["start"], end=w["end"], word=w["word"], probability=w.get("score"))
                for w in seg.get("words", [])
                if "start" in w and "end" in w
            ]
            segments.append(
                TimedSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg.get("text", "").strip(),
                    language=language,
                    words=words,
                )
            )

        logger.info("WhisperX produced %d segments", len(segments))
        return segments
