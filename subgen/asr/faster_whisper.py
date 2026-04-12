"""faster-whisper ASR engine (CTranslate2, CPU-friendly)."""

from __future__ import annotations

import logging
from pathlib import Path

from subgen.asr.base import BaseASR
from subgen.config import ASRConfig, resolve_device
from subgen.models import SpeechRegion, TimedSegment, TimedWord

logger = logging.getLogger(__name__)


class FasterWhisperASR(BaseASR):
    """ASR using faster-whisper (CTranslate2 backend)."""

    def __init__(self, cfg: ASRConfig) -> None:
        super().__init__(cfg)
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        device = resolve_device(self.cfg.device)
        compute_type = self.cfg.compute_type
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"

        model_size = self.cfg.model
        logger.info(
            "Loading faster-whisper model=%s device=%s compute=%s",
            model_size, device, compute_type,
        )
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        regions: list[SpeechRegion] | None = None,
    ) -> list[TimedSegment]:
        self._load()
        assert self._model is not None
        logger.info("faster-whisper transcribing %s (lang=%s)", audio_path, language)

        transcribe_kwargs: dict = dict(
            language=language,
            word_timestamps=True,
            beam_size=self.cfg.beam_size,
            best_of=self.cfg.best_of,
            patience=self.cfg.patience,
            condition_on_previous_text=self.cfg.condition_on_previous_text,
            temperature=tuple(self.cfg.temperature),
            compression_ratio_threshold=self.cfg.compression_ratio_threshold,
            log_prob_threshold=self.cfg.log_prob_threshold,
            no_speech_threshold=self.cfg.no_speech_threshold,
            vad_filter=True,
            vad_parameters=dict(
                min_speech_duration_ms=self.cfg.vad_min_speech_ms,
                min_silence_duration_ms=self.cfg.vad_min_silence_ms,
                speech_pad_ms=self.cfg.vad_speech_pad_ms,
                threshold=self.cfg.vad_threshold,
            ),
        )
        if self.cfg.initial_prompt:
            transcribe_kwargs["initial_prompt"] = self.cfg.initial_prompt

        raw_segments, info = self._model.transcribe(str(audio_path), **transcribe_kwargs)

        logger.info(
            "Detected language: %s (prob=%.2f)", info.language, info.language_probability
        )

        segments: list[TimedSegment] = []
        for seg in raw_segments:
            text = seg.text.strip()
            # Skip hallucinated / repeated filler segments
            if not text or seg.no_speech_prob > self.cfg.hallucination_no_speech_prob:
                continue
            words = [
                TimedWord(start=w.start, end=w.end, word=w.word.strip())
                for w in (seg.words or [])
            ]
            segments.append(
                TimedSegment(
                    start=seg.start,
                    end=seg.end,
                    text=text,
                    language=info.language,
                    words=words,
                )
            )

        logger.info("faster-whisper produced %d segments", len(segments))
        return segments
