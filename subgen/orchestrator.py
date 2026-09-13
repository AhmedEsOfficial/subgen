"""Pipeline orchestrator — connects all stages together."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from subgen.asr.base import BaseASR
from subgen.audio.denoiser import denoise
from subgen.audio.extractor import extract_audio
from subgen.config import AppConfig
from subgen.langdetect.base import BaseLangDetector
from subgen.models import SubtitleEntry, TimedSegment
from subgen.subtitle.formatter import format_subtitles
from subgen.subtitle.writer import write_srt
from subgen.translation.base import BaseTranslator
from subgen.vad.base import BaseVAD

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end subtitle generation pipeline."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._vad: BaseVAD | None = None
        self._lang_detector: BaseLangDetector | None = None
        self._asr: BaseASR | None = None
        self._translator: BaseTranslator | None = None

    def _init_components(self) -> None:
        """Lazily initialize pipeline components from config."""
        if self._vad is None:
            self._vad = BaseVAD.from_config(self.cfg.vad)
        if self._lang_detector is None:
            self._lang_detector = BaseLangDetector.from_config(self.cfg.language_detection)
        if self._asr is None:
            self._asr = BaseASR.from_config(self.cfg.asr)
        if self._translator is None:
            self._translator = BaseTranslator.from_config(self.cfg.translation)

    def run(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> Path:
        """Run the full subtitle generation pipeline.

        Parameters
        ----------
        video_path:
            Input video file.
        output_path:
            Output SRT path.  Defaults to ``<video_stem>.srt``.
        source_lang:
            Override source language (skip auto-detection).
        target_lang:
            Override target language for translation.

        Returns the path to the generated SRT file.
        """
        video_path = Path(video_path)
        if output_path is None:
            output_path = video_path.with_suffix(".srt")
        else:
            output_path = Path(output_path)

        self._init_components()
        assert self._vad and self._lang_detector and self._asr

        audio_path: Path | None = None
        try:
            # --- Stage 1: Audio extraction ---
            logger.info("=== Stage 1: Audio extraction ===")
            audio_path = extract_audio(video_path, self.cfg.audio)

            # --- Stage 1b: Optional denoising ---
            if self.cfg.audio.denoise:
                logger.info("=== Stage 1b: Denoising ===")
                audio_path = denoise(audio_path)

            # --- Stage 2: Voice Activity Detection ---
            logger.info("=== Stage 2: VAD ===")
            regions = self._vad.detect(audio_path)
            logger.info("Found %d speech regions", len(regions))
            if not regions:
                logger.warning("VAD found no speech — ASR will process the full audio")

            # --- Stage 3: Language detection ---
            logger.info("=== Stage 3: Language detection ===")
            if source_lang:
                detected_lang = source_lang
                logger.info("Using provided source language: %s", detected_lang)
            else:
                detected_lang = self._lang_detector.detect(audio_path)
                logger.info("Detected language: %s", detected_lang)

            # --- Stage 4: ASR ---
            logger.info("=== Stage 4: ASR (transcription) ===")
            segments: list[TimedSegment] = self._asr.transcribe(
                audio_path, language=detected_lang, regions=regions
            )
            logger.info("Transcribed %d segments", len(segments))

            # --- Stage 5: Translation ---
            if self._translator is not None and self.cfg.translation.engine != "none":
                logger.info(
                    "=== Stage 5: Translation (%s → %s, engine=%s) ===",
                    detected_lang,
                    target_lang or self.cfg.translation.target_lang,
                    self.cfg.translation.engine,
                )
                texts = [seg.text for seg in segments]
                translated = self._translator.translate_batch(texts)
                for seg, trans in zip(segments, translated):
                    seg.text = trans
                    # Word timings describe the source text; keeping them would make
                    # the formatter rebuild cues out of untranslated words.
                    seg.words = []
                logger.info("Translated %d segments", len(translated))
            else:
                logger.info("=== Stage 5: Translation skipped ===")

            # --- Stage 6: Subtitle formatting & writing ---
            logger.info("=== Stage 6: Formatting & writing SRT ===")
            entries: list[SubtitleEntry] = format_subtitles(segments, self.cfg.subtitle)
            result_path = write_srt(entries, output_path)

            logger.info("Pipeline complete: %s → %s (%d subtitles)", video_path, result_path, len(entries))
            return result_path

        finally:
            # Clean up temporary audio file
            if audio_path and audio_path.exists() and "/tmp" in str(audio_path):
                try:
                    os.unlink(audio_path)
                    logger.debug("Cleaned up temp file: %s", audio_path)
                except OSError:
                    pass
            # Shut down auto-launched llama-server if applicable
            if self._translator is not None and hasattr(self._translator, "shutdown"):
                self._translator.shutdown()
