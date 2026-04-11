"""HuggingFace Transformers Whisper ASR engine."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from subgen.asr.base import BaseASR
from subgen.audio.loader import load_audio
from subgen.config import ASRConfig, resolve_device
from subgen.models import SpeechRegion, TimedSegment, TimedWord

logger = logging.getLogger(__name__)

HF_SAMPLE_RATE = 16000


class WhisperHFASR(BaseASR):
    """ASR using HuggingFace transformers Whisper pipeline."""

    def __init__(self, cfg: ASRConfig) -> None:
        super().__init__(cfg)
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        from transformers import pipeline

        device = resolve_device(self.cfg.device)
        model_id = self.cfg.model
        # Resolve short names to full HF model IDs
        if "/" not in model_id:
            model_id = f"openai/whisper-{model_id}"

        logger.info("Loading HF Whisper pipeline: %s on %s", model_id, device)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=dtype,
            device=device,
        )

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        regions: list[SpeechRegion] | None = None,
    ) -> list[TimedSegment]:
        self._load()
        assert self._pipe is not None
        logger.info("HF Whisper transcribing %s (lang=%s)", audio_path, language)

        wav, _ = load_audio(audio_path, target_sr=HF_SAMPLE_RATE)
        audio_array = wav.numpy()

        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language

        result = self._pipe(
            audio_array,
            return_timestamps="word",
            generate_kwargs=generate_kwargs,
        )

        segments: list[TimedSegment] = []
        current_words: list[TimedWord] = []
        seg_start: float | None = None

        for chunk in result.get("chunks", []):
            ts = chunk.get("timestamp")
            if ts is None or ts[0] is None or ts[1] is None:
                continue
            word = TimedWord(start=ts[0], end=ts[1], word=chunk["text"].strip())
            if seg_start is None:
                seg_start = word.start

            # Split on sentence-ending punctuation or long pauses
            current_words.append(word)
            is_sentence_end = word.word.rstrip().endswith((".", "?", "!", "،", "。"))
            gap = (
                current_words[-1].end - current_words[-2].end
                if len(current_words) > 1
                else 0
            )

            if is_sentence_end or gap > 1.5 or len(current_words) >= 30:
                text = " ".join(w.word for w in current_words)
                segments.append(
                    TimedSegment(
                        start=seg_start,
                        end=current_words[-1].end,
                        text=text,
                        language=language,
                        words=list(current_words),
                    )
                )
                current_words = []
                seg_start = None

        # Flush remaining words
        if current_words and seg_start is not None:
            text = " ".join(w.word for w in current_words)
            segments.append(
                TimedSegment(
                    start=seg_start,
                    end=current_words[-1].end,
                    text=text,
                    language=language,
                    words=list(current_words),
                )
            )

        logger.info("HF Whisper produced %d segments", len(segments))
        return segments
