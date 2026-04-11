"""Whisper-based language identification."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from subgen.audio.loader import load_audio
from subgen.config import LanguageDetectionConfig
from subgen.langdetect.base import BaseLangDetector

logger = logging.getLogger(__name__)

LID_MODEL = "openai/whisper-large-v3-turbo"
LID_SAMPLE_RATE = 16000
LID_DURATION_SEC = 30


class WhisperLangDetector(BaseLangDetector):
    """Detect language using Whisper's built-in language identification."""

    def __init__(self, cfg: LanguageDetectionConfig) -> None:
        super().__init__(cfg)
        self._processor: WhisperProcessor | None = None
        self._model: WhisperForConditionalGeneration | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading Whisper model for language detection: %s", LID_MODEL)
        self._processor = WhisperProcessor.from_pretrained(LID_MODEL)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._model = WhisperForConditionalGeneration.from_pretrained(
            LID_MODEL, torch_dtype=dtype
        ).to(device)
        self._dtype = dtype

    def detect(self, audio_path: str | Path) -> str:
        self._load()
        assert self._processor is not None and self._model is not None

        wav, _ = load_audio(audio_path, target_sr=LID_SAMPLE_RATE)

        # Use only the first 30 seconds
        max_samples = LID_DURATION_SEC * LID_SAMPLE_RATE
        wav = wav[:max_samples]

        inputs = self._processor(
            wav.numpy(), sampling_rate=LID_SAMPLE_RATE, return_tensors="pt"
        )
        input_features = inputs.input_features.to(device=self._model.device, dtype=self._dtype)

        with torch.no_grad():
            predicted_ids = self._model.generate(input_features, max_new_tokens=1)

        # Decode the first token which contains the language tag
        predicted_lang = self._processor.batch_decode(predicted_ids, skip_special_tokens=False)[0]
        # Extract language token like <|ar|>
        for token in predicted_lang.split("|"):
            token = token.strip("<>")
            if len(token) == 2 and token.isalpha():
                logger.info("Detected language: %s", token)
                return token

        # Fallback: use detect_language API
        lang_id = self._model.detect_language(input_features)[0]
        if isinstance(lang_id, dict):
            lang = max(lang_id, key=lang_id.get)  # type: ignore[arg-type]
        else:
            lang = str(lang_id)
        lang = lang.strip("<|>")
        logger.info("Detected language (fallback): %s", lang)
        return lang
