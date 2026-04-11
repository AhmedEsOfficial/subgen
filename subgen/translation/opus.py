"""Helsinki-NLP/Opus-MT translator."""

from __future__ import annotations

import logging

import torch
from transformers import MarianMTModel, MarianTokenizer

from subgen.config import TranslationConfig
from subgen.translation.base import BaseTranslator

logger = logging.getLogger(__name__)

# Common Opus-MT model mapping (source → target)
OPUS_MODEL_MAP = {
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
}


class OpusMTTranslator(BaseTranslator):
    """Translation using Helsinki-NLP/Opus-MT Marian models."""

    def __init__(self, cfg: TranslationConfig) -> None:
        super().__init__(cfg)
        self._tokenizer = None
        self._model = None

    def _resolve_model_id(self) -> str:
        """Resolve the Opus-MT model ID from source/target language config."""
        # If the user specified a full model path, use it
        if "/" in self.cfg.model and "opus" in self.cfg.model.lower():
            return self.cfg.model

        # Try to map NLLB-style language codes to short codes
        src = self.cfg.source_lang.split("_")[0][:2].lower()  # arz_Arab → ar
        tgt = self.cfg.target_lang.split("_")[0][:2].lower()  # eng_Latn → en

        key = (src, tgt)
        if key in OPUS_MODEL_MAP:
            return OPUS_MODEL_MAP[key]

        # Fallback: construct the model name
        return f"Helsinki-NLP/opus-mt-{src}-{tgt}"

    def _load(self):
        if self._model is not None:
            return
        model_id = self._resolve_model_id()
        logger.info("Loading Opus-MT model: %s", model_id)

        self._tokenizer = MarianTokenizer.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = MarianMTModel.from_pretrained(model_id).to(device)
        self._device = device

    def translate(self, text: str) -> str:
        return self.translate_batch([text])[0]

    def translate_batch(self, texts: list[str]) -> list[str]:
        self._load()
        assert self._tokenizer is not None and self._model is not None

        results: list[str] = []
        batch_size = self.cfg.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self._device)

            with torch.no_grad():
                generated = self._model.generate(**inputs, max_length=512)

            decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            results.extend(decoded)

        return results
