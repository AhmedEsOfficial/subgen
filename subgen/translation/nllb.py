"""NLLB (No Language Left Behind) translator."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from subgen.config import TranslationConfig
from subgen.translation.base import BaseTranslator

logger = logging.getLogger(__name__)


class NLLBTranslator(BaseTranslator):
    """Translation using facebook/nllb-200 models."""

    def __init__(self, cfg: TranslationConfig) -> None:
        super().__init__(cfg)
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        model_id = self.cfg.model
        logger.info("Loading NLLB model: %s", model_id)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang=self.cfg.source_lang)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)
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
                generated = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(
                        self.cfg.target_lang
                    ),
                    max_new_tokens=256,
                )

            decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            results.extend(decoded)

        return results
