"""Local LLM translator using causal language models (Qwen, Gemma, etc.)."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from subgen.config import TranslationConfig
from subgen.translation.base import BaseTranslator

logger = logging.getLogger(__name__)

# Language name mapping for prompts
LANG_NAMES = {
    "ar": "Arabic",
    "arz_Arab": "Egyptian Arabic",
    "arb_Arab": "Arabic",
    "en": "English",
    "eng_Latn": "English",
    "fr": "French",
    "fra_Latn": "French",
    "es": "Spanish",
    "spa_Latn": "Spanish",
}

TRANSLATION_PROMPT = (
    "Translate the following {source_lang} dialogue into natural, idiomatic {target_lang}. "
    "This is from a classic film — preserve tone and register. "
    "Output ONLY the translation, nothing else.\n\n"
    "{source_lang}: {text}\n{target_lang}:"
)


class LocalLLMTranslator(BaseTranslator):
    """Translation using a local causal LM with prompted generation."""

    def __init__(self, cfg: TranslationConfig) -> None:
        super().__init__(cfg)
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        model_id = self.cfg.model
        logger.info("Loading local LLM for translation: %s", model_id)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto" if device == "cuda" else None
        )
        if device == "cpu":
            self._model = self._model.to(device)
        self._device = device

    def _get_lang_name(self, code: str) -> str:
        return LANG_NAMES.get(code, code)

    def translate(self, text: str) -> str:
        self._load()
        assert self._tokenizer is not None and self._model is not None

        source_name = self._get_lang_name(self.cfg.source_lang)
        target_name = self._get_lang_name(self.cfg.target_lang)

        prompt = TRANSLATION_PROMPT.format(
            source_lang=source_name,
            target_lang=target_name,
            text=text,
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.3,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = output[0][prompt_len:]
        result = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        # Take only the first line (avoid rambling)
        result = result.split("\n")[0].strip()
        return result
