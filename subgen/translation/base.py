"""Abstract base class for translation engines."""

from __future__ import annotations

from abc import ABC, abstractmethod

from subgen.config import TranslationConfig


class BaseTranslator(ABC):
    """Translation engine interface."""

    def __init__(self, cfg: TranslationConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def translate(self, text: str) -> str:
        """Translate a single text string from source to target language."""

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate a batch of texts.  Default: sequential fallback."""
        return [self.translate(t) for t in texts]

    @classmethod
    def from_config(cls, cfg: TranslationConfig) -> BaseTranslator | None:
        """Factory: create the appropriate translator from config.

        Returns None if engine is 'none' (no translation needed).
        """
        if cfg.engine == "none":
            return None
        if cfg.engine == "nllb":
            from subgen.translation.nllb import NLLBTranslator

            return NLLBTranslator(cfg)
        if cfg.engine == "opus":
            from subgen.translation.opus import OpusMTTranslator

            return OpusMTTranslator(cfg)
        if cfg.engine == "local_llm":
            from subgen.translation.local_llm import LocalLLMTranslator

            return LocalLLMTranslator(cfg)
        if cfg.engine == "llama_cpp":
            from subgen.translation.llama_cpp import LlamaCppTranslator

            return LlamaCppTranslator(cfg)
        raise ValueError(f"Unknown translation engine: {cfg.engine}")
