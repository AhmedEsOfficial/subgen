"""Translation via a llama.cpp server (OpenAI-compatible API).

Supports any GGUF model served by llama-server, llama-cli, or ollama.
The server can be auto-launched or assumed to be already running.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from subgen.config import TranslationConfig
from subgen.translation.base import BaseTranslator

logger = logging.getLogger(__name__)

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

SYSTEM_PROMPT = (
    "You are a professional film subtitle translator specializing in Egyptian Arabic dialect. "
    "Translate the given Egyptian Arabic dialogue into natural, idiomatic English. "
    "Preserve the tone, register, and cultural context. "
    "For colloquial expressions, provide the closest natural English equivalent. "
    "Output ONLY the translation — no explanations, no notes, no transliteration."
)

BATCH_PROMPT = (
    "Translate each numbered line of Egyptian Arabic dialogue below into natural English. "
    "Output ONLY the numbered translations, one per line, matching the input numbering exactly. "
    "Do not add any explanations or notes.\n\n{lines}"
)

CONTEXT_SYSTEM_PROMPT = (
    "You are a professional translator specializing in {source} dialect. "
    "You translate {domain} one fragment at a time. Speech was split at pauses, so a "
    "fragment may begin or end mid-sentence: translate exactly that fragment into natural, "
    "idiomatic, informal {target}, and keep it a fragment. Never include content that "
    "belongs to the surrounding lines. "
    "Output ONLY the {target} translation — no notes, no transliteration, no numbering, no quotes."
)

CONTEXT_USER_PROMPT = (
    "Transcript around the fragment, for context only (do not translate these):\n"
    "{context}\n\n"
    "Translate ONLY line {number}:\n{text}"
)

# Lines of preceding / following transcript shown to the model per fragment.
# Few following lines: more makes the model pull the next fragment into its answer.
CONTEXT_BEFORE = 8
CONTEXT_AFTER = 1

_NUMBERED_LINE = re.compile(r"^[\s*_]*(\d+)[\s*_]*[.):\-][\s*_]*(.*\S)\s*$")
_REASONING_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LlamaCppTranslator(BaseTranslator):
    """Translation using a llama.cpp server with OpenAI-compatible API."""

    def __init__(self, cfg: TranslationConfig) -> None:
        super().__init__(cfg)
        self._server_process: subprocess.Popen | None = None
        self._base_url: str = cfg.llama_server_url or "http://127.0.0.1:8081"
        self._model_path: str = cfg.model
        self._auto_launch: bool = cfg.llama_auto_launch
        self._gpu_layers: str = cfg.llama_gpu_layers
        self._ctx_size: int = cfg.llama_ctx_size
        self._server_ready: bool = False

    def _ensure_server(self) -> None:
        """Ensure the llama-server is running, launching it if needed."""
        if self._server_ready:
            return
        if self._is_server_up():
            logger.info("llama-server already running at %s", self._base_url)
            self._server_ready = True
            return

        if not self._auto_launch:
            raise RuntimeError(
                f"llama-server not reachable at {self._base_url} and auto_launch is disabled. "
                "Start it manually or set llama_auto_launch=true."
            )

        llama_server = shutil.which("llama-server")
        if not llama_server:
            raise RuntimeError("llama-server not found on PATH")

        model_path = Path(self._model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Extract port from URL
        port = self._base_url.rstrip("/").split(":")[-1]

        cmd = [
            llama_server,
            "-m", str(model_path),
            "-ngl", self._gpu_layers,
            "--port", port,
            "-c", str(self._ctx_size),
            "--log-disable",
        ]
        logger.info("Launching llama-server: %s", " ".join(cmd))
        self._server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for server to be ready
        for _ in range(60):
            if self._is_server_up():
                logger.info("llama-server ready at %s", self._base_url)
                self._server_ready = True
                return
            time.sleep(1)

        raise RuntimeError("llama-server failed to start within 60 seconds")

    def _is_server_up(self) -> bool:
        try:
            req = Request(f"{self._base_url}/health", method="GET")
            with urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                return data.get("status") == "ok"
        except (URLError, OSError, json.JSONDecodeError):
            return False

    def _chat_completion(self, messages: list[dict], max_tokens: int = 2048) -> str:
        """Call the OpenAI-compatible chat completions endpoint."""
        payload = json.dumps({
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }).encode()

        req = Request(
            f"{self._base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        for attempt in range(3):
            try:
                with urlopen(req, timeout=600) as resp:
                    data = json.loads(resp.read())
                content = data["choices"][0]["message"].get("content") or ""
                return _REASONING_BLOCK.sub("", content).strip()
            except (URLError, OSError, json.JSONDecodeError) as e:
                logger.warning("Chat completion attempt %d failed: %s", attempt + 1, e)
                if attempt == 2:
                    raise
                time.sleep(2)

    def translate(self, text: str) -> str:
        self._ensure_server()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        result = self._chat_completion(messages, max_tokens=1024)
        return result.split("\n")[0].strip() if result else text

    def translate_batch(self, texts: list[str]) -> list[str]:
        self._ensure_server()

        batch_size = self.cfg.batch_size
        all_results: list[str] = []

        total_batches = (len(texts) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            batch = texts[i : i + batch_size]
            logger.info(
                "Translating batch %d/%d (lines %d-%d of %d)",
                batch_idx + 1, total_batches, i + 1, i + len(batch), len(texts),
            )
            numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(batch))
            prompt = BATCH_PROMPT.format(lines=numbered)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            # Scale max_tokens with batch size
            max_tokens = min(4096, 256 * len(batch))
            raw = self._chat_completion(messages, max_tokens=max_tokens)

            # Keyed by the model's own numbering, so a skipped, merged, or echoed
            # line cannot shift every later translation onto the wrong source line.
            translations = self._parse_numbered_output(raw, len(batch))

            for j, source in enumerate(batch):
                trans = translations.get(j, "")
                if not trans or self._is_arabic(trans):
                    logger.info(
                        "Line %d %s after batch, retranslating individually",
                        i + j + 1, "missing" if not trans else "still Arabic",
                    )
                    trans = self._retranslate(source, fallback=trans)
                all_results.append(trans)

        return all_results

    def _retranslate(self, source: str, fallback: str) -> str:
        """Translate one line on its own, up to 2 attempts; keep *fallback* or source on failure."""
        for _attempt in range(2):
            try:
                single = self.translate(source)
            except (URLError, OSError, KeyError, json.JSONDecodeError) as e:
                logger.warning("Individual translation failed: %s", e)
                continue
            if single and not self._is_arabic(single):
                return single
        logger.warning("Line left untranslated: %s", source[:60])
        return fallback or source

    def translate_with_context(self, texts: list[str], domain: str) -> list[str]:
        """One request per fragment, with neighbouring lines as context.

        Slower than translate_batch but keeps every output aligned to its input and
        lets the model resolve fragments that were split mid-sentence.  Suited to
        short material such as voice notes.
        """
        self._ensure_server()

        system = CONTEXT_SYSTEM_PROMPT.format(
            source=LANG_NAMES.get(self.cfg.source_lang, self.cfg.source_lang),
            target=LANG_NAMES.get(self.cfg.target_lang, self.cfg.target_lang),
            domain=domain,
        )
        results: list[str] = []
        for i, text in enumerate(texts):
            lo, hi = max(0, i - CONTEXT_BEFORE), min(len(texts), i + CONTEXT_AFTER + 1)
            context = "\n".join(f"{k+1}. {texts[k]}" for k in range(lo, hi))
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": CONTEXT_USER_PROMPT.format(
                    context=context, number=i + 1, text=text,
                )},
            ]

            translation = ""
            for attempt in range(3):
                candidate = self._last_answer_line(self._chat_completion(messages, max_tokens=1024))
                if candidate and not self._is_arabic(candidate):
                    translation = candidate
                    break
                logger.info("Line %d attempt %d unusable: %r", i + 1, attempt + 1, candidate[:60])
            if not translation:
                logger.warning("Line %d left untranslated: %s", i + 1, text[:60])
                translation = text

            logger.info("Translated line %d/%d", i + 1, len(texts))
            results.append(translation)
        return results

    @staticmethod
    def _last_answer_line(raw: str) -> str:
        """Final non-empty line of a reply, without numbering or wrapping quotes."""
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        if not lines:
            return ""
        match = _NUMBERED_LINE.match(lines[-1])
        answer = match.group(2) if match else lines[-1]
        return answer.strip().strip('"“”').strip()

    @staticmethod
    def _is_arabic(text: str) -> bool:
        """Check if text is predominantly Arabic script."""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06ff' or '\u0750' <= c <= '\u077f')
        return arabic_chars > len(text) * 0.3

    @staticmethod
    def _parse_numbered_output(raw: str, expected: int) -> dict[int, str]:
        """Parse '1. text' / '2) text' lines into {0-based index: text}.

        Unnumbered lines (reasoning, notes) and numbers outside 1..expected are
        ignored; the first occurrence of a number wins.
        """
        results: dict[int, str] = {}
        for line in raw.splitlines():
            match = _NUMBERED_LINE.match(line)
            if not match:
                continue
            index = int(match.group(1)) - 1
            if 0 <= index < expected and index not in results:
                results[index] = match.group(2).strip()
        return results

    def shutdown(self) -> None:
        """Shut down the auto-launched server."""
        if self._server_process is not None:
            logger.info("Shutting down llama-server (PID %d)", self._server_process.pid)
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            self._server_process = None

    def __del__(self):
        self.shutdown()
