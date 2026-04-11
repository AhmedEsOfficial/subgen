"""Translation via a llama.cpp server (OpenAI-compatible API).

Supports any GGUF model served by llama-server, llama-cli, or ollama.
The server can be auto-launched or assumed to be already running.
"""

from __future__ import annotations

import json
import logging
import os
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

    def _ensure_server(self) -> None:
        """Ensure the llama-server is running, launching it if needed."""
        if self._is_server_up():
            logger.info("llama-server already running at %s", self._base_url)
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
                return data["choices"][0]["message"].get("content", "").strip()
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

            translations = self._parse_numbered_output(raw, len(batch))

            # Fallback: if batch parsing returned fewer lines, translate missing individually
            if len(translations) < len(batch):
                logger.warning(
                    "Batch returned %d/%d — retranslating missing lines individually",
                    len(translations), len(batch),
                )
                for j in range(len(translations), len(batch)):
                    try:
                        single = self.translate(batch[j])
                        translations.append(single)
                    except Exception:
                        logger.warning("Individual translation failed for line %d, keeping original", j)
                        translations.append(batch[j])

            # Verify translations are actually in English (detect leftover Arabic)
            for j, trans in enumerate(translations):
                if self._is_arabic(trans) and j < len(batch):
                    logger.info("Line %d still Arabic after batch, retranslating: %s", j, trans[:60])
                    for _attempt in range(2):
                        try:
                            single = self.translate(batch[j])
                            if not self._is_arabic(single):
                                translations[j] = single
                                break
                        except Exception:
                            pass

            all_results.extend(translations[: len(batch)])

        return all_results

    @staticmethod
    def _is_arabic(text: str) -> bool:
        """Check if text is predominantly Arabic script."""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06ff' or '\u0750' <= c <= '\u077f')
        return arabic_chars > len(text) * 0.3

    def _parse_numbered_output(self, raw: str, expected: int) -> list[str]:
        """Parse numbered translation output like '1. text\\n2. text'."""
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        results: list[str] = []

        for line in lines:
            # Strip numbering: "1. text" → "text", "1) text" → "text"
            for sep in [". ", ") ", ": "]:
                parts = line.split(sep, 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    line = parts[1].strip()
                    break
            results.append(line)
            if len(results) >= expected:
                break

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
