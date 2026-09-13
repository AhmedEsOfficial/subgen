"""Voice note transcription + translation (WhatsApp / Telegram PTT audio).

Differs from the film Pipeline in ways that matter for short, informal audio:

* translation is per-line with surrounding context (``translate_with_context``),
  which is slow for a film but keeps short notes coherent and aligned;
* all files are transcribed first and the ASR model is unloaded before
  translation starts, so whisper and the LLM never share VRAM and llama-server
  is launched once per run;
* output is a readable bilingual Markdown transcript plus source/target SRTs,
  with low-confidence ASR words listed for a human to re-listen to.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import soundfile as sf

from subgen.asr.base import BaseASR
from subgen.audio.extractor import extract_audio
from subgen.config import AppConfig
from subgen.langdetect.base import BaseLangDetector
from subgen.models import TimedSegment, TimedWord
from subgen.subtitle.formatter import format_subtitles
from subgen.subtitle.writer import write_srt
from subgen.translation.base import BaseTranslator
from subgen.vad.base import BaseVAD

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".ogg", ".oga", ".opus", ".m4a", ".mp3", ".wav", ".aac", ".amr", ".flac", ".webm"}
TRANSLATION_DOMAIN = "an informal WhatsApp voice note"
# Words the ASR decoded below this probability are listed for manual review.  Calibrated
# on real notes: 0.7 caught the misheard words on one, while flagging ~6% of words (mostly
# fillers) on another.  It is a pointer only — whisper can mishear a word confidently.
LOW_CONFIDENCE = 0.7
# Shorter words are fillers/particles (اه، يا، او) and flagging them is noise.
MIN_REVIEW_LETTERS = 3

# NLLB-style codes → short codes for output filenames.
SHORT_LANG = {"eng_Latn": "en", "arz_Arab": "ar", "arb_Arab": "ar", "fra_Latn": "fr", "spa_Latn": "es"}


@dataclass
class VoiceNote:
    audio: Path
    out_dir: Path
    duration: float = 0.0
    language: str = ""
    segments: list[TimedSegment] = field(default_factory=list)
    translations: list[str] = field(default_factory=list)

    @property
    def markdown_path(self) -> Path:
        return self.out_dir / f"{self.audio.stem}.md"

    def srt_path(self, lang: str) -> Path:
        return self.out_dir / f"{self.audio.stem}.{lang}.srt"


def collect_inputs(paths: list[Path]) -> list[Path]:
    """Expand directories into their audio files (sorted, non-recursive); keep explicit files."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
            )
        else:
            files.append(path)
    return list(dict.fromkeys(files))  # de-duplicate, keep order


class VoiceNoteTranscriber:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    def run(self, files: list[Path], out_dir: Path | None = None, force: bool = False) -> tuple[list[VoiceNote], list[Path]]:
        """Process *files*.  Returns (completed notes, failed files)."""
        notes: list[VoiceNote] = []
        for audio in files:
            note = VoiceNote(audio=audio, out_dir=out_dir or audio.parent)
            if note.markdown_path.exists() and not force:
                logger.info("Skipping %s — %s exists (use --force to redo)", audio, note.markdown_path)
                continue
            notes.append(note)

        if not notes:
            return [], []

        failed: list[Path] = []
        notes = self._transcribe_all(notes, failed)
        if notes:
            self._translate_all(notes)
        for note in notes:
            self._write_markdown(note)
            logger.info("Done: %s → %s", note.audio, note.markdown_path)
        return notes, failed

    # --- Stage 1: ASR for every file, then free the model -------------------

    def _transcribe_all(self, notes: list[VoiceNote], failed: list[Path]) -> list[VoiceNote]:
        vad = BaseVAD.from_config(self.cfg.vad)
        lang_detector = BaseLangDetector.from_config(self.cfg.language_detection)
        asr = BaseASR.from_config(self.cfg.asr)

        done: list[VoiceNote] = []
        try:
            for note in notes:
                try:
                    self._transcribe(note, vad, lang_detector, asr)
                except Exception:
                    logger.exception("Transcription failed for %s", note.audio)
                    failed.append(note.audio)
                    continue
                if not note.segments:
                    logger.warning("No speech found in %s", note.audio)
                note.out_dir.mkdir(parents=True, exist_ok=True)
                self._write_srt(note, note.language, [s.text for s in note.segments])
                done.append(note)
        finally:
            del asr, vad, lang_detector
            _release_gpu_memory()
        return done

    def _transcribe(self, note: VoiceNote, vad: BaseVAD, lang_detector: BaseLangDetector, asr: BaseASR) -> None:
        logger.info("=== Transcribing %s ===", note.audio)
        wav = extract_audio(note.audio, self.cfg.audio)
        try:
            note.duration = sf.info(str(wav)).duration
            regions = vad.detect(wav)
            note.language = lang_detector.detect(wav)
            segments = asr.transcribe(wav, language=note.language, regions=regions)
        finally:
            os.unlink(wav)

        # Whisper pads the final window past the end of the file; clamp to real audio.
        note.segments = [
            TimedSegment(
                start=seg.start,
                end=min(seg.end, note.duration),
                text=seg.text,
                language=seg.language,
                words=seg.words,
            )
            for seg in segments
            if seg.start < note.duration
        ]
        logger.info("%s: %d segments, %.1fs", note.audio.name, len(note.segments), note.duration)

    # --- Stage 2: translation ------------------------------------------------

    def _translate_all(self, notes: list[VoiceNote]) -> None:
        translator = BaseTranslator.from_config(self.cfg.translation)
        if translator is None:
            logger.info("Translation disabled (engine=none)")
            return
        try:
            for note in notes:
                if not note.segments:
                    continue
                logger.info("=== Translating %s (%d lines) ===", note.audio.name, len(note.segments))
                note.translations = translator.translate_with_context(
                    [s.text for s in note.segments], domain=TRANSLATION_DOMAIN
                )
                self._write_srt(note, self._target_code(), note.translations)
        finally:
            if hasattr(translator, "shutdown"):
                translator.shutdown()

    # --- Output --------------------------------------------------------------

    def _target_code(self) -> str:
        target = self.cfg.translation.target_lang
        return SHORT_LANG.get(target, target)

    def _write_srt(self, note: VoiceNote, lang: str, texts: list[str]) -> None:
        # Words only describe the source text; the formatter checks this, but be explicit.
        keep_words = texts == [s.text for s in note.segments]
        segments = [
            TimedSegment(start=s.start, end=s.end, text=t, language=lang, words=s.words if keep_words else [])
            for s, t in zip(note.segments, texts)
        ]
        entries = format_subtitles(segments, self.cfg.subtitle)
        for entry in entries:  # min-duration padding must not run past the file
            entry.end = min(entry.end, note.duration)
        write_srt(entries, note.srt_path(lang))

    def _write_markdown(self, note: VoiceNote) -> None:
        target = self._target_code()
        lines = [
            f"# {note.audio.name}",
            "",
            f"- **Duration:** {_clock(note.duration)}",
            f"- **Language:** {note.language}",
            f"- **Transcription:** {self.cfg.asr.engine} `{self.cfg.asr.model}`",
        ]
        if note.translations:
            lines.append(
                f"- **Translation:** {self.cfg.translation.engine} `{Path(self.cfg.translation.model).name}`"
            )
        lines += [f"- **Generated:** {datetime.now():%Y-%m-%d %H:%M}", ""]

        if not note.segments:
            lines += ["_No speech detected._", ""]
        if note.translations:
            lines += [f"## Translation ({target})", ""]
            lines += [f"- **[{_clock(s.start)}]** {t}" for s, t in zip(note.segments, note.translations)]
            lines.append("")
        if note.segments:
            lines += [f"## Transcript ({note.language})", "", '<div dir="rtl">', ""]
            lines += [f"- **[{_clock(s.start)}]** {s.text}" for s in note.segments]
            lines += ["", "</div>", ""]

        review = [
            (s, [w.word for w in s.words if _needs_review(w)])
            for s in note.segments
        ]
        review = [(s, words) for s, words in review if words]
        if review:
            lines += [
                "## Worth re-listening",
                "",
                f"Words the speech model was less than {LOW_CONFIDENCE:.0%} sure of. "
                "This is a rough pointer: confidently misheard words will not appear here.",
                "",
            ]
            lines += [f"- **[{_clock(s.start)}]** {'، '.join(words)}" for s, words in review]
            lines.append("")

        note.markdown_path.write_text("\n".join(lines), encoding="utf-8")


def _needs_review(word: TimedWord) -> bool:
    letters = sum(ch.isalpha() for ch in word.word)
    return word.probability is not None and word.probability < LOW_CONFIDENCE and letters >= MIN_REVIEW_LETTERS


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02}:{secs:02}"


def _release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
