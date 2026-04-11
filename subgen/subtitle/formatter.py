"""Subtitle formatting: line-wrap, CPS enforcement, merge/split logic."""

from __future__ import annotations

import logging
import textwrap

from subgen.config import SubtitleConfig
from subgen.models import SubtitleEntry, TimedSegment

logger = logging.getLogger(__name__)


def format_subtitles(
    segments: list[TimedSegment],
    cfg: SubtitleConfig,
) -> list[SubtitleEntry]:
    """Convert timed segments into formatted subtitle entries.

    Applies line-wrapping, CPS enforcement, merging of short segments,
    and splitting of long ones.
    """
    entries: list[SubtitleEntry] = []

    for seg in segments:
        if not seg.text.strip():
            continue
        duration = seg.end - seg.start
        if duration <= 0:
            continue

        text = seg.text.strip()
        cps = len(text) / duration

        # Split segments that are too long or have too-high CPS
        if duration > cfg.max_duration or cps > cfg.max_cps * 1.5:
            sub_entries = _split_segment(seg, cfg)
            entries.extend(sub_entries)
        else:
            entries.append(SubtitleEntry(index=0, start=seg.start, end=seg.end, text=text))

    # Merge segments that are too short
    entries = _merge_short(entries, cfg)

    # Apply line wrapping
    for entry in entries:
        entry.text = _wrap_text(entry.text, cfg.max_chars_per_line, cfg.max_lines)

    # Enforce minimum duration
    for entry in entries:
        if entry.end - entry.start < cfg.min_duration:
            entry.end = entry.start + cfg.min_duration

    # Re-index
    for i, entry in enumerate(entries, start=1):
        entry.index = i

    logger.info("Formatted %d subtitle entries", len(entries))
    return entries


def _split_segment(seg: TimedSegment, cfg: SubtitleConfig) -> list[SubtitleEntry]:
    """Split a long segment into multiple subtitle entries."""
    text = seg.text.strip()
    duration = seg.end - seg.start

    # If we have word-level timestamps, split at word boundaries
    if seg.words and len(seg.words) > 1:
        return _split_by_words(seg, cfg)

    # Otherwise split text evenly
    max_chars = cfg.max_chars_per_line * cfg.max_lines
    chunks = textwrap.wrap(text, width=max_chars)
    if not chunks:
        return []

    chunk_duration = duration / len(chunks)
    entries = []
    for i, chunk in enumerate(chunks):
        start = seg.start + i * chunk_duration
        end = start + chunk_duration
        entries.append(SubtitleEntry(index=0, start=start, end=end, text=chunk))
    return entries


def _split_by_words(seg: TimedSegment, cfg: SubtitleConfig) -> list[SubtitleEntry]:
    """Split a segment using word-level timestamps."""
    max_chars = cfg.max_chars_per_line * cfg.max_lines
    entries: list[SubtitleEntry] = []
    current_words: list[str] = []
    current_start = seg.words[0].start if seg.words else seg.start
    current_len = 0

    for word in seg.words:
        word_text = word.word.strip()
        new_len = current_len + len(word_text) + (1 if current_words else 0)

        if new_len > max_chars and current_words:
            text = " ".join(current_words)
            entries.append(
                SubtitleEntry(index=0, start=current_start, end=word.start, text=text)
            )
            current_words = [word_text]
            current_start = word.start
            current_len = len(word_text)
        else:
            current_words.append(word_text)
            current_len = new_len

    if current_words:
        text = " ".join(current_words)
        end = seg.words[-1].end if seg.words else seg.end
        entries.append(SubtitleEntry(index=0, start=current_start, end=end, text=text))

    return entries


def _merge_short(entries: list[SubtitleEntry], cfg: SubtitleConfig) -> list[SubtitleEntry]:
    """Merge consecutive subtitle entries that are too short."""
    if not entries:
        return entries

    max_chars = cfg.max_chars_per_line * cfg.max_lines
    merged: list[SubtitleEntry] = [entries[0]]

    for entry in entries[1:]:
        prev = merged[-1]
        prev_duration = prev.end - prev.start
        combined_text = prev.text + " " + entry.text
        combined_duration = entry.end - prev.start

        if (
            prev_duration < cfg.min_duration
            and len(combined_text) <= max_chars
            and combined_duration <= cfg.max_duration
        ):
            merged[-1] = SubtitleEntry(
                index=0, start=prev.start, end=entry.end, text=combined_text
            )
        else:
            merged.append(entry)

    return merged


def _wrap_text(text: str, max_chars: int, max_lines: int) -> str:
    """Wrap text to fit within line length and line count constraints."""
    if len(text) <= max_chars:
        return text

    lines = textwrap.wrap(text, width=max_chars)
    if len(lines) > max_lines:
        # Truncate to max_lines, keeping as much text as possible
        lines = lines[:max_lines]
        # Add ellipsis to last line if truncated
        if len(lines[-1]) > max_chars - 3:
            lines[-1] = lines[-1][: max_chars - 3] + "..."

    return "\n".join(lines)
