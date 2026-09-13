"""Subtitle formatting: line-wrap, CPS enforcement, merge/split logic."""

from __future__ import annotations

import logging
import math
import re
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
    and splitting of long ones.  Text is never dropped: anything that does not
    fit in one cue is split into additional timed cues.
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
        too_many_lines = len(_wrap_lines(text, cfg.max_chars_per_line)) > cfg.max_lines

        if duration > cfg.max_duration or cps > cfg.max_cps * 1.5 or too_many_lines:
            entries.extend(_split_segment(seg, cfg))
        else:
            entries.append(SubtitleEntry(index=0, start=seg.start, end=seg.end, text=text))

    # Merge segments that are too short
    entries = _merge_short(entries, cfg)

    # Apply line wrapping
    for entry in entries:
        entry.text = _wrap_text(entry.text, cfg.max_chars_per_line)

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
    # Word timestamps are only usable while they still spell out the text —
    # after translation, seg.words holds the source-language words.
    if len(seg.words) > 1 and _words_match_text(seg):
        return _split_by_words(seg, cfg)
    return _split_by_text(seg, cfg)


def _words_match_text(seg: TimedSegment) -> bool:
    """True if the word list reconstructs the segment text (ignoring whitespace)."""
    joined = re.sub(r"\s+", "", "".join(w.word for w in seg.words))
    return joined == re.sub(r"\s+", "", seg.text)


def _split_by_text(seg: TimedSegment, cfg: SubtitleConfig) -> list[SubtitleEntry]:
    """Split text into cues of at most max_lines wrapped lines, timed by character share.

    Uses enough cues that each lasts about max_duration or less, balancing text
    across them.
    """
    text = seg.text.strip()
    lines = _wrap_lines(text, cfg.max_chars_per_line)
    if not lines:
        return []

    duration = seg.end - seg.start
    cue_count = max(math.ceil(len(lines) / cfg.max_lines), math.ceil(duration / cfg.max_duration))

    chunks: list[str] = []
    for chunk in _balanced_chunks(text.split(), cue_count):
        if len(_wrap_lines(chunk, cfg.max_chars_per_line)) <= cfg.max_lines:
            chunks.append(chunk)
        else:  # rare: balancing produced one cue too wide to fit — split it by lines
            chunk_lines = _wrap_lines(chunk, cfg.max_chars_per_line)
            chunks.extend(
                " ".join(chunk_lines[i : i + cfg.max_lines])
                for i in range(0, len(chunk_lines), cfg.max_lines)
            )

    total_chars = sum(len(c) for c in chunks)

    entries: list[SubtitleEntry] = []
    start = seg.start
    for i, chunk in enumerate(chunks):
        end = seg.end if i == len(chunks) - 1 else start + duration * len(chunk) / total_chars
        entries.append(SubtitleEntry(index=0, start=start, end=end, text=chunk))
        start = end
    return entries


def _balanced_chunks(words: list[str], count: int) -> list[str]:
    """Split words into *count* runs of roughly equal character length."""
    count = max(1, min(count, len(words)))
    # ends[i] = character offset just after word i (including one joining space)
    ends: list[int] = []
    offset = 0
    for word in words:
        offset += len(word) + 1
        ends.append(offset)
    total = ends[-1]

    chunks: list[str] = []
    start = 0
    for k in range(1, count):
        target = total * k / count
        # Leave at least one word for each remaining chunk.
        last_allowed = len(words) - (count - k)
        cut = min(range(start, last_allowed), key=lambda i: abs(ends[i] - target)) + 1
        chunks.append(" ".join(words[start:cut]))
        start = cut
    chunks.append(" ".join(words[start:]))
    return chunks


def _split_by_words(seg: TimedSegment, cfg: SubtitleConfig) -> list[SubtitleEntry]:
    """Split a segment using word-level timestamps."""
    entries: list[SubtitleEntry] = []
    current_words: list[str] = []
    current_start = seg.words[0].start

    for word in seg.words:
        word_text = word.word.strip()
        candidate = " ".join(current_words + [word_text])
        too_long = len(_wrap_lines(candidate, cfg.max_chars_per_line)) > cfg.max_lines
        too_slow = word.end - current_start > cfg.max_duration

        if current_words and (too_long or too_slow):
            entries.append(
                SubtitleEntry(
                    index=0, start=current_start, end=word.start, text=" ".join(current_words)
                )
            )
            current_words = [word_text]
            current_start = word.start
        else:
            current_words.append(word_text)

    if current_words:
        entries.append(
            SubtitleEntry(
                index=0, start=current_start, end=seg.words[-1].end, text=" ".join(current_words)
            )
        )

    return entries


def _merge_short(entries: list[SubtitleEntry], cfg: SubtitleConfig) -> list[SubtitleEntry]:
    """Merge consecutive subtitle entries that are too short."""
    if not entries:
        return entries

    merged: list[SubtitleEntry] = [entries[0]]

    for entry in entries[1:]:
        prev = merged[-1]
        prev_duration = prev.end - prev.start
        combined_text = prev.text + " " + entry.text
        combined_duration = entry.end - prev.start

        if (
            prev_duration < cfg.min_duration
            and len(_wrap_lines(combined_text, cfg.max_chars_per_line)) <= cfg.max_lines
            and combined_duration <= cfg.max_duration
        ):
            merged[-1] = SubtitleEntry(
                index=0, start=prev.start, end=entry.end, text=combined_text
            )
        else:
            merged.append(entry)

    return merged


def _wrap_lines(text: str, max_chars: int) -> list[str]:
    return textwrap.wrap(text, width=max_chars)


def _wrap_text(text: str, max_chars: int) -> str:
    """Wrap text to the line length.  Never truncates — splitting handles line count."""
    if len(text) <= max_chars:
        return text
    return "\n".join(_wrap_lines(text, max_chars))
