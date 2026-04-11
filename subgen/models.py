"""Shared data models used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TimedWord:
    """A single word with precise timestamps."""

    start: float  # seconds
    end: float
    word: str


@dataclass
class TimedSegment:
    """A speech segment with timestamps and optional word-level detail."""

    start: float  # seconds
    end: float
    text: str
    language: str | None = None
    words: list[TimedWord] = field(default_factory=list)


@dataclass
class SpeechRegion:
    """A VAD-detected speech region (no text yet)."""

    start: float  # seconds
    end: float


@dataclass
class SubtitleEntry:
    """A single subtitle cue ready for SRT output."""

    index: int
    start: float  # seconds
    end: float
    text: str
