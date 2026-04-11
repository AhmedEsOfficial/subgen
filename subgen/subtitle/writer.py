"""SRT file writer."""

from __future__ import annotations

import logging
from pathlib import Path

import pysrt

from subgen.models import SubtitleEntry

logger = logging.getLogger(__name__)


def _seconds_to_subrip_time(seconds: float) -> pysrt.SubRipTime:
    """Convert seconds to pysrt SubRipTime."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return pysrt.SubRipTime(hours=hours, minutes=minutes, seconds=secs, milliseconds=millis)


def write_srt(entries: list[SubtitleEntry], output_path: str | Path) -> Path:
    """Write subtitle entries to an SRT file."""
    output_path = Path(output_path)
    logger.info("Writing %d subtitles to %s", len(entries), output_path)

    srt_file = pysrt.SubRipFile()

    for entry in entries:
        item = pysrt.SubRipItem(
            index=entry.index,
            start=_seconds_to_subrip_time(entry.start),
            end=_seconds_to_subrip_time(entry.end),
            text=entry.text,
        )
        srt_file.append(item)

    srt_file.save(str(output_path), encoding="utf-8")
    logger.info("SRT file saved: %s", output_path)
    return output_path
