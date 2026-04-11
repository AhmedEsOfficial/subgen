"""Extract audio from video files using ffmpeg."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import ffmpeg

from subgen.config import AudioConfig

logger = logging.getLogger(__name__)


def extract_audio(
    video_path: str | Path,
    cfg: AudioConfig,
    output_path: str | Path | None = None,
) -> Path:
    """Extract mono WAV audio from a video file.

    Returns the path to the extracted WAV file.  When *output_path* is None a
    temporary file is created (caller is responsible for cleanup).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = Path(tmp.name)
        tmp.close()
    else:
        output_path = Path(output_path)

    logger.info("Extracting audio from %s → %s", video_path, output_path)

    stream = ffmpeg.input(str(video_path))
    stream = ffmpeg.output(
        stream,
        str(output_path),
        ac=1 if cfg.mono else 2,
        ar=cfg.sample_rate,
        format="wav",
    )
    ffmpeg.run(stream, overwrite_output=True, quiet=True)

    logger.info("Audio extracted: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path
