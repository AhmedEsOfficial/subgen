"""Optional audio denoising via Demucs vocal isolation."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def is_demucs_available() -> bool:
    """Check whether the demucs CLI is installed."""
    return shutil.which("demucs") is not None or shutil.which("python") is not None


def denoise(audio_path: str | Path) -> Path:
    """Run Demucs htdemucs to isolate vocals from the audio.

    Returns the path to the isolated vocals WAV.  Falls back to the
    original file if demucs is not available.
    """
    audio_path = Path(audio_path)
    if not shutil.which("demucs"):
        logger.warning("demucs not found — skipping denoising")
        return audio_path

    tmp_dir = tempfile.mkdtemp(prefix="subgen_demucs_")
    logger.info("Running demucs on %s (output dir: %s)", audio_path, tmp_dir)

    subprocess.run(
        [
            "demucs",
            "--two-stems", "vocals",
            "-n", "htdemucs",
            "-o", tmp_dir,
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )

    vocals_path = Path(tmp_dir) / "htdemucs" / audio_path.stem / "vocals.wav"
    if not vocals_path.exists():
        logger.warning("Demucs vocals output not found at %s — using original", vocals_path)
        return audio_path

    logger.info("Denoised audio: %s", vocals_path)
    return vocals_path
