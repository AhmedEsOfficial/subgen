"""Audio loading utility using soundfile (avoids torchaudio/torchcodec issues)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_audio(path: str | Path, target_sr: int | None = None) -> tuple[torch.Tensor, int]:
    """Load audio file and return (waveform, sample_rate).

    Returns a 1-D float32 tensor (mono) and the sample rate.
    If *target_sr* is given the audio is resampled to that rate.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # Mix to mono: average channels
    mono = data.mean(axis=1)
    wav = torch.from_numpy(mono)

    if target_sr is not None and sr != target_sr:
        wav = _resample(wav, sr, target_sr)
        sr = target_sr

    return wav, sr


def _resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Simple linear-interpolation resample."""
    ratio = target_sr / orig_sr
    new_len = int(len(wav) * ratio)
    indices = torch.linspace(0, len(wav) - 1, new_len)
    left = indices.long().clamp(max=len(wav) - 1)
    right = (left + 1).clamp(max=len(wav) - 1)
    frac = indices - left.float()
    return wav[left] * (1 - frac) + wav[right] * frac
