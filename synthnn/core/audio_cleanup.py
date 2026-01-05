"""
Resonant-only Audio Cleanup Engine for SynthNN.

This project direction is explicit: the cleanup primitive is implemented using
SynthNN's resonant dynamics (complex state + damping), not an STFT/FFT core.
"""

# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .artifacts import ArtifactProfile, ArtifactType
from .resonant_cleanup import (
    ResonantCleanupConfig,
    detect_artifacts_resonant,
    cleanup_resonant_only,
    ProgressCallback,
)


@dataclass
class CleanupResult:
    cleaned: np.ndarray
    artifacts: list[ArtifactProfile]


class ArtifactDetector:
    """Resonant-only artifact detector."""

    sample_rate: int

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = int(sample_rate)

    def detect(
        self,
        audio: np.ndarray,
        cfg: ResonantCleanupConfig | None = None,
        *,
        progress_cb: ProgressCallback | None = None,
    ) -> list[ArtifactProfile]:
        cfg = cfg or ResonantCleanupConfig()
        return detect_artifacts_resonant(audio, self.sample_rate, cfg, progress_cb=progress_cb)

    def detect_whistling(self, audio: np.ndarray, threshold: float = 0.1) -> list[ArtifactProfile]:
        """
        Compatibility shim. Kept because callers/tests expect it.
        This maps a single threshold into resonant detector settings.
        """
        t = float(max(0.01, min(0.5, threshold)))
        cfg = ResonantCleanupConfig(
            min_persistence=float(max(0.3, min(0.95, 0.75 - (t * 0.5)))),
            min_phase_coherence=float(max(0.05, min(0.95, 0.6 - (t * 0.3)))),
        )
        return self.detect(audio, cfg)


class AudioCleanupEngine:
    """Resonant-only cleanup engine."""

    sample_rate: int
    detector: ArtifactDetector

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = int(sample_rate)
        self.detector = ArtifactDetector(sample_rate)

    def analyze_artifacts(
        self,
        audio: np.ndarray,
        cfg: ResonantCleanupConfig | None = None,
        *,
        progress_cb: ProgressCallback | None = None,
    ) -> list[ArtifactProfile]:
        return self.detector.detect(audio, cfg=cfg, progress_cb=progress_cb)

    def cleanup(
        self,
        audio: np.ndarray,
        cfg: ResonantCleanupConfig | None = None,
        artifacts: list[ArtifactProfile] | None = None,
        *,
        progress_cb: ProgressCallback | None = None,
    ) -> CleanupResult:
        cfg = cfg or ResonantCleanupConfig()
        artifacts = artifacts or self.analyze_artifacts(audio, cfg=cfg, progress_cb=progress_cb)
        cleaned = cleanup_resonant_only(audio, self.sample_rate, artifacts, cfg, progress_cb=progress_cb)
        return CleanupResult(cleaned=cleaned, artifacts=artifacts)

    # Backwards compatible name (now always resonant-only)
    def cleanup_resonant_subtraction(
        self,
        audio: np.ndarray,
        cfg: ResonantCleanupConfig | None = None,
        *,
        progress_cb: ProgressCallback | None = None,
    ) -> tuple[np.ndarray, list[ArtifactProfile]]:
        result = self.cleanup(audio, cfg=cfg, progress_cb=progress_cb)
        return result.cleaned, result.artifacts


def create_cleanup_pipeline() -> Callable[[np.ndarray], np.ndarray]:
    """Return a resonant-only cleanup function."""
    engine = AudioCleanupEngine()

    def fn(audio: np.ndarray) -> np.ndarray:
        cleaned, _artifacts = engine.cleanup_resonant_subtraction(audio)
        return cleaned

    return fn


__all__ = [
    "ArtifactType",
    "ArtifactProfile",
    "ResonantCleanupConfig",
    "ArtifactDetector",
    "AudioCleanupEngine",
    "CleanupResult",
    "create_cleanup_pipeline",
]

