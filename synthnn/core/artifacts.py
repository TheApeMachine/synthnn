"""
Shared artifact schema used across SynthNN cleanup components.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ArtifactType(Enum):
    """Types of audio artifacts to clean."""

    WHISTLING = "whistling"
    HUM = "hum"
    NOISE = "noise"
    DISTORTION = "distortion"
    PHASE_ISSUES = "phase_issues"
    RESONANCE_PEAKS = "resonance_peaks"


@dataclass
class ArtifactProfile:
    """Profile of a detected artifact."""

    artifact_type: ArtifactType
    frequency: float | None = None
    frequency_range: tuple[float, float] | None = None
    strength: float = 0.0
    time_range: tuple[float, float] | None = None

    # Stationary-tone evidence
    persistence: float = 0.0          # fraction of frames/blocks present
    phase_coherence: float = 0.0      # phase stability score
    multires_support: int = 1         # how many resolutions supported detection

