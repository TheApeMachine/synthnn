"""
Cleanup report schema + metrics for the coherence-based audio cleanup pipeline.

This module intentionally contains no UI code. It standardizes outputs so the CLI
and Streamlit demo can present consistent evidence (inspectability-first).
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, welch

from .audio_cleanup import ArtifactProfile


SCHEMA_VERSION = "cleanup_report_v1"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bandpower_welch(x: np.ndarray, sr: int, f0: float, bw_hz: float) -> float:
    f, pxx = welch(x, fs=sr, nperseg=min(16384, len(x)))
    lo = max(0.0, f0 - bw_hz / 2.0)
    hi = min(sr / 2.0, f0 + bw_hz / 2.0)
    mask = (f >= lo) & (f <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(pxx[mask], f[mask]))


def spectral_flatness(x: np.ndarray, sr: int) -> float:
    _f, pxx = welch(x, fs=sr, nperseg=min(16384, len(x)))
    pxx = np.maximum(pxx, 1e-20)
    gmean = float(np.exp(np.mean(np.log(pxx))))
    amean = float(np.mean(pxx))
    return gmean / amean if amean > 0 else 0.0


def transient_envelope(x: np.ndarray, sr: int, hp_hz: float = 1000.0) -> np.ndarray:
    """
    Crude transient proxy:
    - high-pass
    - rectify
    - smooth
    """
    nyq = sr / 2.0
    cut = float(np.clip(hp_hz / nyq, 1e-4, 0.99))
    b, a = butter(4, cut, btype="high")
    hp = filtfilt(b, a, x).astype(np.float64)
    env = np.abs(hp)
    win = max(8, int(sr * 0.01))  # 10ms smoothing
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(env, kernel, mode="same")


def transient_preservation(x_in: np.ndarray, x_out: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Return a small set of transient preservation indicators without needing a clean reference.
    """
    e_in = transient_envelope(x_in, sr)
    e_out = transient_envelope(x_out, sr)

    # Pearson correlation (guard degenerate cases)
    if np.std(e_in) < 1e-12 or np.std(e_out) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(e_in, e_out)[0, 1])

    energy_ratio = float((np.mean(e_out**2) + 1e-20) / (np.mean(e_in**2) + 1e-20))

    return {
        "transient_env_corr": corr,
        "transient_env_energy_ratio": energy_ratio,
    }


def artifact_confidence(a: ArtifactProfile, expected_support: int = 3) -> float:
    """
    Evidence-weighted confidence in [0, 1].
    """
    p = float(np.clip(getattr(a, "persistence", 0.0), 0.0, 1.0))
    c = float(np.clip(getattr(a, "phase_coherence", 0.0), 0.0, 1.0))
    s = float(np.clip(getattr(a, "multires_support", 1) / max(expected_support, 1), 0.0, 1.0))
    conf = 0.45 * p + 0.45 * c + 0.10 * s
    return float(np.clip(conf, 0.0, 1.0))


def cleanup_severity(artifacts: List[ArtifactProfile]) -> float:
    """
    A simple severity scalar in [0, 1] summarizing how 'artifacty' the input looks.

    This is *not* a quality score. It's an evidence-weighted summary of detected artifacts.
    """
    if not artifacts:
        return 0.0
    confs = [artifact_confidence(a) for a in artifacts if a.frequency]
    if not confs:
        return 0.0
    # saturating mean of confidences
    return float(np.clip(np.mean(confs), 0.0, 1.0))


def artifact_to_schema(a: ArtifactProfile, expected_support: int = 3) -> Dict[str, Any]:
    d = asdict(a)
    # keep only stable fields; rename for schema clarity
    out: Dict[str, Any] = {
        "artifact_type": a.artifact_type.value,
        "frequency_hz": float(a.frequency) if a.frequency is not None else None,
        "frequency_range_hz": (
            [float(a.frequency_range[0]), float(a.frequency_range[1])] if a.frequency_range else None
        ),
        "strength": float(a.strength),
        "persistence": float(getattr(a, "persistence", 0.0)),
        "phase_coherence": float(getattr(a, "phase_coherence", 0.0)),
        "multires_support": int(getattr(a, "multires_support", 1)),
        "confidence": artifact_confidence(a, expected_support=expected_support),
    }
    return out


def build_report_v1(
    *,
    input_meta: Dict[str, Any],
    config: Dict[str, Any],
    artifacts: List[ArtifactProfile],
    methods: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a schema-locked report. `methods` is a dict like:
        {
          "coherence_filter": {"metrics": {...}, "per_artifact": {...}},
          "notch_baseline": {...},
        }
    """
    expected_support = len(config.get("detector", {}).get("window_sizes", [])) or 3
    art_schema = [artifact_to_schema(a, expected_support=expected_support) for a in artifacts if a.frequency]

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _iso_now(),
        "input": input_meta,
        "config": config,
        "summary": {
            "num_artifacts": int(len(art_schema)),
            "cleanup_severity": cleanup_severity(artifacts),
        },
        "artifacts": art_schema,
        "results": {
            "methods": methods,
        },
    }
    return report


def method_metrics(
    *,
    x_in: np.ndarray,
    x_out: np.ndarray,
    sr: int,
    artifacts: List[ArtifactProfile],
    bandwidth_hz: float,
    runtime_sec: float,
) -> Dict[str, Any]:
    """
    Standard metrics block for a single cleanup method.
    """
    duration_sec = float(len(x_in) / sr)
    realtime_factor = duration_sec / runtime_sec if runtime_sec > 1e-9 else float("inf")

    flat_in = spectral_flatness(x_in, sr)
    flat_out = spectral_flatness(x_out, sr)

    per_artifact: Dict[str, Any] = {}
    for a in artifacts:
        if not a.frequency:
            continue
        f0 = float(a.frequency)
        p_before = bandpower_welch(x_in, sr, f0, bandwidth_hz)
        p_after = bandpower_welch(x_out, sr, f0, bandwidth_hz)
        red_db = float(10.0 * np.log10((p_before + 1e-30) / (p_after + 1e-30)))
        per_artifact[f"{f0:.2f}"] = {
            "bandpower_before": float(p_before),
            "bandpower_after": float(p_after),
            "reduction_db": red_db,
        }

    trans = transient_preservation(x_in, x_out, sr)

    return {
        "metrics": {
            "runtime_sec": float(runtime_sec),
            "realtime_factor": float(realtime_factor),
            "spectral_flatness_before": float(flat_in),
            "spectral_flatness_after": float(flat_out),
            "spectral_flatness_change": float(flat_out - flat_in),
            **trans,
        },
        "per_artifact": per_artifact,
    }

