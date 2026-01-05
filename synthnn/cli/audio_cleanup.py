"""
CLI: synthnn-clean

Thin-slice productization of the audio cleanup engine:
- input WAV -> cleaned WAV
- JSON report (detected artifacts + metrics)
- A/B spectrum plot
"""

# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportMissingImports=false

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
# Ensure matplotlib/fontconfig caches are writable (important in sandboxed envs).
_cwd = Path.cwd()
_ = os.environ.setdefault("MPLCONFIGDIR", str(_cwd / ".mplconfig"))
_ = os.environ.setdefault("XDG_CACHE_HOME", str(_cwd / ".cache"))
_mpl = Path(os.environ["MPLCONFIGDIR"])
_xdg = Path(os.environ["XDG_CACHE_HOME"])
_mpl.mkdir(parents=True, exist_ok=True)
_xdg.mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib
from scipy.io import wavfile
from scipy.signal import welch

from synthnn.core.audio_cleanup import AudioCleanupEngine, ResonantCleanupConfig
from synthnn.core.cleanup_report import build_report_v1, method_metrics
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _to_float_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        maxv = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float32) / maxv
    else:
        x = x.astype(np.float32)
    if np.max(np.abs(x)) > 0:
        x = np.clip(x, -1.0, 1.0)
    return x


def _from_float_to_int16(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    if y.size == 0:
        return y.astype(np.int16)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y)))
    # Only attenuate to avoid clipping; never boost.
    if peak > 0.98 and peak > 1e-12:
        y = y * (0.98 / peak)
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).astype(np.int16)


def _plot_spectrum_ab(
    noisy: np.ndarray,
    cleaned: np.ndarray,
    sr: int,
    artifact_freqs: list[float],
    out_path: Path,
    max_hz: float = 12000.0,
) -> None:
    f1, p1 = welch(noisy, fs=sr, nperseg=min(16384, len(noisy)))
    f2, p2 = welch(cleaned, fs=sr, nperseg=min(16384, len(cleaned)))

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    _ = ax.semilogy(f1, p1 + 1e-20, label="before")
    _ = ax.semilogy(f2, p2 + 1e-20, label="after")

    for f0 in artifact_freqs:
        _ = ax.axvline(f0, color="red", linestyle="--", alpha=0.35)

    _ = ax.set_title("SynthNN Audio Cleanup: A/B Spectrum (Welch PSD)")
    _ = ax.set_xlabel("Frequency (Hz)")
    _ = ax.set_ylabel("Power")
    _ = ax.set_xlim(0, max_hz)
    ax.grid(True, alpha=0.2)
    _ = ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="synthnn-clean",
        description="Resonance-based audio artifact cleanup (WAV -> WAV + report + plot).",
    )
    _ = p.add_argument("-i", "--input", required=True, help="Input WAV path")
    _ = p.add_argument("-o", "--output", required=True, help="Output WAV path (cleaned)")
    _ = p.add_argument("--report", default=None, help="Write JSON report to this path")
    _ = p.add_argument("--plot", default=None, help="Write A/B spectrum plot PNG to this path")

    _ = p.add_argument("--freq-min", type=float, default=1000.0, help="Min frequency (Hz) for resonant bank.")
    _ = p.add_argument("--freq-max", type=float, default=12000.0, help="Max frequency (Hz) for resonant bank.")
    _ = p.add_argument("--num-nodes", type=int, default=96, help="Number of resonant candidates in the bank (higher = finer resolution, slower).")
    _ = p.add_argument("--bandwidth-hz", type=float, default=40.0, help="Resonant estimator bandwidth (Hz). Higher tracks drift faster; lower is more selective.")
    _ = p.add_argument("--min-persistence", type=float, default=0.6, help="Min persistence (fraction of blocks present). Lower detects more; higher is stricter.")
    _ = p.add_argument("--min-phase-coherence", type=float, default=0.4, help="Min phase-coherence. Higher focuses on stronger coherence.")
    _ = p.add_argument("--presence-db", type=float, default=6.0, help="Presence threshold in dB above per-block median.")
    _ = p.add_argument("--max-artifacts", type=int, default=8, help="Max artifacts to cancel.")
    _ = p.add_argument("--strength-multiplier", type=float, default=1.6, help="Multiply cancellation strength (higher = more aggressive).")

    _ = p.add_argument("--adaptive-refine", action="store_true", help="Enable adaptive frequency refinement (recommended).")
    _ = p.add_argument("--refine-top-k", type=int, default=8, help="How many top candidates to refine.")
    _ = p.add_argument("--refine-range-hz", type=float, default=120.0, help="Refinement search range (±Hz) around each candidate.")
    _ = p.add_argument("--refine-steps", type=int, default=15, help="Refinement grid steps per candidate.")

    _ = p.add_argument("--inhibit-radius-hz", type=float, default=80.0, help="Lateral inhibition radius (Hz).")
    _ = p.add_argument("--inhibit-strength", type=float, default=0.7, help="Lateral inhibition strength (0..1).")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    report_path = Path(args.report).expanduser() if args.report else output_path.with_suffix(".report.json")
    plot_path = Path(args.plot).expanduser() if args.plot else output_path.with_suffix(".spectrum.png")

    sr, x_raw = wavfile.read(str(input_path))
    x = _to_float_mono(x_raw)

    engine = AudioCleanupEngine(sample_rate=int(sr))

    # Run cleanup
    t0 = time.perf_counter()
    cfg = ResonantCleanupConfig(
        freq_range_hz=(float(args.freq_min), float(args.freq_max)),
        num_nodes=int(args.num_nodes),
        bandwidth_hz=float(args.bandwidth_hz),
        min_persistence=float(args.min_persistence),
        min_phase_coherence=float(args.min_phase_coherence),
        presence_db=float(args.presence_db),
        max_artifacts=int(args.max_artifacts),
        strength_multiplier=float(args.strength_multiplier),
        adaptive_refine=bool(args.adaptive_refine),
        refine_top_k=int(args.refine_top_k),
        refine_range_hz=float(args.refine_range_hz),
        refine_steps=int(args.refine_steps),
        inhibit_radius_hz=float(args.inhibit_radius_hz),
        inhibit_strength=float(args.inhibit_strength),
    )
    y, artifacts = engine.cleanup_resonant_subtraction(x, cfg=cfg)
    t1 = time.perf_counter()

    # Write audio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_path), int(sr), _from_float_to_int16(y))

    runtime_sec = float(t1 - t0)

    input_meta = {
        "path": str(input_path),
        "sample_rate": int(sr),
        "num_samples": int(len(x)),
        "duration_sec": float(len(x) / sr),
    }

    config = {
        "detector": {
            "min_persistence": float(args.min_persistence),
            "min_phase_coherence": float(args.min_phase_coherence),
            "freq_range_hz": [float(args.freq_min), float(args.freq_max)],
            "num_nodes": int(args.num_nodes),
            "bandwidth_hz": float(args.bandwidth_hz),
            "presence_db": float(args.presence_db),
            "adaptive_refine": bool(args.adaptive_refine),
            "refine_top_k": int(args.refine_top_k),
            "refine_range_hz": float(args.refine_range_hz),
            "refine_steps": int(args.refine_steps),
            "inhibit_radius_hz": float(args.inhibit_radius_hz),
            "inhibit_strength": float(args.inhibit_strength),
        },
        "method": {
            "max_artifacts": int(args.max_artifacts),
            "strength_multiplier": float(args.strength_multiplier),
        },
    }

    methods = {
        "selected_method": method_metrics(
            x_in=x,
            x_out=y,
            sr=int(sr),
            artifacts=artifacts[: int(args.max_artifacts)],
            bandwidth_hz=float(args.bandwidth_hz),
            runtime_sec=runtime_sec,
        )
    }

    report = build_report_v1(
        input_meta=input_meta,
        config=config,
        artifacts=artifacts[: int(args.max_artifacts)],
        methods=methods,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    _ = report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    # Plot
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_freqs = [float(a.frequency) for a in artifacts[: int(args.max_artifacts)] if a.frequency]
    _plot_spectrum_ab(x, y, int(sr), artifact_freqs, plot_path)

    print(f"Wrote: {output_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()

