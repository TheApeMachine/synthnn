"""
Resonant-only audio cleanup (no STFT core).

This implements a resonator-bank / lock-in style coherence detector and canceller
using SynthNN primitives (complex state + damping).

Key idea:
  For each candidate frequency f0, demodulate the input by exp(-j 2π f0 t),
  integrate with a damped complex state (a ResonantNode with freq=0),
  then re-modulate to synthesize the coherent component and subtract it.
"""

# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .resonant_node import ResonantNode
from .artifacts import ArtifactProfile, ArtifactType
from typing import Callable


ProgressCallback = Callable[[str, float, dict[str, object]], None]


@dataclass
class ResonantCleanupConfig:
    # Candidate bank
    freq_range_hz: tuple[float, float] = (1000.0, 12000.0)
    num_nodes: int = 96

    # Estimator bandwidth (Hz). Higher tracks drift faster but is less selective.
    bandwidth_hz: float = 40.0

    # Detection thresholds
    min_persistence: float = 0.6
    min_phase_coherence: float = 0.4
    presence_db: float = 6.0
    max_artifacts: int = 8

    # Cancellation
    strength_multiplier: float = 1.6

    # Self-organization (adaptive retuning)
    adaptive_refine: bool = True
    refine_top_k: int = 8
    refine_range_hz: float = 120.0
    refine_steps: int = 15

    # Lateral inhibition (avoid selecting many nearby peaks)
    inhibit_radius_hz: float = 80.0
    inhibit_strength: float = 0.7  # 0=no inhibition, 1=strong inhibition


class ResonantBank:
    """
    Bank of demodulated integrators (one per candidate frequency).

    Each entry holds:
      - f0: target frequency (Hz)
      - node: ResonantNode with natural_freq=0 storing complex baseband estimate z
    """

    sample_rate: int
    freqs_hz: np.ndarray
    nodes: list[ResonantNode]
    decay_rate: float

    def __init__(self, sample_rate: int, freqs_hz: np.ndarray, bandwidth_hz: float):
        self.sample_rate = int(sample_rate)
        self.freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        self.nodes = [
            ResonantNode(node_id=f"lockin_{i}", frequency=0.0, damping=0.0, amplitude=0.0, phase=0.0)
            for i in range(len(self.freqs_hz))
        ]
        # Damping override is a decay rate (1/s) in the node.step() model.
        # We map bandwidth (Hz) -> rate via 2π bw.
        self.decay_rate = float(max(0.0, 2.0 * np.pi * float(bandwidth_hz)))

    def reset(self) -> None:
        for n in self.nodes:
            n.signal = 0j

    def process_block(self, x: np.ndarray, start_sample: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Process a block of audio and return per-node amplitude/phase traces (downsampled per block).
        """
        sr = float(self.sample_rate)
        dt = 1.0 / sr
        n = int(x.shape[0])
        t0 = float(start_sample) / sr
        # time vector for this block
        t = t0 + (np.arange(n, dtype=np.float64) / sr)

        amps = np.zeros(len(self.nodes), dtype=np.float64)
        phs = np.zeros(len(self.nodes), dtype=np.float64)

        # For each node, do lock-in integration across the block.
        # This is O(num_nodes * block_size) but avoids STFT as requested.
        for i, (f0, node) in enumerate(zip(self.freqs_hz, self.nodes, strict=False)):
            # Demodulate input into baseband
            ref = np.exp(-1j * 2.0 * np.pi * f0 * t)
            mixed = x.astype(np.float64) * ref

            # Integrate with exponential decay by stepping the resonant node (freq=0)
            # Equivalent to: z <- (1 - decay*dt) z + mixed
            z: complex = node.signal
            decay = self.decay_rate
            for m in mixed:
                # node.step for freq=0: z += coupling ; z *= (1 - decay*dt)
                z = (z + complex(m)) * (1.0 - decay * dt)
            node.signal = z

            amps[i] = float(abs(z))
            phs[i] = float(np.angle(z))

        return amps, phs

    def synthesize_component(self, f0: float, z: complex, t: np.ndarray) -> np.ndarray:
        """
        Re-modulate baseband estimate z back to time domain.

        For cosine input, lock-in yields z ≈ (A/2) e^{jφ}. Reconstruction: 2*Re(z e^{j2πf0 t})
        """
        return (2.0 * np.real(z * np.exp(1j * 2.0 * np.pi * f0 * t))).astype(np.float64)


def _score_candidates(
    amp_hist: np.ndarray,
    ph_hist: np.ndarray,
    freqs: np.ndarray,
    presence_db: float,
    min_persistence: float,
    min_phase_coherence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-node:
      - persistence
      - phase coherence
      - strength proxy
    """
    amp_db = 20.0 * np.log10(np.maximum(amp_hist, 1e-12))
    floor = np.median(amp_db, axis=0, keepdims=True)
    present = amp_db > (floor + float(presence_db))

    persistence = np.mean(present, axis=1).astype(np.float64)

    coherence = np.zeros(len(freqs), dtype=np.float64)
    for i in range(len(freqs)):
        consec = present[i, :-1] & present[i, 1:]
        if np.any(consec):
            ph = ph_hist[i, :]
            dphi = np.angle(np.exp(1j * (ph[1:] - ph[:-1])))
            coherence[i] = float(np.abs(np.mean(np.exp(1j * dphi[consec]))))
        else:
            coherence[i] = 0.0

    strength = np.median(amp_hist, axis=1).astype(np.float64)

    # Mask weak candidates
    keep = (persistence >= float(min_persistence)) & (coherence >= float(min_phase_coherence))
    persistence = persistence * keep
    coherence = coherence * keep
    strength = strength * keep

    return persistence, coherence, strength


def _select_with_inhibition(
    freqs: np.ndarray,
    persistence: np.ndarray,
    coherence: np.ndarray,
    strength: np.ndarray,
    cfg: ResonantCleanupConfig,
) -> list[int]:
    """
    Greedy selection with lateral inhibition: pick strongest, then suppress neighbors.
    """
    # Base score emphasizes evidence (persistence/coherence) and uses strength as a tie-breaker
    score = (persistence * coherence) + 0.05 * (strength / (np.max(strength) + 1e-12))
    score = score.astype(np.float64)

    chosen: list[int] = []
    suppressed = np.zeros_like(score, dtype=np.float64)

    for _ in range(int(cfg.max_artifacts)):
        effective = score * (1.0 - suppressed)
        j = int(np.argmax(effective))
        if effective[j] <= 0:
            break
        chosen.append(j)

        if cfg.inhibit_radius_hz > 0 and cfg.inhibit_strength > 0:
            dist = np.abs(freqs - freqs[j])
            mask = dist <= float(cfg.inhibit_radius_hz)
            # Linear inhibition inside radius
            falloff = 1.0 - (dist[mask] / float(cfg.inhibit_radius_hz))
            suppressed[mask] = np.clip(suppressed[mask] + float(cfg.inhibit_strength) * falloff, 0.0, 1.0)
        else:
            suppressed[j] = 1.0

    return chosen


def _estimate_lockin_amplitude(
    audio: np.ndarray,
    sample_rate: int,
    f0: float,
    bandwidth_hz: float,
    block_size: int,
) -> float:
    """
    Estimate steady-state baseband amplitude for a single frequency (coarse proxy).
    """
    bank = ResonantBank(sample_rate, np.array([float(f0)], dtype=np.float64), bandwidth_hz=float(bandwidth_hz))
    x = np.asarray(audio, dtype=np.float64)
    num_blocks = max(1, int(np.ceil(len(x) / block_size)))
    amps = []
    for b in range(num_blocks):
        start = b * block_size
        end = min(len(x), start + block_size)
        a, _p = bank.process_block(x[start:end], start_sample=start)
        amps.append(float(a[0]))
    return float(np.median(amps)) if amps else 0.0


def _refine_frequencies(
    audio: np.ndarray,
    sample_rate: int,
    freqs: np.ndarray,
    chosen_idx: list[int],
    cfg: ResonantCleanupConfig,
    block_size: int,
    progress_cb: ProgressCallback | None = None,
) -> np.ndarray:
    """
    Local frequency refinement around chosen candidates using lock-in amplitude maximization.
    """
    refined = freqs.copy()
    if not cfg.adaptive_refine:
        return refined

    k = min(int(cfg.refine_top_k), len(chosen_idx))
    for ci, idx in enumerate(chosen_idx[:k]):
        f0 = float(freqs[idx])
        lo = max(cfg.freq_range_hz[0], f0 - float(cfg.refine_range_hz))
        hi = min(cfg.freq_range_hz[1], f0 + float(cfg.refine_range_hz))
        grid = np.linspace(lo, hi, int(cfg.refine_steps), dtype=np.float64)
        best_f = f0
        best_a = -1.0
        for gi, f in enumerate(grid):
            a = _estimate_lockin_amplitude(audio, sample_rate, float(f), cfg.bandwidth_hz, block_size)
            if a > best_a:
                best_a = a
                best_f = float(f)
            if progress_cb is not None:
                # Fractional progress within refinement stage
                denom = max(1, k)
                frac = (ci + (gi + 1) / max(1, len(grid))) / denom
                progress_cb(
                    "refine",
                    float(frac),
                    {
                        "candidate_index": int(ci),
                        "candidate_node": int(idx),
                        "candidate_freq_hz": float(f0),
                        "grid_index": int(gi),
                        "grid_size": int(len(grid)),
                        "trial_freq_hz": float(f),
                        "best_freq_hz": float(best_f),
                        "best_amp": float(best_a),
                    },
                )
        refined[idx] = best_f

    return refined


def detect_artifacts_resonant(
    audio: np.ndarray,
    sample_rate: int,
    cfg: ResonantCleanupConfig,
    block_size: int = 2048,
    progress_cb: ProgressCallback | None = None,
    progress_top_k: int = 8,
) -> list[ArtifactProfile]:
    """
    Resonant-only artifact detection: run the bank over blocks and score candidates by
    persistence and phase coherence of the baseband estimate.
    """
    x = np.asarray(audio, dtype=np.float64)
    sr = int(sample_rate)

    fmin, fmax = cfg.freq_range_hz
    freqs = np.linspace(float(fmin), float(fmax), int(cfg.num_nodes), dtype=np.float64)
    bank = ResonantBank(sr, freqs, cfg.bandwidth_hz)

    num_blocks = max(1, int(np.ceil(len(x) / block_size)))
    amp_hist = np.zeros((len(freqs), num_blocks), dtype=np.float64)
    ph_hist = np.zeros((len(freqs), num_blocks), dtype=np.float64)

    for b in range(num_blocks):
        start = b * block_size
        end = min(len(x), start + block_size)
        amps, phs = bank.process_block(x[start:end], start_sample=start)
        amp_hist[:, b] = amps
        ph_hist[:, b] = phs

        if progress_cb is not None:
            # Presence is measured relative to a robust per-block floor.
            amp_db = 20.0 * np.log10(np.maximum(amps, 1e-12))
            floor_db = float(np.median(amp_db)) if amp_db.size else -120.0
            pres_db = (amp_db - floor_db).astype(np.float64)

            k = int(max(1, min(int(progress_top_k), len(freqs))))
            idx = np.argpartition(amps, -k)[-k:]
            idx = idx[np.argsort(amps[idx])[::-1]]
            progress_cb(
                "detect",
                float((b + 1) / max(1, num_blocks)),
                {
                    "block": int(b + 1),
                    "num_blocks": int(num_blocks),
                    "top_idx": [int(i) for i in idx.tolist()],
                    "top_freqs_hz": [float(freqs[i]) for i in idx.tolist()],
                    "top_amps": [float(amps[i]) for i in idx.tolist()],
                    "top_phs": [float(phs[i]) for i in idx.tolist()],
                    "top_presence_db": [float(pres_db[i]) for i in idx.tolist()],
                    "floor_db": float(floor_db),
                },
            )

    persistence, coherence, strength = _score_candidates(
        amp_hist, ph_hist, freqs, cfg.presence_db, cfg.min_persistence, cfg.min_phase_coherence
    )

    chosen = _select_with_inhibition(freqs, persistence, coherence, strength, cfg)

    # Optional adaptive refinement: move chosen freqs to local maxima
    if cfg.adaptive_refine and chosen:
        freqs_ref = _refine_frequencies(audio, sr, freqs, chosen, cfg, block_size, progress_cb=progress_cb)
    else:
        freqs_ref = freqs

    if progress_cb is not None:
        progress_cb(
            "select",
            1.0,
            {
                "chosen_idx": [int(i) for i in chosen],
                "chosen_freqs_hz": [float(freqs_ref[i]) for i in chosen],
                "inhibit_radius_hz": float(cfg.inhibit_radius_hz),
            },
        )

    bin_hz = float((fmax - fmin) / max(int(cfg.num_nodes) - 1, 1))
    artifacts: list[ArtifactProfile] = []
    for i in chosen:
        f0 = float(freqs_ref[i])
        artifacts.append(
            ArtifactProfile(
                artifact_type=ArtifactType.WHISTLING,
                frequency=f0,
                frequency_range=(float(f0 - bin_hz * 0.5), float(f0 + bin_hz * 0.5)),
                strength=float(strength[i]),
                persistence=float(persistence[i]),
                phase_coherence=float(coherence[i]),
                multires_support=1,
            )
        )

    artifacts.sort(key=lambda a: (a.persistence * a.phase_coherence, a.strength), reverse=True)
    return artifacts[: int(cfg.max_artifacts)]


def cleanup_resonant_only(
    audio: np.ndarray,
    sample_rate: int,
    artifacts: list[ArtifactProfile],
    cfg: ResonantCleanupConfig,
    block_size: int = 2048,
    progress_cb: ProgressCallback | None = None,
    progress_preview_len: int = 256,
) -> np.ndarray:
    """
    Resonant-only cancellation: re-run a bank only for selected artifact frequencies and subtract.
    """
    x = np.asarray(audio, dtype=np.float64)
    sr = int(sample_rate)
    # Build a bank for the selected artifact freqs only
    freqs = np.array([float(a.frequency) for a in artifacts if a.frequency], dtype=np.float64)
    if freqs.size == 0:
        return audio

    bank = ResonantBank(sr, freqs, cfg.bandwidth_hz)

    y = x.copy()
    num_blocks = max(1, int(np.ceil(len(x) / block_size)))
    for b in range(num_blocks):
        start = b * block_size
        end = min(len(x), start + block_size)
        xb = x[start:end]
        if xb.size == 0:
            continue

        # Update estimates for each node over this block
        _amps, _phs = bank.process_block(xb, start_sample=start)

        t = (np.arange(start, end, dtype=np.float64) / float(sr))
        cancel = np.zeros_like(xb, dtype=np.float64)
        per_artifact_rms: list[float] = []

        for f0, node, art in zip(freqs, bank.nodes, artifacts, strict=False):
            strength = float(np.clip(float(cfg.strength_multiplier) * float(art.phase_coherence), 0.0, 2.0))
            comp = bank.synthesize_component(float(f0), node.signal, t) * strength
            cancel += comp
            per_artifact_rms.append(float(np.sqrt(np.mean(comp**2))) if comp.size else 0.0)

        y[start:end] = y[start:end] - cancel

        if progress_cb is not None:
            # Lightweight previews for UI: downsample block waveforms
            stride = int(max(1, int(xb.size // max(16, int(progress_preview_len)))))
            x_prev = xb[::stride]
            c_prev = cancel[::stride]
            y_prev = y[start:end][::stride]
            progress_cb(
                "cancel",
                float((b + 1) / max(1, num_blocks)),
                {
                    "block": int(b + 1),
                    "num_blocks": int(num_blocks),
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "cancel_rms": float(np.sqrt(np.mean(cancel**2))) if cancel.size else 0.0,
                    "artifact_freqs_hz": [float(f) for f in freqs.tolist()],
                    "per_artifact_rms": [float(v) for v in per_artifact_rms],
                    "preview_stride": int(stride),
                    "x_preview": [float(v) for v in x_prev.tolist()],
                    "cancel_preview": [float(v) for v in c_prev.tolist()],
                    "y_preview": [float(v) for v in y_prev.tolist()],
                },
            )

    # Safety: avoid clipping
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)

