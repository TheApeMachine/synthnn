"""
SynthNN Coherence Filter Demo (Streamlit)

Single-purpose UI:
- Upload a WAV (e.g. a Suno song with whistle artifacts)
- Detect coherent stationary tones (multi-res, phase/persistence)
- Run two methods on the same detected artifacts:
    1) Coherence Filter (resonant subtraction, phase-tracked)
    2) Notch Baseline (static iirnotch at detected freqs)
- Inspect outputs + metrics + plots + JSON report (schema v1)
"""

# pyright: reportAny=false, reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.io import wavfile
from scipy.signal import welch, butter, filtfilt

# Make matplotlib figures match Streamlit's dark UI.
plt.style.use("dark_background")
plt.rcParams.update(
    {
        "figure.facecolor": "#0E1117",
        "axes.facecolor": "#0E1117",
        "savefig.facecolor": "#0E1117",
        "axes.edgecolor": "#3A3A3A",
        "axes.labelcolor": "#E6E6E6",
        "xtick.color": "#CFCFCF",
        "ytick.color": "#CFCFCF",
        "grid.color": "#2A2A2A",
        "text.color": "#E6E6E6",
    }
)

# Allow running directly from the repo without requiring `pip install -e .`
import sys as _sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from synthnn.core.audio_cleanup import AudioCleanupEngine, ResonantCleanupConfig, ArtifactProfile, ArtifactType  # noqa: E402
from synthnn.core.cleanup_report import build_report_v1, method_metrics  # noqa: E402


def to_float_audio(x: np.ndarray) -> np.ndarray:
    """
    Convert WAV samples to float32 in [-1, 1].
    Preserves channel shape: (n,) or (n, channels).
    """
    if np.issubdtype(x.dtype, np.integer):
        maxv = float(np.iinfo(x.dtype).max)
        y = x.astype(np.float32) / maxv
    else:
        y = x.astype(np.float32)
    # Normalize only if input is out of range (some float WAVs)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    y = np.clip(y, -1.0, 1.0)
    return y


def to_mono_mix(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return x.mean(axis=1).astype(np.float32)


def prepare_audio_for_playback(x: np.ndarray, headroom: float = 0.98) -> np.ndarray:
    """
    Avoid audible distortion by preventing hard clipping when exporting to int16.
    Important: this function will **not boost** audio (to avoid making artifacts
    sound louder). It only attenuates if needed to fit in headroom.
    - remove DC offset
    - replace non-finite values
    - attenuate if peak exceeds headroom
    """
    y = np.asarray(x, dtype=np.float64)
    if y.size == 0:
        return y.astype(np.float32)

    # Replace NaN/Inf
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove DC per-channel (helps avoid asymmetric clipping)
    if y.ndim == 1:
        y = y - float(np.mean(y))
    else:
        y = y - np.mean(y, axis=0, keepdims=True)

    peak = float(np.max(np.abs(y)))
    if peak > float(headroom) and peak > 1e-12:
        y = y * (float(headroom) / peak)

    # Final safety clip (should rarely engage after normalization)
    y = np.clip(y, -1.0, 1.0)
    return y.astype(np.float32)


def float_to_wav_bytes(x: np.ndarray, sr: int, *, reference_peak: float | None = None) -> bytes:
    """
    Convert float audio to WAV bytes for Streamlit playback/download.
    If reference_peak is provided, we will never amplify above that reference;
    we only attenuate to avoid clipping.
    """
    y = np.asarray(x, dtype=np.float64)
    if reference_peak is not None:
        # If output is quieter than reference, do NOT amplify. If louder, attenuate down to reference.
        out_peak = float(np.max(np.abs(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)))) if y.size else 0.0
        if out_peak > 1e-12 and out_peak > float(reference_peak):
            y = y * (float(reference_peak) / out_peak)
    y = prepare_audio_for_playback(y)
    audio_int16 = (y.astype(np.float32) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, audio_int16)
    _ = buf.seek(0)
    return buf.read()


def _stats(x: np.ndarray) -> dict[str, object]:
    y = np.asarray(x)
    if y.size == 0:
        return {"peak": 0.0, "rms": 0.0, "nan": 0, "inf": 0}
    y64 = y.astype(np.float64, copy=False)
    return {
        "peak": float(np.max(np.abs(y64))),
        "rms": float(np.sqrt(np.mean(y64**2))),
        "nan": int(np.isnan(y64).sum()),
        "inf": int(np.isinf(y64).sum()),
        "shape": list(y.shape),
    }


def parse_window_sizes(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            pass
    return tuple(out) if out else (1024, 2048, 4096)


def parse_freq_list(s: str) -> list[float]:
    """
    Parse a comma/space-separated list of frequencies in Hz.
    """
    parts = [p.strip() for p in s.replace(" ", ",").split(",") if p.strip()]
    out: list[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            continue
    return out


def bandpass_audio(x: np.ndarray, sr: int, center_hz: float, bw_hz: float) -> np.ndarray:
    """
    Butterworth bandpass around center_hz with width bw_hz (applies per channel).
    """
    nyq = sr / 2.0
    lo = max(1.0, center_hz - bw_hz / 2.0) / nyq
    hi = min(nyq - 1.0, center_hz + bw_hz / 2.0) / nyq
    if not (0.0 < lo < hi < 1.0):
        return np.zeros_like(x)
    ba = cast(tuple[np.ndarray, np.ndarray], butter(4, [lo, hi], btype="band", output="ba"))  # type: ignore[misc]
    b = ba[0]
    a = ba[1]
    if x.ndim == 1:
        return filtfilt(b, a, x).astype(np.float32)
    ys = []
    for ch in range(x.shape[1]):
        ys.append(filtfilt(b, a, x[:, ch]))
    return np.stack(ys, axis=1).astype(np.float32)


def plot_psd_overlay(
    x_in: np.ndarray,
    x_out: np.ndarray,
    sr: int,
    artifact_freqs: list[float],
    selected_freq: float | None,
    max_hz: float = 12000.0,
) -> Figure:
    f0, p0 = welch(x_in, fs=sr, nperseg=min(16384, len(x_in)))
    f1, p1 = welch(x_out, fs=sr, nperseg=min(16384, len(x_out)))

    fig, ax = plt.subplots(figsize=(12, 4))
    _ = ax.semilogy(f0, p0 + 1e-20, label="input")
    _ = ax.semilogy(f1, p1 + 1e-20, label="resonant output")

    for f in artifact_freqs:
        lw = 2.5 if (selected_freq is not None and abs(f - selected_freq) < 1e-6) else 1.0
        alpha = 0.7 if lw > 1.0 else 0.35
        _ = ax.axvline(f, color="red", linestyle="--", linewidth=lw, alpha=alpha)

    _ = ax.set_title("A/B Spectrum (Welch PSD) + Detected Artifact Frequencies")
    _ = ax.set_xlabel("Frequency (Hz)")
    _ = ax.set_ylabel("Power")
    _ = ax.set_xlim(0, max_hz)
    ax.grid(True, alpha=0.2)
    _ = ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="SynthNN Coherence Filter Demo", layout="wide")
    st.title("Coherence Filter (Audio Artifact Cleanup)")

    st.markdown(
        """
Upload a WAV (e.g. a Suno track with whistling artifacts). This demo uses a **resonant bank** (lock‑in style)
to detect and cancel coherent components using **complex resonant state + damping** (no STFT/FFT core).
"""
    )

    with st.sidebar:
        st.subheader("Detection")
        st.caption("These settings control *what gets detected*. If nothing changes audibly, start here: lower thresholds (detect more) or increase max artifacts.")
        st.caption("Resonant bank coverage:")
        freq_min = st.number_input("Min frequency (Hz)", min_value=20.0, max_value=20000.0, value=1000.0, step=10.0)
        freq_max = st.number_input("Max frequency (Hz)", min_value=100.0, max_value=24000.0, value=12000.0, step=10.0)
        num_nodes = st.number_input("Bank nodes", min_value=16, max_value=512, value=96, step=8, help="More nodes = finer frequency resolution, slower.")
        bandwidth_hz = st.slider(
            "Estimator bandwidth (Hz)",
            5.0,
            200.0,
            40.0,
            1.0,
            help="Higher tracks drift faster but is less selective; lower is more selective but may miss drift.",
        )
        min_persistence = st.slider(
            "Min persistence",
            0.3,
            0.95,
            0.70,
            0.01,
            help=(
                "Fraction of frames a tone must be present to be considered an artifact. "
                "Lower = detect more (risk false positives), higher = only very steady tones."
            ),
        )
        min_phase_coherence = st.slider(
            "Min phase coherence",
            0.0,
            0.99,
            0.60,
            0.01,
            help=(
                "How phase-locked the tone is across frames. "
                "Higher = more strictly 'coherent' (good for whistles), lower = allow wobblier tones."
            ),
        )
        presence_db = st.slider(
            "Presence threshold (dB)",
            0.0,
            24.0,
            6.0,
            0.5,
            help="How far above the per-block median a node must be to count as 'present'.",
        )
        max_artifacts = st.number_input(
            "Max artifacts",
            min_value=1,
            max_value=20,
            value=8,
            step=1,
            help="Max number of detected tones to treat as artifacts (ranked by strength/evidence).",
        )
        manual_freqs = st.text_input(
            "Manual target frequencies (Hz)",
            value="",
            help=(
                "Optional override. Provide comma-separated frequencies (e.g. '3750, 7500') to force cleanup targets "
                "even if detection misses or latches onto the wrong tone."
            ),
        )

        st.subheader("Self‑organization")
        adaptive_refine = st.checkbox(
            "Adaptive refinement",
            value=True,
            help="After coarse detection, locally refine top candidates to maximize resonant response.",
        )
        refine_top_k = st.slider("Refine top‑K", 1, 16, 8, 1)
        refine_range_hz = st.slider("Refine range (±Hz)", 20.0, 600.0, 120.0, 10.0)
        refine_steps = st.slider("Refine steps", 5, 41, 15, 2)

        st.subheader("Lateral inhibition")
        inhibit_radius_hz = st.slider("Inhibition radius (Hz)", 0.0, 800.0, 80.0, 10.0)
        inhibit_strength = st.slider("Inhibition strength", 0.0, 1.0, 0.7, 0.05)

        st.subheader("Cancellation")
        strength_multiplier = st.slider(
            "Strength multiplier",
            0.5,
            2.0,
            1.6,
            0.05,
            help="Higher = more aggressive cancellation. Too high can remove desired tones if mis-detected.",
        )

    weights = (1, 1)
    left, right = st.columns(weights, gap="large")

    # --- Left: load/play/control
    with left:
        uploaded = st.file_uploader("Upload WAV", type=["wav"])
        status_slot = None
        progress_slot = None
        if uploaded is None:
            st.info("Upload a WAV to start.")
            run_clicked = False
            x = None
            sr = 0
        else:
            try:
                sr, x_raw = wavfile.read(uploaded)
                x = to_float_audio(x_raw)
            except Exception as e:
                st.error(f"Failed to read WAV: {e}")
                run_clicked = False
                x = None
                sr = 0

        if x is not None and sr > 0:
            channels = 1 if x.ndim == 1 else int(x.shape[1])
            st.write(f"**Loaded:** {len(x) / sr:.2f}s @ {sr} Hz ({'mono' if channels==1 else f'{channels}-ch'})")

            # Play original bytes (avoids export-induced artifacts)
            try:
                uploaded.seek(0)
                st.audio(uploaded.read(), format="audio/wav")
            except Exception:
                st.audio(float_to_wav_bytes(x, sr), format="audio/wav")

            with st.expander("Signal diagnostics"):
                st.json({"input": _stats(x)})

            run_col, status_col = st.columns([1, 2], gap="medium")
            with run_col:
                run_clicked = st.button("Run detection + cleanup", type="primary")
            with status_col:
                status_slot = st.empty()
            progress_slot = st.empty()
        else:
            run_clicked = False

    # --- Right: always-reserved visualization area
    with right:
        if uploaded is None or x is None or sr <= 0:
            st.info("Upload a WAV on the left, then click **Run detection + cleanup**.")
            return

    # Reference peak used to avoid "making artifacts louder" during export/playback.
    ref_peak = float(np.max(np.abs(np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)))) if x.size else 0.0

    if not run_clicked:
        if status_slot is not None:
            status_slot.info("Ready. Adjust settings in the sidebar if needed, then run cleanup from the left panel.")
        if progress_slot is not None:
            progress_slot.empty()
        return

    # --- Run: visuals on the right, outputs on the left
    engine = AudioCleanupEngine(sample_rate=int(sr))
    cfg = ResonantCleanupConfig(
        freq_range_hz=(float(freq_min), float(freq_max)),
        num_nodes=int(num_nodes),
        bandwidth_hz=float(bandwidth_hz),
        min_persistence=float(min_persistence),
        min_phase_coherence=float(min_phase_coherence),
        presence_db=float(presence_db),
        max_artifacts=int(max_artifacts),
        strength_multiplier=float(strength_multiplier),
        adaptive_refine=bool(adaptive_refine),
        refine_top_k=int(refine_top_k),
        refine_range_hz=float(refine_range_hz),
        refine_steps=int(refine_steps),
        inhibit_radius_hz=float(inhibit_radius_hz),
        inhibit_strength=float(inhibit_strength),
    )

    x_mono = to_mono_mix(x)
    channels = 1 if x.ndim == 1 else int(x.shape[1])

    # Live status/progress moved next to the Run button (left)
    if status_slot is None or progress_slot is None:
        # Shouldn't happen, but avoid crashing the app if layout changes.
        status_slot = st.empty()
        progress_slot = st.empty()
    live_info = status_slot
    live_progress = progress_slot.progress(0.0, text="Starting…")

    with right:
        live_heatmap = st.empty()
        live_cancel_line = st.empty()
        live_cancel_table = st.empty()
        live_wave = st.empty()
        live_evidence = st.empty()
        live_events = st.empty()
        t_start = time.perf_counter()
        last_ui = 0.0

        # Rolling “heatmap” buffer: nodes (rows) × time (cols)
        heat_cols = 180
        heat = np.zeros((int(num_nodes), int(heat_cols)), dtype=np.float32)
        cancel_hist: list[float] = []

        # Live “focused candidate” evidence (top-1 over time)
        ev_cols = 180
        ev_presence_db = np.zeros(ev_cols, dtype=np.float64)
        ev_present = np.zeros(ev_cols, dtype=np.float64)
        ev_phase = np.zeros(ev_cols, dtype=np.float64)
        ev_persist = np.zeros(ev_cols, dtype=np.float64)
        ev_coh = np.zeros(ev_cols, dtype=np.float64)
        ev_freq = np.zeros(ev_cols, dtype=np.float64)
        ev_state: dict[str, str] = {"value": "SEARCH"}
        ev_log: list[dict[str, object]] = []
        selected_idx: list[int] = []

        def progress_cb(stage: str, frac: float, payload: dict[str, object]) -> None:
            nonlocal last_ui
            now = time.perf_counter()
            if (now - last_ui) < 0.12 and frac < 0.999:
                return
            last_ui = now

            frac_c = float(max(0.0, min(1.0, frac)))
            elapsed = now - t_start
            eta = (elapsed * (1.0 - frac_c) / max(frac_c, 1e-6)) if frac_c > 0 else 0.0

            blk = payload.get("block")
            nblk = payload.get("num_blocks")
            blk_s = f"{blk}/{nblk}" if (blk is not None and nblk is not None) else "—"

            live_progress.progress(frac_c, text=f"{stage}: {frac_c*100:.1f}%")
            live_info.markdown(f"**Stage:** {stage}  \n**Progress:** {frac_c*100:.1f}%  \n**ETA:** {eta:.1f}s  \n**Block:** {blk_s}")

            if stage == "detect":
                top_freqs = payload.get("top_freqs_hz")
                top_amps = payload.get("top_amps")
                top_idx = payload.get("top_idx")
                top_phs = payload.get("top_phs")
                top_pres = payload.get("top_presence_db")
                if (
                    isinstance(top_freqs, list)
                    and isinstance(top_amps, list)
                    and isinstance(top_idx, list)
                    and len(top_freqs) == len(top_amps) == len(top_idx)
                ):
                    heat[:, :-1] = heat[:, 1:]
                    heat[:, -1] = 0.0
                    for i, a in zip(top_idx, top_amps, strict=False):
                        if isinstance(i, int) and 0 <= i < heat.shape[0]:
                            heat[i, -1] = float(a)

                    img = np.log10(1e-12 + heat)
                    fig, ax = plt.subplots(figsize=(10, 2.6))
                    _ = ax.imshow(img, aspect="auto", origin="lower", interpolation="nearest", cmap="magma")
                    _ = ax.set_title("Resonant bank activity (top-K sparse heatmap; brighter = stronger lock-in amplitude)")
                    _ = ax.set_xlabel("time (rolling)")
                    _ = ax.set_ylabel("bank node (freq ↑)")
                    if top_idx and isinstance(top_idx[0], int):
                        _ = ax.axhline(float(top_idx[0]), color="cyan", linewidth=1.25, alpha=0.9)
                        if inhibit_radius_hz > 0:
                            bin_hz = (float(freq_max) - float(freq_min)) / max(int(num_nodes) - 1, 1)
                            rad_nodes = float(inhibit_radius_hz) / max(bin_hz, 1e-6)
                            _ = ax.axhspan(float(top_idx[0]) - rad_nodes, float(top_idx[0]) + rad_nodes, color="cyan", alpha=0.08)
                    for si in selected_idx:
                        _ = ax.axhline(float(si), color="red", linewidth=1.5, alpha=0.85)

                    _ = ax.set_yticks([0, heat.shape[0] // 2, heat.shape[0] - 1])
                    _ = ax.set_yticklabels(
                        [
                            f"{float(freq_min):.0f} Hz",
                            f"{(float(freq_min) + float(freq_max)) / 2.0:.0f} Hz",
                            f"{float(freq_max):.0f} Hz",
                        ]
                    )
                    fig.tight_layout()
                    live_heatmap.pyplot(fig)
                    plt.close(fig)

                    if (
                        isinstance(top_phs, list)
                        and isinstance(top_pres, list)
                        and len(top_phs) == len(top_pres) == len(top_idx)
                        and len(top_idx) >= 1
                        and isinstance(top_idx[0], int)
                        and isinstance(top_freqs[0], (int, float))
                        and isinstance(top_pres[0], (int, float))
                        and isinstance(top_phs[0], (int, float))
                    ):
                        ev_presence_db[:-1] = ev_presence_db[1:]
                        ev_present[:-1] = ev_present[1:]
                        ev_phase[:-1] = ev_phase[1:]
                        ev_persist[:-1] = ev_persist[1:]
                        ev_coh[:-1] = ev_coh[1:]
                        ev_freq[:-1] = ev_freq[1:]

                        pres0 = float(top_pres[0])
                        ph0 = float(top_phs[0])
                        ev_presence_db[-1] = pres0
                        ev_phase[-1] = ph0
                        ev_freq[-1] = float(top_freqs[0])
                        is_present = 1.0 if pres0 >= float(presence_db) else 0.0
                        ev_present[-1] = is_present
                        alpha = 0.08
                        ev_persist[-1] = (1.0 - alpha) * ev_persist[-2] + alpha * is_present if ev_cols >= 2 else is_present

                        w = 24
                        ph_win = ev_phase[-w:]
                        pr_win = ev_present[-w:]
                        if np.any(pr_win > 0.5):
                            dphi = np.angle(np.exp(1j * (ph_win[1:] - ph_win[:-1])))
                            mask = (pr_win[1:] > 0.5) & (pr_win[:-1] > 0.5)
                            if np.any(mask):
                                ev_coh[-1] = float(np.abs(np.mean(np.exp(1j * dphi[mask]))))
                            else:
                                ev_coh[-1] = 0.0
                        else:
                            ev_coh[-1] = 0.0

                        prev_state = ev_state["value"]
                        if ev_persist[-1] < float(min_persistence):
                            ev_state["value"] = "SEARCH"
                        elif ev_coh[-1] < float(min_phase_coherence):
                            ev_state["value"] = "PRESENT"
                        else:
                            ev_state["value"] = "LOCKED"
                        if prev_state != ev_state["value"]:
                            ev_log.append(
                                {
                                    "t": f"{(time.perf_counter() - t_start):.1f}s",
                                    "event": f"{prev_state} → {ev_state['value']}",
                                    "freq_hz": float(ev_freq[-1]),
                                    "presence_db": float(ev_presence_db[-1]),
                                    "persistence": float(ev_persist[-1]),
                                    "coherence": float(ev_coh[-1]),
                                }
                            )
                            if len(ev_log) > 30:
                                del ev_log[: len(ev_log) - 30]

                        fig2, axs = plt.subplots(3, 1, figsize=(10, 3.2), sharex=True)
                        xax = np.arange(ev_cols)
                        axs[0].plot(xax, ev_presence_db, color="white", linewidth=1.0)
                        axs[0].axhline(float(presence_db), color="red", linestyle="--", linewidth=1.0, alpha=0.8)
                        axs[0].set_ylabel("presence (dB)")
                        axs[0].grid(True, alpha=0.15)

                        axs[1].plot(xax, ev_coh, color="cyan", linewidth=1.0)
                        axs[1].axhline(float(min_phase_coherence), color="red", linestyle="--", linewidth=1.0, alpha=0.8)
                        axs[1].set_ylabel("coherence")
                        axs[1].set_ylim(0.0, 1.05)
                        axs[1].grid(True, alpha=0.15)

                        axs[2].plot(xax, ev_persist, color="orange", linewidth=1.0)
                        axs[2].axhline(float(min_persistence), color="red", linestyle="--", linewidth=1.0, alpha=0.8)
                        axs[2].set_ylabel("persistence")
                        axs[2].set_ylim(0.0, 1.05)
                        axs[2].grid(True, alpha=0.15)
                        axs[2].set_xlabel("time (rolling)")

                        _ = fig2.suptitle(f"Focused candidate: {float(ev_freq[-1]):.1f} Hz   state: {ev_state['value']}")
                        fig2.tight_layout()
                        live_evidence.pyplot(fig2)
                        plt.close(fig2)

                        if ev_log:
                            live_events.dataframe(ev_log, width="stretch", height=180)
            elif stage == "refine":
                cf = payload.get("candidate_freq_hz")
                bf = payload.get("best_freq_hz")
                gi = payload.get("grid_index")
                gs = payload.get("grid_size")
                if isinstance(cf, (int, float)) and isinstance(bf, (int, float)):
                    live_info.markdown(
                        f"**Stage:** refine  \n**Progress:** {frac_c*100:.1f}%  \n**ETA:** {eta:.1f}s  \n**Candidate:** {float(cf):.2f} Hz → **best so far** {float(bf):.2f} Hz  \n**Grid:** {gi}/{gs}"
                    )
            elif stage == "cancel":
                cr = payload.get("cancel_rms")
                if isinstance(cr, (int, float)):
                    cancel_hist.append(float(cr))
                    if len(cancel_hist) > 500:
                        del cancel_hist[: len(cancel_hist) - 500]
                    df = pd.DataFrame({"cancel_rms": cancel_hist})
                    live_cancel_line.line_chart(df, height=160)

                af = payload.get("artifact_freqs_hz")
                pr = payload.get("per_artifact_rms")
                if isinstance(af, list) and isinstance(pr, list) and len(af) == len(pr) and af:
                    df2 = pd.DataFrame({"freq_hz": af, "rms": pr})
                    df2["freq_hz"] = df2["freq_hz"].astype(float)
                    df2["rms"] = df2["rms"].astype(float)
                    df2 = df2.sort_values("rms", ascending=False)
                    df2 = df2.rename(columns={"freq_hz": "frequency_hz"})
                    live_cancel_table.dataframe(df2.head(12), width="stretch", height=180)

                xp = payload.get("x_preview")
                cp = payload.get("cancel_preview")
                yp = payload.get("y_preview")
                stride = payload.get("preview_stride")
                if isinstance(xp, list) and isinstance(cp, list) and isinstance(yp, list) and len(xp) == len(cp) == len(yp) and len(xp) >= 8:
                    eff_sr = float(sr) / float(stride) if isinstance(stride, int) and stride > 0 else float(sr)
                    tt = np.arange(len(xp), dtype=np.float64) / eff_sr
                    fig, ax = plt.subplots(figsize=(10, 2.8))
                    _ = ax.plot(tt, xp, label="input (preview)", linewidth=1.0, alpha=0.9)
                    _ = ax.plot(tt, cp, label="cancel signal (preview)", linewidth=1.0, alpha=0.9)
                    _ = ax.plot(tt, yp, label="output (preview)", linewidth=1.0, alpha=0.9)
                    _ = ax.set_title("Live cancel preview (downsampled): input vs cancel vs output")
                    _ = ax.set_xlabel("time (s)")
                    _ = ax.set_ylabel("amplitude")
                    ax.grid(True, alpha=0.2)
                    _ = ax.legend(loc="upper right")
                    fig.tight_layout()
                    live_wave.pyplot(fig)
                    plt.close(fig)
            elif stage == "select":
                chosen = payload.get("chosen_idx")
                if isinstance(chosen, list):
                    selected_idx.clear()
                    for v in chosen:
                        if isinstance(v, int):
                            selected_idx.append(v)

        manual = parse_freq_list(manual_freqs)
        if manual:
            artifacts = [
                ArtifactProfile(
                    artifact_type=ArtifactType.WHISTLING,
                    frequency=float(f),
                    frequency_range=(float(f - 6.0), float(f + 6.0)),
                    strength=1.0,
                    persistence=1.0,
                    phase_coherence=1.0,
                    multires_support=0,
                )
                for f in manual
            ]
        else:
            artifacts = engine.analyze_artifacts(x_mono, cfg=cfg, progress_cb=progress_cb)

        artifact_freqs = [float(a.frequency) for a in artifacts if a.frequency]
        if not artifact_freqs:
            st.warning("No coherent stationary tones detected with current thresholds.")
            return

        st.subheader("Detected artifacts")
        rows = []
        for a in artifacts:
            if not a.frequency:
                continue
            rows.append(
                {
                    "frequency_hz": float(a.frequency),
                    "persistence": float(getattr(a, "persistence", 0.0)),
                    "phase_coherence": float(getattr(a, "phase_coherence", 0.0)),
                    "multires_support": int(getattr(a, "multires_support", 1)),
                    "strength": float(a.strength),
                }
            )
        st.dataframe(rows, width="stretch")

        def _fmt_freq(v: float | None) -> str:
            return "None" if v is None else f"{v:.2f} Hz"

        selected_freq = st.selectbox(
            "Highlight a detected frequency in plots",
            options=[None] + sorted(artifact_freqs),
            format_func=_fmt_freq,
        )

        t0 = time.perf_counter()
        if channels == 1:
            y_coh, _ = engine.cleanup_resonant_subtraction(x_mono, cfg=cfg, progress_cb=progress_cb)
        else:
            ys = []
            for ch in range(channels):
                ys.append(engine.cleanup_resonant_subtraction(x[:, ch], cfg=cfg, progress_cb=progress_cb)[0])
            y_coh = np.stack(ys, axis=1).astype(np.float32)
        t1 = time.perf_counter()
        coh_runtime = float(t1 - t0)

        live_progress.progress(1.0, text="Done")

        coh_block = method_metrics(
            x_in=x_mono,
            x_out=to_mono_mix(y_coh),
            sr=int(sr),
            artifacts=artifacts[: int(max_artifacts)],
            bandwidth_hz=float(cfg.bandwidth_hz),
            runtime_sec=coh_runtime,
        )

        report = build_report_v1(
            input_meta={
                "path": str(getattr(uploaded, "name", "uploaded.wav")),
                "sample_rate": int(sr),
                "num_samples": int(len(x)),
                "duration_sec": float(len(x) / sr),
            },
            config={
                "detector": {
                    "freq_range_hz": [float(cfg.freq_range_hz[0]), float(cfg.freq_range_hz[1])],
                    "num_nodes": int(cfg.num_nodes),
                    "bandwidth_hz": float(cfg.bandwidth_hz),
                    "presence_db": float(cfg.presence_db),
                    "min_persistence": float(cfg.min_persistence),
                    "min_phase_coherence": float(cfg.min_phase_coherence),
                    "adaptive_refine": bool(cfg.adaptive_refine),
                    "refine_top_k": int(cfg.refine_top_k),
                    "refine_range_hz": float(cfg.refine_range_hz),
                    "refine_steps": int(cfg.refine_steps),
                    "inhibit_radius_hz": float(cfg.inhibit_radius_hz),
                    "inhibit_strength": float(cfg.inhibit_strength),
                },
                "method": {"max_artifacts": int(cfg.max_artifacts), "strength_multiplier": float(cfg.strength_multiplier)},
            },
            artifacts=artifacts[: int(max_artifacts)],
            methods={"resonant": coh_block},
        )

        st.subheader("Telemetry")
        m1 = report["results"]["methods"]["resonant"]["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Realtime factor", f"{m1['realtime_factor']:.1f}×")
        c2.metric("Flatness Δ", f"{m1['spectral_flatness_change']:+.4f}")
        c3.metric("Transient env corr", f"{m1['transient_env_corr']:.3f}")

        st.subheader("Spectrum inspector")
        fig = plot_psd_overlay(x_mono, to_mono_mix(y_coh), sr, artifact_freqs, selected_freq)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Per‑artifact reduction (dB)")
        coh_pa = report["results"]["methods"]["resonant"]["per_artifact"]
        rows2 = []
        for fkey in sorted(set(coh_pa.keys()), key=lambda s: float(s)):
            rows2.append({"frequency_hz": float(fkey), "reduction_db": float(coh_pa.get(fkey, {}).get("reduction_db", 0.0))})
        st.dataframe(rows2, width="stretch")

    # Left-side outputs after computation
    with left:
        st.subheader("Output (resonant)")
        st.audio(float_to_wav_bytes(y_coh, sr, reference_peak=ref_peak), format="audio/wav")
        st.download_button(
            "Download cleaned WAV",
            data=float_to_wav_bytes(y_coh, sr, reference_peak=ref_peak),
            file_name="cleaned_resonant.wav",
            mime="audio/wav",
        )

        with st.expander("Band inspector (listen)", expanded=False):
            st.caption("Listen to a narrow band around a frequency and the removed component (input band − output band).")
            listen_freq = selected_freq if selected_freq is not None else (artifact_freqs[0] if artifact_freqs else None)
            if listen_freq is not None:
                listen_bw = st.slider(
                    "Listen bandwidth (Hz)",
                    20.0,
                    800.0,
                    max(100.0, float(bandwidth_hz) * 4.0),
                    10.0,
                    help="Bandpass width used only for listening/inspection (does not change cleanup).",
                )

                bp_in = bandpass_audio(x, int(sr), float(listen_freq), float(listen_bw))
                bp_coh = bandpass_audio(y_coh, int(sr), float(listen_freq), float(listen_bw))
                rm_coh = bp_in - bp_coh

                ref_bp_peak = (
                    float(np.max(np.abs(np.nan_to_num(bp_in.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))))
                    if bp_in.size
                    else 0.0
                )
                st.markdown("**Bandpassed input**")
                st.audio(float_to_wav_bytes(bp_in, int(sr), reference_peak=ref_bp_peak), format="audio/wav")
                st.markdown("**Bandpassed output**")
                st.audio(float_to_wav_bytes(bp_coh, int(sr), reference_peak=ref_bp_peak), format="audio/wav")
                st.markdown("**Removed component (band)**")
                st.audio(float_to_wav_bytes(rm_coh, int(sr), reference_peak=ref_bp_peak), format="audio/wav")

        st.download_button(
            "Download report JSON",
            data=json.dumps(report, indent=2, sort_keys=True).encode("utf-8"),
            file_name="cleanup_report.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()

