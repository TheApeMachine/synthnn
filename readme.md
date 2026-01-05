# SynthNN (current focus): resonant-network coherence filter (audio cleanup)

SynthNN contains a working **coherence filter** for canceling common “whistling / ringing” artifacts in AI-generated music (e.g. Suno-style outputs).

The core idea is simple and inspectable:

- **Detect** coherent stationary tones using a **resonant bank** (lock‑in style):
  - **persistence** over time (node stays “present”)
  - **phase-coherence** (node phase stays stable)
  - optional **adaptive refinement** (retune candidates toward local maxima)
  - optional **lateral inhibition** (avoid selecting many nearby peaks)
- **Correct** by **re-synthesizing coherent components from node state and subtracting**
- **Report everything** in a stable, versioned JSON schema (`cleanup_report_v1`)

This is not a generic denoiser. It targets **structural artifacts** (unwanted coherence) and is designed to minimize collateral damage.

---

## Install

```bash
pip install -e .
```

### Optional: PyAudio (realtime I/O)

PyAudio requires PortAudio headers.

- macOS:

```bash
brew install portaudio
pip install -e ".[music]"
```

---

## CLI (WAV → cleaned WAV + report + plot)

```bash
synthnn-clean -i input.wav -o cleaned.wav --adaptive-refine
```

Outputs:

- `cleaned.wav`
- `cleaned.report.json` (schema: `cleanup_report_v1`)
- `cleaned.spectrum.png`

---

## Streamlit demo (inspectable “tactile” UI)

```bash
streamlit run demos/streamlit_app.py
```

The app lets you upload a WAV and inspect:

- detected artifacts (frequency, persistence, phase coherence, multi-res support)
- Audio output + download
- PSD overlay with artifact markers (select to highlight)
- metrics: per-artifact narrowband reduction, spectral flatness delta, transient proxy, runtime factor
- report JSON download

---

## Report schema

The JSON report schema is standardized in `synthnn/core/cleanup_report.py`:

- `schema_version`: `cleanup_report_v1`
- `artifacts`: per-artifact evidence + confidence
- `summary.cleanup_severity`: evidence-weighted scalar in \([0, 1]\) summarizing “how artifacty” the input looks
- `results.methods`: method blocks with metrics + per-artifact bandpower reductions

---

## Limitations / expectations

- Input must be **WAV** for the demo/CLI.
- This targets **stationary coherent tones**; it will not remove broadband noise well.
- This repo is intentionally **resonant-only** (no DSP notch-baseline mode).

