# SynthNN Coherence Filter Demo (Streamlit)

This Streamlit app is a single-purpose **audio cleanup inspector** (resonant-only):

- Upload a WAV (e.g. Suno-style “whistling” artifacts)
- Detect **persistent, phase-coherent narrowband tones** using a **resonant bank** (lock-in style)
- Cancel coherent components by re-synthesizing from node state and subtracting
- Inspect deep telemetry, plots, and the standardized JSON report (`cleanup_report_v1`)

## What you can inspect

- **Artifact table**: frequency, persistence, phase coherence, bank support
- **Audio output**: download + listen
- **Spectrum overlay**: PSD with artifact markers (select to highlight)
- **Metrics**: per-artifact narrowband reduction, spectral flatness delta, transient proxy, runtime factor
- **Report JSON**: download and use in your own evaluation pipelines

## Running the App

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (optional):
# CUDA: pip install cupy-cuda11x
# Metal: pip install mlx
# PyTorch: pip install torch
```

### Launch the App

```bash
streamlit run demos/streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Usage

1. Upload a WAV.
2. Adjust detection thresholds if needed.
3. Click **Run detection + cleanup**.
4. Listen to both outputs and inspect the telemetry + report.

## Troubleshooting

- **Audio not playing**: Check browser audio permissions.
- **No artifacts detected**: Lower persistence/coherence thresholds or increase max artifacts.
- **Import errors**: Ensure dependencies are installed (notably `streamlit`, `numpy`, `scipy`, `matplotlib`).
