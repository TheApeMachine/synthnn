# SynthNN Interactive Demo App

An interactive Streamlit application showcasing the capabilities of Synthetic Resonant Neural Networks (SynthNN) for music generation, pattern visualization, and real-time interaction.

## Features

### 🎵 Music Generation

- Generate unique audio patterns from resonant neural networks
- Adjustable duration and real-time playback
- Download generated audio as WAV files
- Waveform visualization

### 📊 Network Visualization

- Interactive network graph showing nodes and connections
- Real-time phase distribution on polar plots
- Network synchronization monitoring
- Step-by-step simulation controls

### 🔬 Audio Analysis

- Frequency spectrum analysis with peak detection
- Fundamental frequency extraction
- Phase coherence evolution over time
- Network metrics display

### 🎮 Interactive Control

- Apply external stimuli to specific nodes
- Three stimulus types: Impulse, Continuous, Rhythmic
- Real-time parameter adjustment
- Observe network response to perturbations

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
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage Guide

1. **Initialize Network**: Use the sidebar controls to configure network parameters

   - Number of nodes (3-50)
   - Frequency range for oscillators
   - Connection density between nodes
   - Coupling strength and damping

2. **Choose Backend**: Select from available compute backends

   - CPU (always available)
   - CUDA (if NVIDIA GPU available)
   - Metal (if on Apple Silicon)

3. **Load Presets**: Quick-start with predefined configurations

   - Harmonic Series: Natural harmonics
   - Pentatonic: Musical pentatonic scale
   - Chaotic: Random frequency distribution
   - Bell-like: Bell sound synthesis

4. **Generate Music**: Click "Generate Audio" to create sounds

   - Adjust duration (0.5-5 seconds)
   - Listen with built-in player
   - Download as WAV file

5. **Visualize**: Explore the network structure and dynamics

   - Network graph with weighted connections
   - Phase distribution on polar plot
   - Synchronization evolution

6. **Analyze**: Understand the generated audio

   - Frequency spectrum with peaks
   - Phase coherence metrics
   - Fundamental frequency detection

7. **Interact**: Apply stimuli and observe responses
   - Target specific nodes
   - Choose stimulus patterns
   - Adjust strength and observe effects

## Backend Performance

The app automatically detects available backends:

- **CPU**: Universal, uses optimized NumPy
- **Metal**: Apple Silicon GPU acceleration (M1/M2/M3)
- **CUDA**: NVIDIA GPU acceleration

Performance scales with network size:

- Small networks (10 nodes): Real-time on CPU
- Medium networks (50 nodes): Benefits from GPU
- Large networks (100+ nodes): Requires GPU for smooth operation

## Tips

- Start with presets to understand network behavior
- Lower coupling strength for more independent oscillators
- Higher coupling leads to synchronization
- Connection density affects emergence of patterns
- External stimuli can trigger interesting transitions

## Troubleshooting

- **Audio not playing**: Check browser audio permissions
- **Slow performance**: Try reducing number of nodes or use GPU backend
- **Import errors**: Ensure all dependencies are installed
- **Backend not available**: Install appropriate GPU libraries (cupy/mlx/torch)

## Examples

### Creating a Harmonic Drone

1. Load "Harmonic Series" preset
2. Set coupling strength to 0.5
3. Generate 3-second audio
4. Observe partial synchronization in spectrum

### Chaotic Patterns

1. Load "Chaotic" preset
2. Set high coupling (5.0)
3. Apply rhythmic stimulus
4. Watch synchronization emerge

### Bell Synthesis

1. Load "Bell-like" preset
2. Apply single impulse
3. Listen to natural decay
4. Analyze inharmonic spectrum
