# SynthNN: Resonant Neural Networks

_An AI Framework Based on Wave Physics and Resonance_

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green.svg)](https://github.com/theapemachine/synthnn)

---

## 🌊 What is SynthNN?

SynthNN represents a **paradigm shift in artificial intelligence** - moving from traditional matrix-based neural networks to **resonant networks** inspired by wave physics, neuroscience, and musical harmony. Instead of discrete computational layers, SynthNN uses **continuously oscillating nodes** that communicate through **phase coupling** and **frequency synchronization**.

Think of it as **neural networks that sing**.

### ⚡ Key Breakthrough: Resonant Computation

```python
# Traditional Neural Network
output = activation(weights @ inputs + bias)

# SynthNN Resonant Network
output = Σ(amplitude_i * sin(2π * frequency_i * time + phase_i))
# Where phases couple through: phase_coupling = Σ(weight_ij * sin(phase_j - phase_i))
```

---

## 🚀 Features

### 🧠 **Biologically Plausible Architecture**

- Mimics **neural oscillations** found in real brains
- **Phase synchronization** for information binding
- **Continuous learning** without discrete training phases
- **Emergent computation** from simple resonant interactions

### ⚡ **High-Performance Computing**

- **GPU Acceleration**: CUDA (NVIDIA) and Metal (Apple Silicon) support
- **10-16x performance improvements** over CPU-only implementations
- **Automatic backend selection** and optimization
- **Real-time processing** capabilities

### 🎵 **Multi-Modal Intelligence**

- **Universal Pattern Codec**: Seamlessly translate between audio, text, and images
- **Cross-modal learning**: Train on music, generate images, understand text
- **Microtonal music systems**: Advanced tuning and cultural music traditions
- **Emotional resonance mapping**: AI that understands and generates emotions

### 🌐 **Novel Applications**

- **Adaptive music generation** with mode detection and real-time retuning
- **Signal processing** without FFT artifacts
- **Pattern recognition** through resonance matching
- **Control systems** with emergent coordination
- **Consciousness research** through global synchronization studies

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        SynthNN Core                         │
├─────────────────────────────────────────────────────────────┤
│  ResonantNode: Oscillating computational units              │
│  ├─ Frequency, Phase, Amplitude                             │
│  ├─ Damping and Energy dynamics                             │
│  └─ Adaptive retuning capabilities                          │
│                                                             │
│  ResonantNetwork: Phase-coupled node collections            │
│  ├─ Kuramoto-style synchronization                          │
│  ├─ Connection weights and delays                           │
│  ├─ Emergent behavior analysis                              │
│  └─ NetworkX integration for graph analysis                 │
│                                                             │
│  UniversalPatternCodec: Multi-modal translation             │
│  ├─ Audio ↔ Resonant patterns                               │
│  ├─ Text ↔ Frequency mappings                               │
│  ├─ Images ↔ Spatial resonance                              │
│  └─ Cross-modal learning                                    │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Performance Layer                         │
├─────────────────────────────────────────────────────────────┤
│  Backend Manager: Automatic acceleration                    │
│  ├─ CPU: Optimized NumPy/SciPy                              │
│  ├─ CUDA: NVIDIA GPU acceleration via CuPy/PyTorch          │
│  ├─ Metal: Apple Silicon acceleration via MLX               │
│  └─ Automatic device detection and selection                │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                 Application Modules                         │
├─────────────────────────────────────────────────────────────┤
│  Musical Extensions: Advanced music generation              │
│  ├─ Mode detection and adaptive retuning                    │
│  ├─ Microtonal systems and cultural scales                  │
│  ├─ Chord progressions and harmonic analysis                │
│  └─ Real-time interactive performance                       │
│                                                             │
│  Emotional Resonance: Emotion-aware AI                      │
│  ├─ Emotion-to-frequency mapping                            │
│  ├─ Empathetic response generation                          │
│  ├─ Cross-cultural emotion recognition                      │
│  └─ Therapeutic soundscape creation                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install synthnn

# With GPU acceleration (NVIDIA)
pip install synthnn[cuda]

# With Apple Silicon acceleration
pip install synthnn[metal]

# Development installation
git clone https://github.com/theapemachine/synthnn.git
cd synthnn
pip install -e .
```

### Your First Resonant Network

```python
from synthnn.core import ResonantNetwork, ResonantNode
import numpy as np

# Create a network
network = ResonantNetwork(name="my_first_network")

# Add resonant nodes with different frequencies
for i, freq in enumerate([110, 165, 220, 330]):  # A2 harmonic series
    node = ResonantNode(
        node_id=f"harmonic_{i}",
        frequency=freq,
        amplitude=1.0 / len([110, 165, 220, 330]),
        phase=np.random.uniform(0, 2*np.pi)
    )
    network.add_node(node)

# Connect nodes with phase coupling
node_ids = list(network.nodes.keys())
for i in range(len(node_ids) - 1):
    network.connect(node_ids[i], node_ids[i+1], weight=0.5)

# Generate audio
duration = 3.0  # seconds
sample_rate = 44100
audio = network.generate_signals(duration, sample_rate)

print(f"Generated {len(audio)} audio samples")
print(f"Network synchronization: {network.measure_synchronization():.3f}")
```

### GPU-Accelerated Music Generation

```python
from synthnn.core import AcceleratedMusicalNetwork
from synthnn.performance import BackendType

# Create accelerated musical network
network = AcceleratedMusicalNetwork(
    name="gpu_music_network",
    backend=BackendType.METAL,  # or CUDA for NVIDIA
    base_freq=440.0,  # A4
    mode="Dorian"
)

# Create harmonic nodes
network.create_harmonic_nodes([1, 1.2, 1.5, 1.8, 2.0, 2.4])

# Generate a chord progression
chords = [
    [1.0, 1.25, 1.5],     # Major triad
    [1.0, 1.2, 1.5],      # Minor triad
    [1.0, 1.25, 1.5, 1.875], # Major 7th
    [1.0, 1.5, 2.0]       # Perfect fifth octave
]

audio = network.generate_chord_progression(chords, duration_per_chord=1.5)
```

---

## 🎮 Interactive Demos

### Launch the Streamlit App

```bash
cd synthnn
streamlit run demos/streamlit_app.py
```

**Features:**

- 🎵 **Real-time music generation** with parameter control
- 📊 **Network visualization** and synchronization analysis
- 🌍 **Microtonal exploration** with cultural scales
- 👁️‍🗨️ **Multi-modal playground** for cross-modal translation
- 💖 **Emotional resonance engine** for emotion-aware AI

### Command Line Demos

```bash
# Basic resonant network demonstration
python main.py --examples

# Advanced music generation
python main.py --music

# GPU-accelerated performance demo
python main.py --accelerated

# Interactive Python shell with SynthNN loaded
python main.py --shell
```

---

## 🎯 Real-World Applications

### 🎵 **Music & Audio**

- **Adaptive composition**: AI that responds to musical context
- **Microtonal systems**: Support for diverse cultural music traditions
- **Real-time performance**: Live electronic music with AI accompaniment
- **Audio restoration**: Signal processing without FFT artifacts

### 🧠 **AI Research**

- **Consciousness studies**: Global synchronization as awareness model
- **Continuous learning**: No discrete training/inference phases
- **Emergent behavior**: Complex patterns from simple interactions
- **Cross-modal AI**: Unified understanding across modalities

### 🔧 **Engineering**

- **Adaptive control**: Swarm robotics and distributed systems
- **Signal processing**: Real-time analysis without windowing artifacts
- **Pattern recognition**: Resonance-based matching and classification
- **Data compression**: Perceptually-aware lossy compression

### 🌐 **Innovative Domains**

- **Therapeutic applications**: Personalized healing soundscapes
- **Quantum computing interfaces**: Bridge classical-quantum systems
- **Neuromorphic hardware**: Direct implementation on oscillator chips
- **Environmental monitoring**: Resonance-based sensor networks

---

## 📊 Performance Benchmarks

| Operation                | CPU (Intel i7) | NVIDIA RTX 4090 | Apple M2 Ultra |
| ------------------------ | -------------- | --------------- | -------------- |
| 1000-node network step   | 45ms           | 3ms (15x)       | 4ms (11x)      |
| Audio generation (10s)   | 2.1s           | 0.15s (14x)     | 0.19s (11x)    |
| Cross-modal translation  | 890ms          | 67ms (13x)      | 71ms (12x)     |
| Synchronization analysis | 120ms          | 12ms (10x)      | 15ms (8x)      |

_Benchmarks on networks with 1000+ nodes generating 44.1kHz audio_

---

## 📚 Documentation & Examples

### Core Examples

- **[Basic Usage](examples/basic_usage.py)**: Fundamental network operations
- **[Performance Demo](examples/performance_demo.py)**: GPU acceleration showcase
- **[Multimodal Demo](examples/multimodal_demo.py)**: Cross-modal AI capabilities
- **[Music Migration](examples/music_migration_demo.py)**: Upgrading existing music systems

### Advanced Applications

- **[Microtonal Systems](examples/microtonal_demo.py)**: Cultural music scales
- **[Emotional AI](examples/emotional_resonance_demo.py)**: Emotion-aware systems
- **[Adaptive Learning](examples/adaptive_learning_demo.py)**: Continuous learning
- **[Consciousness Studies](examples/consciousness_demo.py)**: Awareness modeling

### API Documentation

```bash
# Generate comprehensive API docs
cd docs/
make html
open _build/html/index.html
```

---

## 🔬 Scientific Foundations

SynthNN is grounded in rigorous scientific principles:

### **Neuroscience**

- **Neural oscillations**: Gamma, theta, alpha wave modeling
- **Phase synchronization**: Information binding mechanisms
- **Critical dynamics**: Edge-of-chaos computation
- **Plasticity**: Continuous adaptation and learning

### **Physics**

- **Kuramoto model**: Phase-coupled oscillator dynamics
- **Wave interference**: Constructive/destructive pattern formation
- **Resonance theory**: Natural frequency matching and amplification
- **Non-linear dynamics**: Emergent behavior and bifurcations

### **Music Theory**

- **Harmonic series**: Natural frequency relationships
- **Microtonal systems**: Cultural and mathematical tuning systems
- **Mode theory**: Tonal centers and modal characteristics
- **Emotional mapping**: Psychoacoustic frequency-emotion relationships

---

## 🛠️ Development & Contributing

### Development Setup

```bash
git clone https://github.com/theapemachine/synthnn.git
cd synthnn

# Create virtual environment
python -m venv synthnn-dev
source synthnn-dev/bin/activate  # On Windows: synthnn-dev\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,cuda,metal,all-backends]"

# Run tests
pytest tests/ -v

# Run performance benchmarks
python examples/performance_demo.py
```

### Contributing Guidelines

We welcome contributions! Areas where help is especially valuable:

- 🧠 **Neuroscience integration**: EEG/brain signal processing
- 🎵 **Music theory**: Advanced harmonic analysis and generation
- ⚡ **Performance optimization**: GPU kernels and parallel algorithms
- 🌐 **Applications**: Novel use cases and domain-specific extensions
- 📖 **Documentation**: Tutorials, examples, and theoretical explanations

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 🔮 Vision & Future Directions

### **Near-term Goals (2024-2025)**

- [ ] **Neuromorphic hardware** integration with oscillator-based chips
- [ ] **Quantum extensions** for quantum-classical hybrid systems
- [ ] **Real-time VST plugins** for music production
- [ ] **Mobile deployment** with CoreML/TensorFlow Lite optimization

### **Medium-term Vision (2025-2027)**

- [ ] **Swarm intelligence** with distributed resonant consensus
- [ ] **Brain-computer interfaces** using EEG synchronization
- [ ] **Therapeutic applications** with personalized healing protocols
- [ ] **Educational platforms** for interactive physics/music learning

### **Long-term Impact (2027+)**

- [ ] **Artificial consciousness** through global synchronization studies
- [ ] **Universal translation** between any sensory modalities
- [ ] **Ecosystem monitoring** via environmental resonance networks
