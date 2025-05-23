# SynthNN Core Module

The core module provides the fundamental building blocks for the SynthNN framework.

## Components

### ResonantNode

The basic computational unit that oscillates at a specific frequency with phase and amplitude.

```python
from synthnn.core import ResonantNode

node = ResonantNode(
    node_id="example",
    frequency=440.0,  # Hz
    phase=0.0,        # radians
    amplitude=1.0,    # strength
    damping=0.1       # energy dissipation
)

# Get signal at time t
signal = node.oscillate(t=1.0)

# Update phase with coupling
node.update_phase(dt=0.01, phase_coupling=0.1)
```

### ResonantNetwork

A collection of interconnected ResonantNodes with emergent dynamics.

```python
from synthnn.core import ResonantNetwork, ResonantNode

# Create network
network = ResonantNetwork(name="my_network")

# Add nodes
network.add_node(ResonantNode("node1", frequency=1.0))
network.add_node(ResonantNode("node2", frequency=1.5))

# Connect nodes
network.connect("node1", "node2", weight=0.5, delay=0.0)

# Step simulation
network.step(dt=0.01)

# Measure synchronization
sync = network.measure_synchronization()
```

### SignalProcessor

Comprehensive signal analysis and transformation tools.

```python
from synthnn.core import SignalProcessor

processor = SignalProcessor(sample_rate=44100)

# Analyze spectrum
freqs, mags = processor.analyze_spectrum(signal_data)

# Extract fundamental frequency
fundamental = processor.extract_fundamental(signal_data)

# Compute phase coherence between signals
coherence = processor.compute_phase_coherence({"sig1": data1, "sig2": data2})
```

### Pattern Codecs

Encode and decode between different data types and resonant patterns.

```python
from synthnn.core import UniversalPatternCodec

codec = UniversalPatternCodec()

# Encode any data type
params = codec.encode("Hello World", data_type='text')
params = codec.encode(audio_array, data_type='audio')
params = codec.encode(image_array, data_type='image')

# Decode network state
text = codec.decode(network, output_type='text')
audio = codec.decode(network, output_type='audio')
image = codec.decode(network, output_type='image')
```

## Design Principles

1. **Modularity**: Each component is self-contained and can be used independently
2. **Type Safety**: Uses Python type hints for better IDE support and documentation
3. **Efficiency**: Optimized for numerical computations with NumPy
4. **Extensibility**: Abstract base classes allow easy extension for new data types

## Next Steps

See the `examples/` directory for complete usage examples.
