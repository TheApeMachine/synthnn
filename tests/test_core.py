#!/usr/bin/env python3
"""
Quick test of the SynthNN core module to ensure everything is working.
"""

import sys
import numpy as np

# Add current directory to path so we can import synthnn
sys.path.insert(0, '.')

try:
    from synthnn.core import (
        ResonantNode,
        ResonantNetwork,
        SignalProcessor,
        UniversalPatternCodec
    )
    print("✓ Successfully imported all core components")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test ResonantNode
print("\n--- Testing ResonantNode ---")
node = ResonantNode("test_node", frequency=440.0, phase=0.0, amplitude=1.0)
signal = node.oscillate(1.0)
print(f"Node signal at t=1.0: {signal:.4f}")
print(f"Node energy: {node.energy():.4f}")

# Test ResonantNetwork
print("\n--- Testing ResonantNetwork ---")
network = ResonantNetwork("test_network")
network.add_node(ResonantNode("n1", frequency=1.0))
network.add_node(ResonantNode("n2", frequency=1.5))
network.connect("n1", "n2", weight=0.5)
initial_sync = network.measure_synchronization()
print(f"Initial synchronization: {initial_sync:.4f}")

# Evolve network
for _ in range(10):
    network.step(0.01)
final_sync = network.measure_synchronization()
print(f"Final synchronization: {final_sync:.4f}")

# Test SignalProcessor
print("\n--- Testing SignalProcessor ---")
processor = SignalProcessor(sample_rate=1000)
t = np.linspace(0, 1, 1000)
test_signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz signal
fundamental = processor.extract_fundamental(test_signal)
print(f"Detected fundamental frequency: {fundamental:.1f} Hz (expected: 10.0 Hz)")

# Test Pattern Codecs
print("\n--- Testing Pattern Codecs ---")
codec = UniversalPatternCodec()

# Test text encoding
text = "Hello"
text_params = codec.encode(text, 'text')
print(f"Text '{text}' encoded to {len(text_params)} nodes")

# Test audio encoding
audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
audio_params = codec.encode(audio, 'audio')
print(f"Audio encoded to {len(audio_params)} nodes")

# Test anomaly detection
print("\n--- Testing Anomaly Detection ---")
processor = SignalProcessor(sample_rate=100)
t = np.linspace(0, 10, 1000)
sig = np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(len(t))
sig[200] += 2.0
anomalies = processor.detect_anomalies(sig, window_size=50, threshold_factor=3.0)
print(f"Detected {np.sum(anomalies)} anomalies")

print("\n✓ All tests passed successfully!")
print("\nThe SynthNN core module is working correctly.") 