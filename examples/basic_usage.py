"""
Basic usage example for the SynthNN framework.

This script demonstrates:
1. Creating a resonant network
2. Encoding different data types
3. Processing signals
4. Decoding to different outputs
"""

import numpy as np
import matplotlib.pyplot as plt
from synthnn.core import (
    ResonantNode,
    ResonantNetwork,
    SignalProcessor,
    UniversalPatternCodec
)


def demo_text_to_audio():
    """Convert text to audio through resonant patterns."""
    print("\n=== Text to Audio Demo ===")
    
    # Create network and codec
    network = ResonantNetwork(name="text_to_audio")
    codec = UniversalPatternCodec()
    
    # Encode text
    text = "Hello, Resonant World!"
    print(f"Input text: {text}")
    
    node_params = codec.encode(text, 'text')
    
    # Add nodes to network
    for node_id, params in node_params.items():
        node = ResonantNode(
            node_id=node_id,
            frequency=params['frequency'],
            phase=params['phase'],
            amplitude=params['amplitude']
        )
        network.add_node(node)
    
    # Create connections (simple chain)
    node_ids = list(node_params.keys())
    for i in range(len(node_ids) - 1):
        network.connect(node_ids[i], node_ids[i+1], weight=0.5)
    
    # Evolve network
    dt = 0.01
    for _ in range(100):
        network.step(dt)
    
    # Decode to audio
    audio = codec.decode(network, 'audio')
    print(f"Generated audio shape: {audio.shape}")
    
    return network, audio


def demo_audio_analysis():
    """Analyze audio signal using resonant network."""
    print("\n=== Audio Analysis Demo ===")
    
    # Generate test signal
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Complex signal with multiple frequencies
    signal = (
        np.sin(2 * np.pi * 440 * t) +  # A4
        0.5 * np.sin(2 * np.pi * 554.37 * t) +  # C#5
        0.3 * np.sin(2 * np.pi * 659.25 * t)  # E5
    )
    
    # Process signal
    processor = SignalProcessor(sample_rate)
    
    # Extract features
    fundamental = processor.extract_fundamental(signal, freq_range=(400, 700))
    centroid = processor.compute_spectral_centroid(signal)
    zcr = processor.compute_zero_crossing_rate(signal)
    
    print(f"Fundamental frequency: {fundamental:.2f} Hz")
    print(f"Spectral centroid: {centroid:.2f} Hz")
    print(f"Zero-crossing rate: {zcr:.2f} Hz")
    
    # Encode to network
    codec = UniversalPatternCodec()
    network = ResonantNetwork(name="audio_analysis")
    
    node_params = codec.encode(signal[:4410], 'audio')  # Use 0.1s of signal
    
    for node_id, params in node_params.items():
        node = ResonantNode(
            node_id=node_id,
            frequency=params['frequency'],
            phase=params['phase'],
            amplitude=params['amplitude']
        )
        network.add_node(node)
    
    # Measure synchronization
    sync = network.measure_synchronization()
    print(f"Network synchronization: {sync:.3f}")
    
    return signal, processor


def demo_image_processing():
    """Process image through resonant patterns."""
    print("\n=== Image Processing Demo ===")
    
    # Create synthetic image (Gaussian blob)
    size = 64
    x, y = np.mgrid[0:size, 0:size]
    center = size // 2
    sigma = size / 6
    image = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    
    # Encode image
    codec = UniversalPatternCodec()
    network = ResonantNetwork(name="image_processing")
    
    node_params = codec.encode(image, 'image')
    
    # Create network
    for node_id, params in node_params.items():
        node = ResonantNode(
            node_id=node_id,
            frequency=params['frequency'],
            phase=params['phase'],
            amplitude=params['amplitude']
        )
        network.add_node(node)
    
    # Add spatial connections (grid topology)
    grid_size = codec.image_encoder.grid_size
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            current = f"img_{i}_{j}"
            # Connect to neighbors
            if i < grid_size[0] - 1:
                network.connect(current, f"img_{i+1}_{j}", weight=0.3)
            if j < grid_size[1] - 1:
                network.connect(current, f"img_{i}_{j+1}", weight=0.3)
    
    # Process
    for _ in range(50):
        network.step(0.01)
    
    # Decode
    reconstructed = codec.decode(network, 'image')
    
    print(f"Original image shape: {image.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    return image, reconstructed, network


def demo_network_adaptation():
    """Demonstrate network adaptation and learning."""
    print("\n=== Network Adaptation Demo ===")
    
    # Create network with random nodes
    network = ResonantNetwork(name="adaptive")
    
    # Add nodes with random frequencies
    for i in range(10):
        node = ResonantNode(
            node_id=f"adaptive_{i}",
            frequency=np.random.uniform(1, 5),
            phase=np.random.uniform(0, 2*np.pi),
            amplitude=1.0
        )
        network.add_node(node)
    
    # Fully connected topology
    node_ids = list(network.nodes.keys())
    for i, src in enumerate(node_ids):
        for j, tgt in enumerate(node_ids):
            if i != j:
                network.connect(src, tgt, weight=np.random.uniform(-0.5, 0.5))
    
    # Track synchronization over time
    sync_history = []
    
    # Adapt network
    for step in range(200):
        network.step(0.01)
        
        # Measure and record sync
        sync = network.measure_synchronization()
        sync_history.append(sync)
        
        # Adapt connections every 10 steps
        if step % 10 == 0:
            network.adapt_connections(target_sync=0.8)
    
    print(f"Initial sync: {sync_history[0]:.3f}")
    print(f"Final sync: {sync_history[-1]:.3f}")
    
    return sync_history


def visualize_results():
    """Create visualizations of the demos."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Text to Audio visualization
    ax = axes[0, 0]
    network, audio = demo_text_to_audio()
    ax.plot(audio[:1000])
    ax.set_title("Text to Audio Output")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    
    # Audio Analysis visualization
    ax = axes[0, 1]
    signal, processor = demo_audio_analysis()
    freqs, mags = processor.analyze_spectrum(signal[:4410])
    ax.semilogy(freqs[:1000], mags[:1000])
    ax.set_title("Audio Spectrum Analysis")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    
    # Image Processing visualization
    ax = axes[1, 0]
    original, reconstructed, _ = demo_image_processing()
    # Show original and reconstructed side by side
    combined = np.hstack([original, reconstructed])
    ax.imshow(combined, cmap='viridis')
    ax.set_title("Image: Original (left) vs Reconstructed (right)")
    ax.axis('off')
    
    # Network Adaptation visualization
    ax = axes[1, 1]
    sync_history = demo_network_adaptation()
    ax.plot(sync_history)
    ax.axhline(y=0.8, color='r', linestyle='--', label='Target')
    ax.set_title("Network Synchronization Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Synchronization")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('synthnn_demo_results.png')
    plt.show()


if __name__ == "__main__":
    print("SynthNN Framework - Basic Usage Examples")
    print("=" * 50)
    
    # Run all demos
    visualize_results()
    
    print("\n" + "=" * 50)
    print("Demos complete! Results saved to synthnn_demo_results.png") 