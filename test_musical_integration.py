#!/usr/bin/env python3
"""
Test script to verify musical extensions and integration work correctly.
"""

import numpy as np
import time
from synthnn.core import MusicalResonantNetwork, AcceleratedMusicalNetwork
from synthnn.performance import BackendManager
from detector import ModeDetector


def test_musical_network():
    """Test basic MusicalResonantNetwork functionality."""
    print("Testing MusicalResonantNetwork...")
    
    # Create network
    mode_detector = ModeDetector()
    network = MusicalResonantNetwork(
        base_freq=440.0,
        mode="Ionian",
        mode_detector=mode_detector
    )
    
    # Create harmonic nodes
    network.create_harmonic_nodes([1, 1.5, 2, 2.5, 3])
    print(f"✓ Created {len(network.nodes)} harmonic nodes")
    
    # Create connections
    network.create_modal_connections("harmonic_series")
    print(f"✓ Created {len(network.connections)} connections")
    
    # Generate audio
    audio = network.compute_harmonic_state(0.5, 44100)
    print(f"✓ Generated audio: {audio.shape}")
    
    # Test mode detection
    test_signal = np.sin(2 * np.pi * 523.25 * np.linspace(0, 1, 44100))  # C5
    detected_mode = network.analyze_and_retune(test_signal)
    print(f"✓ Detected mode: {detected_mode}")
    
    return True


def test_accelerated_network():
    """Test AcceleratedMusicalNetwork with GPU/Metal support."""
    print("\nTesting AcceleratedMusicalNetwork...")
    
    # Check available backends
    manager = BackendManager()
    backends = manager.list_available_backends()
    print(f"Available backends: {[b.value for b in backends]}")
    
    # Create accelerated network
    mode_detector = ModeDetector()
    network = AcceleratedMusicalNetwork(
        base_freq=440.0,
        mode="Ionian",
        mode_detector=mode_detector
    )
    
    # Get backend info
    stats = network.get_performance_stats()
    print(f"✓ Using backend: {stats['backend']}")
    
    # Create nodes and test acceleration
    network.create_harmonic_nodes([1, 1.25, 1.5, 1.75, 2])
    network.create_modal_connections("nearest_neighbor")
    
    # Benchmark audio generation
    start = time.time()
    audio = network.generate_audio_accelerated(1.0, 44100)
    elapsed = time.time() - start
    print(f"✓ Generated 1s audio in {elapsed:.3f}s")
    
    # Test advanced features
    print("\nTesting advanced features...")
    
    # 1. Chord progression
    chords = [[1, 1.25, 1.5], [1.125, 1.4, 1.68]]
    progression = network.generate_chord_progression(chords, 0.5)
    print(f"✓ Generated chord progression: {progression.shape}")
    
    # 2. Mode morphing
    morph = network.morph_between_modes_accelerated("Dorian", morph_time=0.5)
    print(f"✓ Morphed to Dorian: {morph.shape}")
    
    # 3. Batch processing
    signals = [np.sin(2 * np.pi * f * np.linspace(0, 0.1, 4410)) 
               for f in [440, 523.25, 659.25]]
    results = network.batch_process_signals(signals, "analyze")
    print(f"✓ Batch processed {len(results)} signals")
    
    return True


def test_integration():
    """Test integration between old and new systems."""
    print("\nTesting integration...")
    
    # Import both old and new
    from abstract import ResonantNetwork as OldNetwork
    
    # Create similar networks
    mode_detector = ModeDetector()
    
    # Old network
    old_net = OldNetwork(
        num_nodes=5,
        base_freq=440.0,
        harmonic_ratios=[1, 1.5, 2, 2.5, 3],
        target_phase=0.0,
        target_amplitude=1.0
    )
    
    # New network
    new_net = MusicalResonantNetwork(
        base_freq=440.0,
        mode="Ionian",
        mode_detector=mode_detector
    )
    new_net.create_harmonic_nodes([1, 1.5, 2, 2.5, 3])
    
    # Compare node frequencies
    old_freqs = [node.freq for node in old_net.nodes]
    new_freqs = [node.frequency for node in new_net.nodes.values()]
    
    print(f"Old frequencies: {old_freqs}")
    print(f"New frequencies: {new_freqs}")
    
    # Check compatibility
    freq_match = all(abs(o - n) < 0.01 for o, n in zip(old_freqs, new_freqs))
    print(f"✓ Frequency compatibility: {freq_match}")
    
    return freq_match


def main():
    """Run all tests."""
    print("="*60)
    print("SynthNN Musical Integration Tests")
    print("="*60)
    
    try:
        # Test basic musical network
        assert test_musical_network(), "Musical network test failed"
        
        # Test accelerated network
        assert test_accelerated_network(), "Accelerated network test failed"
        
        # Test integration
        assert test_integration(), "Integration test failed"
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 