"""
Migration demonstration: From original music generation to accelerated framework.

This example shows how to migrate existing code that uses the original
abstract.py ResonantNetwork to the new accelerated musical framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import time

# Original imports (what you might have)
from abstract import ResonantNetwork as OriginalNetwork
from detector import ModeDetector

# New framework imports
from synthnn.core import AcceleratedMusicalNetwork, MusicalResonantNetwork
from synthnn.performance import BackendManager, BackendType


def original_approach():
    """
    Example using the original abstract.py implementation.
    """
    print("\n=== Original Approach ===")
    
    # Setup
    base_freq = 440.0  # A4
    harmonic_ratios = [1, 1.5, 1.33, 2, 2.5]  # Perfect fifth, fourth, octave, etc.
    time_steps = np.linspace(0, 2, 44100 * 2)  # 2 seconds at 44.1kHz
    foreign_signal = np.sin(2 * np.pi * 523.25 * time_steps)  # C5
    
    # Create network and detector
    mode_detector = ModeDetector()
    network = OriginalNetwork(
        num_nodes=len(harmonic_ratios),
        base_freq=base_freq,
        harmonic_ratios=harmonic_ratios,
        target_phase=0.0,
        target_amplitude=1.0
    )
    
    # Process
    start_time = time.time()
    network.compute_harmonic_state(time_steps)
    network.compute_dissonant_state(time_steps, foreign_signal)
    network.retune_nodes(foreign_signal, time_steps, mode_detector)
    
    # Generate output
    output = np.sum(network.retuned_outputs, axis=1)
    elapsed = time.time() - start_time
    
    print(f"Processing time: {elapsed:.3f} seconds")
    print(f"Detected mode: {network.mode}")
    print(f"Output shape: {output.shape}")
    
    return output, elapsed


def new_approach_cpu():
    """
    Example using the new MusicalResonantNetwork (CPU only).
    """
    print("\n=== New Approach (CPU) ===")
    
    # Setup - same parameters
    base_freq = 440.0
    harmonic_ratios = [1, 1.5, 1.33, 2, 2.5]
    foreign_signal = np.sin(2 * np.pi * 523.25 * np.linspace(0, 2, 44100 * 2))
    
    # Create network with mode detector
    mode_detector = ModeDetector()
    network = MusicalResonantNetwork(
        name="cpu_musical",
        base_freq=base_freq,
        mode="Ionian",
        mode_detector=mode_detector
    )
    
    # Create harmonic nodes
    network.create_harmonic_nodes(harmonic_ratios)
    
    # Create musical connections
    network.create_modal_connections("harmonic_series", weight_scale=0.5)
    
    # Process
    start_time = time.time()
    
    # Compute states
    harmonic_output = network.compute_harmonic_state(2.0, 44100)
    dissonant_output = network.compute_dissonant_state(2.0, foreign_signal, 44100)
    
    # Analyze and retune
    detected_mode = network.analyze_and_retune(foreign_signal, 44100)
    retuned_output = network.compute_harmonic_state(2.0, 44100)
    
    elapsed = time.time() - start_time
    
    print(f"Processing time: {elapsed:.3f} seconds")
    print(f"Detected mode: {detected_mode}")
    print(f"Output shape: {retuned_output.shape}")
    
    return retuned_output, elapsed


def new_approach_accelerated():
    """
    Example using the AcceleratedMusicalNetwork with GPU/Metal.
    """
    print("\n=== New Approach (Accelerated) ===")
    
    # Check available backends
    manager = BackendManager()
    available = manager.list_available_backends()
    print(f"Available backends: {[b.value for b in available]}")
    
    # Setup - same parameters
    base_freq = 440.0
    harmonic_ratios = [1, 1.5, 1.33, 2, 2.5]
    foreign_signal = np.sin(2 * np.pi * 523.25 * np.linspace(0, 2, 44100 * 2))
    
    # Create accelerated network
    mode_detector = ModeDetector()
    network = AcceleratedMusicalNetwork(
        name="gpu_musical",
        base_freq=base_freq,
        mode="Ionian",
        mode_detector=mode_detector,
        backend=None  # Auto-select best backend
    )
    
    # Create harmonic nodes
    network.create_harmonic_nodes(harmonic_ratios)
    network.create_modal_connections("harmonic_series", weight_scale=0.5)
    
    # Process with acceleration
    start_time = time.time()
    
    # Use accelerated methods
    harmonic_output = network.generate_audio_accelerated(2.0, 44100)
    
    # Analyze and retune
    detected_mode = network.analyze_and_retune(foreign_signal, 44100)
    retuned_output = network.generate_audio_accelerated(2.0, 44100)
    
    elapsed = time.time() - start_time
    
    print(f"Processing time: {elapsed:.3f} seconds")
    print(f"Detected mode: {detected_mode}")
    print(f"Output shape: {retuned_output.shape}")
    print(f"Performance stats: {network.get_performance_stats()['backend']}")
    
    return retuned_output, elapsed


def demonstrate_advanced_features():
    """
    Show new features not available in the original implementation.
    """
    print("\n=== Advanced Features Demo ===")
    
    # Create accelerated network
    mode_detector = ModeDetector()
    network = AcceleratedMusicalNetwork(
        base_freq=440.0,
        mode="Ionian",
        mode_detector=mode_detector
    )
    
    # 1. Chord Progression Generation
    print("\n1. Generating chord progression...")
    chords = [
        [1, 1.25, 1.5],      # Major triad (I)
        [1.125, 1.4, 1.68],  # Minor triad (ii)
        [1.25, 1.56, 1.875], # Major triad (IV)
        [1, 1.25, 1.5],      # Back to I
    ]
    
    progression = network.generate_chord_progression(chords, duration_per_chord=0.5)
    print(f"   Generated {len(progression)/44100:.1f} seconds of chord progression")
    
    # 2. Mode Morphing with Acceleration
    print("\n2. Morphing between modes...")
    network.create_harmonic_nodes([1, 1.122, 1.26, 1.5, 1.68, 1.89, 2])  # 7 notes
    morph_audio = network.morph_between_modes_accelerated("Dorian", morph_time=2.0)
    print(f"   Morphed from {network.mode} to Dorian over 2 seconds")
    
    # 3. Batch Signal Processing
    print("\n3. Batch processing multiple signals...")
    test_signals = [
        np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410)),   # A4
        np.sin(2 * np.pi * 523.25 * np.linspace(0, 0.1, 4410)), # C5
        np.sin(2 * np.pi * 659.25 * np.linspace(0, 0.1, 4410)), # E5
    ]
    
    results = network.batch_process_signals(test_signals, operation="analyze")
    for i, result in enumerate(results):
        print(f"   Signal {i+1}: Fundamental = {result['fundamental']:.1f} Hz")
    
    # 4. Real-time Parameter Control
    print("\n4. Real-time parameter modulation...")
    network.pitch_bend_range = 2.0  # ±2 semitones
    
    # Apply pitch bend to first node
    network.apply_pitch_bend("harmonic_0", 1.0)  # Bend up 1 semitone
    print("   Applied pitch bend to first harmonic")
    
    return progression, morph_audio


def compare_performance():
    """
    Compare performance between implementations.
    """
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Run all approaches
    original_out, original_time = original_approach()
    cpu_out, cpu_time = new_approach_cpu()
    accel_out, accel_time = new_approach_accelerated()
    
    # Calculate speedups
    cpu_speedup = original_time / cpu_time
    accel_speedup = original_time / accel_time
    
    print("\n=== Performance Summary ===")
    print(f"Original implementation: {original_time:.3f} seconds (baseline)")
    print(f"New CPU implementation: {cpu_time:.3f} seconds ({cpu_speedup:.1f}x speedup)")
    print(f"Accelerated implementation: {accel_time:.3f} seconds ({accel_speedup:.1f}x speedup)")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Waveform comparison (first 1000 samples)
    ax1.plot(original_out[:1000], label='Original', alpha=0.7)
    ax1.plot(cpu_out[:1000], label='New CPU', alpha=0.7)
    ax1.plot(accel_out[:1000], label='Accelerated', alpha=0.7)
    ax1.set_title('Output Waveform Comparison')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Performance bar chart
    methods = ['Original', 'New CPU', 'Accelerated']
    times = [original_time, cpu_time, accel_time]
    colors = ['gray', 'blue', 'green']
    
    bars = ax2.bar(methods, times, color=colors)
    ax2.set_title('Processing Time Comparison')
    ax2.set_ylabel('Time (seconds)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add speedup labels
    for i, (bar, speedup) in enumerate(zip(bars[1:], [cpu_speedup, accel_speedup])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('migration_comparison.png', dpi=150)
    plt.show()
    
    # Save audio samples
    wavfile.write('original_output.wav', 44100, (original_out * 32767).astype(np.int16))
    wavfile.write('accelerated_output.wav', 44100, (accel_out * 32767).astype(np.int16))
    print("\nSaved audio files: original_output.wav, accelerated_output.wav")


def show_migration_steps():
    """
    Print migration guide for users.
    """
    print("\n" + "="*60)
    print("MIGRATION GUIDE")
    print("="*60)
    
    migration_guide = """
    Step 1: Update Imports
    ----------------------
    OLD:
        from abstract import ResonantNetwork
        from detector import ModeDetector
    
    NEW:
        from synthnn.core import AcceleratedMusicalNetwork
        from detector import ModeDetector  # Keep using your existing detector
    
    Step 2: Create Network
    ----------------------
    OLD:
        network = ResonantNetwork(num_nodes, base_freq, harmonic_ratios, 
                                phase, amplitude)
    
    NEW:
        network = AcceleratedMusicalNetwork(base_freq=base_freq, 
                                          mode_detector=mode_detector)
        network.create_harmonic_nodes(harmonic_ratios)
    
    Step 3: Process Signals
    -----------------------
    OLD:
        network.compute_harmonic_state(time_steps)
        network.retune_nodes(foreign_signal, time_steps, mode_detector)
        output = np.sum(network.retuned_outputs, axis=1)
    
    NEW:
        network.analyze_and_retune(foreign_signal)
        output = network.generate_audio_accelerated(duration)
    
    Key Benefits:
    -------------
    1. Automatic GPU/Metal acceleration
    2. Cleaner API with more features
    3. Better performance and scalability
    4. New capabilities (chord progressions, mode morphing, batch processing)
    5. Maintains compatibility with existing ModeDetector
    
    Gradual Migration:
    ------------------
    You can run both systems side-by-side during migration:
    - Keep existing code running with abstract.py
    - Gradually port modules to use synthnn.core
    - Compare outputs to ensure consistency
    - Switch to accelerated version when ready
    """
    
    print(migration_guide)


def main():
    """Run all demonstrations."""
    # Show migration steps first
    show_migration_steps()
    
    # Compare implementations
    compare_performance()
    
    # Demonstrate new features
    print("\n" + "="*60)
    print("NEW FEATURES DEMONSTRATION")
    print("="*60)
    
    progression, morph = demonstrate_advanced_features()
    
    # Save advanced outputs
    wavfile.write('chord_progression.wav', 44100, (progression * 32767).astype(np.int16))
    wavfile.write('mode_morph.wav', 44100, (morph * 32767).astype(np.int16))
    print("\nSaved additional files: chord_progression.wav, mode_morph.wav")
    
    print("\n✅ Migration demonstration complete!")
    print("Check the generated audio files and migration_comparison.png")


if __name__ == "__main__":
    main() 