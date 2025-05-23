"""
Demonstration of microtonal capabilities in SynthNN.

This script showcases how the resonance-based architecture naturally handles
microtonal scales, continuous pitch spaces, and non-Western tuning systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthnn.core.microtonal_extensions import (
    MicrotonalResonantNetwork, 
    MicrotonalScaleLibrary,
    AdaptiveMicrotonalSystem
)


def demonstrate_microtonal_scales():
    """Demonstrate various microtonal scales from different cultures."""
    
    print("=== Microtonal Scale Demonstration ===\n")
    
    # Get scale library
    scales = MicrotonalScaleLibrary.get_scales()
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Demonstrate a selection of scales
    demo_scales = ['just_major', 'maqam_rast', 'shruti_22', 
                   'slendro', 'bohlen_pierce', 'golden_ratio']
    
    for idx, scale_name in enumerate(demo_scales):
        if scale_name in scales:
            scale = scales[scale_name]
            print(f"{scale.name}: {scale.description}")
            print(f"  Intervals: {[f'{r:.3f}' for r in scale.intervals[:8]]}")
            
            # Convert to cents
            cents = scale.to_cents()
            print(f"  Cents: {[f'{c:.1f}' for c in cents[:8]]}\n")
            
            # Plot on axes
            ax = axes[idx]
            ax.scatter(range(len(cents[:12])), cents[:12], s=50)
            ax.plot(range(len(cents[:12])), cents[:12], alpha=0.5)
            ax.set_title(scale.name)
            ax.set_xlabel('Scale Degree')
            ax.set_ylabel('Cents')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('microtonal_scales_comparison.png', dpi=150)
    plt.show()


def demonstrate_continuous_pitch_field():
    """Demonstrate continuous pitch manipulation with glissandi."""
    
    print("\n=== Continuous Pitch Field Demonstration ===\n")
    
    # Create network with continuous pitch field
    network = MicrotonalResonantNetwork(name="continuous_demo")
    
    # Create nodes distributed by golden ratio
    network.create_continuous_pitch_field(
        freq_range=(220, 880),  # A3 to A5
        num_nodes=13,
        distribution='golden'
    )
    
    # Create spectral connections
    network.create_spectral_connections(harmonicity_threshold=0.15)
    
    print(f"Created {len(network.nodes)} nodes with golden ratio distribution")
    
    # Generate evolving texture with glissandi
    duration = 5.0
    sample_rate = 44100
    
    # Set some glissando targets
    node_ids = list(network.nodes.keys())
    for i in range(0, len(node_ids), 3):
        # Create upward glissandi
        current_freq = network.nodes[node_ids[i]].frequency
        network.glissando_to_pitch(node_ids[i], current_freq * 1.5)
    
    # Generate audio
    print("Generating continuous pitch field audio...")
    audio = network.generate_microtonal_texture(
        duration=duration,
        density=0.6,
        evolution_rate=0.2,
        sample_rate=sample_rate
    )
    
    # Save audio
    wavfile.write('continuous_pitch_field.wav', sample_rate, 
                  (audio * 32767).astype(np.int16))
    print("Saved continuous_pitch_field.wav")
    
    # Plot pitch trajectories
    plt.figure(figsize=(10, 6))
    for node_id, trajectory in network.pitch_trajectories.items():
        if len(trajectory) > 1:
            plt.plot(trajectory, alpha=0.7, label=node_id)
    
    plt.xlabel('Time Step')
    plt.ylabel('Frequency (Hz)')
    plt.title('Pitch Trajectories in Continuous Field')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('pitch_trajectories.png', dpi=150)
    plt.show()


def demonstrate_maqam_modulation():
    """Demonstrate smooth modulation between Arabic maqamat."""
    
    print("\n=== Maqam Modulation Demonstration ===\n")
    
    # Get maqam scales
    scales = MicrotonalScaleLibrary.get_scales()
    
    # Create network with Maqam Rast
    network = MicrotonalResonantNetwork(
        name="maqam_demo",
        base_freq=293.66,  # D4
        scale=scales['maqam_rast']
    )
    
    network.create_scale_nodes(num_octaves=2)
    network.create_modal_connections('harmonic_series')
    
    sample_rate = 44100
    
    # Generate phrase in Rast
    print("Generating Maqam Rast phrase...")
    rast_audio = network.compute_harmonic_state(2.0, sample_rate)
    
    # Modulate to Bayati
    print("Modulating to Maqam Bayati...")
    network.scale = scales['maqam_bayati']
    
    # Retune nodes to new maqam
    for i, node_id in enumerate(network.nodes.keys()):
        if i < len(network.scale.intervals):
            new_freq = network.scale.get_frequency(network.base_freq, i)
            network.glissando_to_pitch(node_id, new_freq)
    
    # Generate modulation
    modulation_audio = []
    for _ in range(int(2.0 * sample_rate)):
        network.step_with_glissando(1.0 / sample_rate)
        signals = network.get_signals()
        modulation_audio.append(sum(signals.values()) * 0.5)
    
    # Generate phrase in Bayati
    bayati_audio = network.compute_harmonic_state(2.0, sample_rate)
    
    # Concatenate
    full_audio = np.concatenate([rast_audio, modulation_audio, bayati_audio])
    
    # Normalize and save
    full_audio = full_audio / np.max(np.abs(full_audio)) * 0.8
    wavfile.write('maqam_modulation.wav', sample_rate,
                  (full_audio * 32767).astype(np.int16))
    print("Saved maqam_modulation.wav")


def demonstrate_adaptive_learning():
    """Demonstrate learning scales from performance."""
    
    print("\n=== Adaptive Scale Learning Demonstration ===\n")
    
    # Create a simple melodic pattern
    sample_rate = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Create a melody with specific intervals (simulating a performance)
    # Using ratios that approximate a Javanese scale
    frequencies = [293.66, 333.0, 377.0, 448.0, 509.0]  # D-based pentatonic
    melody = np.zeros_like(t)
    
    segment_length = len(t) // len(frequencies)
    for i, freq in enumerate(frequencies):
        start = i * segment_length
        end = (i + 1) * segment_length if i < len(frequencies) - 1 else len(t)
        melody[start:end] = np.sin(2 * np.pi * freq * t[start:end])
    
    # Add some envelope
    envelope = np.exp(-t * 0.5)
    melody *= envelope
    
    # Create adaptive system
    base_network = MicrotonalResonantNetwork("adaptive_demo")
    adaptive_system = AdaptiveMicrotonalSystem(base_network)
    
    # Learn scale from the melody
    print("Learning scale from melodic pattern...")
    learned_scale = adaptive_system.learn_scale_from_performance(
        melody, sample_rate, "learned_javanese"
    )
    
    print(f"Learned scale: {learned_scale.name}")
    print(f"Intervals: {[f'{r:.3f}' for r in learned_scale.intervals]}")
    
    # Find consonant intervals
    print("\nFinding consonant intervals in the 3:2 to 2:1 range...")
    consonant_intervals = adaptive_system.find_consonant_intervals(
        freq_range=(1.5, 2.0),
        resolution_cents=10.0
    )
    
    print(f"Found {len(consonant_intervals)} consonant intervals:")
    for ratio in consonant_intervals[:5]:
        cents = 1200 * np.log2(ratio)
        print(f"  Ratio: {ratio:.3f}, Cents: {cents:.1f}")


def demonstrate_comma_pump():
    """Demonstrate comma pump for exploring microtonal spaces."""
    
    print("\n=== Comma Pump Demonstration ===\n")
    
    # Create network with just intonation
    scales = MicrotonalScaleLibrary.get_scales()
    network = MicrotonalResonantNetwork(
        name="comma_pump_demo",
        scale=scales['just_major']
    )
    
    network.create_scale_nodes()
    network.create_modal_connections('full', weight_scale=0.3)
    
    sample_rate = 44100
    duration = 1.0
    
    # Generate sequence with comma pumps
    audio_segments = []
    
    # Original tuning
    segment = network.compute_harmonic_state(duration, sample_rate)
    audio_segments.append(segment)
    
    # Apply syntonic comma pump
    print("Applying syntonic comma pump...")
    network.apply_comma_pump('syntonic')
    
    # Generate with comma shift
    pump_audio = []
    for _ in range(int(duration * sample_rate)):
        network.step_with_glissando(1.0 / sample_rate)
        signals = network.get_signals()
        pump_audio.append(sum(signals.values()) * 0.5)
    
    audio_segments.append(np.array(pump_audio))
    
    # Return to original (approximately)
    for _ in range(4):
        network.apply_comma_pump('syntonic')
    
    return_audio = []
    for _ in range(int(duration * sample_rate)):
        network.step_with_glissando(1.0 / sample_rate)
        signals = network.get_signals()
        return_audio.append(sum(signals.values()) * 0.5)
    
    audio_segments.append(np.array(return_audio))
    
    # Concatenate and save
    full_audio = np.concatenate(audio_segments)
    full_audio = full_audio / np.max(np.abs(full_audio)) * 0.8
    
    wavfile.write('comma_pump_demo.wav', sample_rate,
                  (full_audio * 32767).astype(np.int16))
    print("Saved comma_pump_demo.wav")
    
    # Plot frequency drift
    plt.figure(figsize=(10, 6))
    freqs = [network.base_freq, network.base_freq * (81/80), 
             network.base_freq * (81/80)**5]
    cents_drift = [0, 1200 * np.log2(81/80), 1200 * np.log2((81/80)**5)]
    
    plt.plot([0, 1, 2], cents_drift, 'o-', markersize=10)
    plt.xlabel('Comma Pump Iterations')
    plt.ylabel('Cents Drift from Original')
    plt.title('Syntonic Comma Pump Effect')
    plt.grid(True, alpha=0.3)
    plt.savefig('comma_pump_drift.png', dpi=150)
    plt.show()


def main():
    """Run all demonstrations."""
    
    print("SynthNN Microtonal Extensions Demo")
    print("==================================\n")
    
    # Create output directory
    os.makedirs('microtonal_outputs', exist_ok=True)
    os.chdir('microtonal_outputs')
    
    # Run demonstrations
    demonstrate_microtonal_scales()
    demonstrate_continuous_pitch_field()
    demonstrate_maqam_modulation()
    demonstrate_adaptive_learning()
    demonstrate_comma_pump()
    
    print("\n=== Demo Complete ===")
    print("Check the 'microtonal_outputs' directory for generated files:")
    print("  - Audio files (.wav)")
    print("  - Visualization plots (.png)")


if __name__ == "__main__":
    main() 