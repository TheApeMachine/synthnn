#!/usr/bin/env python3
"""
Audio Cleanup Demo for SynthNN

Demonstrates how to use SynthNN's resonance-based audio cleanup
to remove artifacts like persistent whistling tones from AI-generated audio.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import os

from synthnn.core.audio_cleanup import (
    AudioCleanupEngine, ArtifactDetector, ResonanceFilter,
    ArtifactType, create_cleanup_pipeline
)
from synthnn.core import MusicalResonantNetwork, CompositionEngine, MusicalStyle


def generate_artificial_artifacts(duration: float = 5.0, sample_rate: int = 44100):
    """Generate audio with artificial artifacts for testing."""
    print("\n1. Generating test audio with artifacts...")
    
    # Create a nice musical piece
    composer = CompositionEngine(style=MusicalStyle.AMBIENT, base_tempo=80)
    composer.create_structure("AB", section_measures=4)
    composer.generate_harmonic_progression(composer.structure.total_measures)
    composer.generate_melody(composer.structure.total_measures)
    
    # Generate clean audio
    clean_audio = composer.render_composition(duration)
    
    # Add artificial artifacts
    time = np.arange(len(clean_audio)) / sample_rate
    
    # Add persistent whistling tone (like what you hear from Suno)
    whistle_freq = 3750.0  # Hz - a common artifact frequency
    whistle = 0.05 * np.sin(2 * np.pi * whistle_freq * time)
    
    # Add another whistle at a harmonic
    whistle2_freq = 7500.0  # Hz - harmonic of first
    whistle2 = 0.03 * np.sin(2 * np.pi * whistle2_freq * time)
    
    # Add some 60Hz hum
    hum = 0.02 * np.sin(2 * np.pi * 60 * time)
    
    # Add random resonance peak
    resonance_freq = 1234.5  # Hz - non-harmonic frequency
    resonance = 0.04 * np.sin(2 * np.pi * resonance_freq * time)
    
    # Combine
    noisy_audio = clean_audio + whistle + whistle2 + hum + resonance
    
    # Add some distortion
    noisy_audio = np.clip(noisy_audio * 1.2, -1, 1)
    
    print(f"   ✓ Generated {duration}s of audio with artifacts")
    print(f"   ✓ Added whistles at {whistle_freq}Hz and {whistle2_freq}Hz")
    print(f"   ✓ Added 60Hz hum and resonance at {resonance_freq}Hz")
    
    return clean_audio, noisy_audio


def visualize_spectrum_comparison(clean: np.ndarray, noisy: np.ndarray, 
                                 cleaned: np.ndarray, sample_rate: int,
                                 title: str = "Spectrum Comparison"):
    """Visualize frequency spectra before and after cleanup."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)
    
    # Compute spectra
    for ax, audio, label in zip(axes.flat, 
                               [clean, noisy, cleaned, noisy - cleaned],
                               ['Original Clean', 'With Artifacts', 'After Cleanup', 'Removed Artifacts']):
        
        spectrum = np.abs(fft(audio))
        frequencies = fftfreq(len(audio), 1/sample_rate)
        
        # Plot positive frequencies only
        pos_mask = frequencies > 0
        ax.semilogy(frequencies[pos_mask][:len(frequencies)//4], 
                   spectrum[pos_mask][:len(frequencies)//4],
                   linewidth=0.5)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10000)
        
    plt.tight_layout()
    return fig


def demo_artifact_detection():
    """Demonstrate artifact detection capabilities."""
    print("\n" + "="*60)
    print("🔍 ARTIFACT DETECTION DEMO")
    print("="*60)
    
    # Generate test audio
    clean_audio, noisy_audio = generate_artificial_artifacts(duration=3.0)
    
    # Create detector
    detector = ArtifactDetector()
    
    # Detect artifacts
    print("\n2. Detecting artifacts...")
    artifacts = detector.detect_whistling(noisy_audio)
    
    print(f"\n   ✓ Found {len(artifacts)} whistling artifacts:")
    for artifact in artifacts:
        print(f"     - {artifact.artifact_type.value} at {artifact.frequency:.1f}Hz "
              f"(strength: {artifact.strength:.3f}, persistence: {artifact.phase_coherence:.2f})")
    
    # Visualize detection
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle("Artifact Detection Results")
    
    # Time domain
    time = np.arange(len(noisy_audio)) / 44100
    ax1.plot(time, noisy_audio, linewidth=0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Audio with Artifacts")
    ax1.grid(True, alpha=0.3)
    
    # Frequency domain with detected artifacts marked
    spectrum = np.abs(fft(noisy_audio))
    frequencies = fftfreq(len(noisy_audio), 1/44100)
    
    pos_mask = frequencies > 0
    ax2.semilogy(frequencies[pos_mask][:len(frequencies)//4], 
                spectrum[pos_mask][:len(frequencies)//4],
                linewidth=0.5, label='Spectrum')
    
    # Mark detected artifacts
    for artifact in artifacts:
        if artifact.frequency:
            ax2.axvline(artifact.frequency, color='red', linestyle='--', 
                       alpha=0.7, label=f'Artifact at {artifact.frequency:.0f}Hz')
    
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Detected Artifacts in Spectrum")
    ax2.set_xlim(0, 10000)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("artifact_detection_demo.png", dpi=150)
    plt.close()
    
    print("\n   ✓ Saved visualization to artifact_detection_demo.png")
    
    return clean_audio, noisy_audio, artifacts


def demo_simple_cleanup():
    """Demonstrate simple notch filtering cleanup."""
    print("\n" + "="*60)
    print("🧹 SIMPLE CLEANUP DEMO")
    print("="*60)
    
    # Get test audio and artifacts
    clean_audio, noisy_audio, artifacts = demo_artifact_detection()
    
    # Create cleanup engine
    engine = AudioCleanupEngine()
    
    # Simple cleanup
    print("\n3. Applying simple notch filter cleanup...")
    cleaned_audio = engine.cleanup_simple(noisy_audio, artifacts)
    
    # Calculate improvement
    noise_before = np.mean((noisy_audio - clean_audio)**2)
    noise_after = np.mean((cleaned_audio - clean_audio)**2)
    improvement = (1 - noise_after/noise_before) * 100
    
    print(f"\n   ✓ Noise reduced by {improvement:.1f}%")
    
    # Save audio files
    wavfile.write("audio_with_artifacts.wav", 44100, 
                 (noisy_audio * 32767).astype(np.int16))
    wavfile.write("audio_cleaned_simple.wav", 44100, 
                 (cleaned_audio * 32767).astype(np.int16))
    
    print("   ✓ Saved audio files:")
    print("     - audio_with_artifacts.wav")
    print("     - audio_cleaned_simple.wav")
    
    # Visualize
    fig = visualize_spectrum_comparison(
        clean_audio, noisy_audio, cleaned_audio, 44100,
        "Simple Notch Filter Cleanup"
    )
    plt.savefig("simple_cleanup_comparison.png", dpi=150)
    plt.close()
    
    print("   ✓ Saved visualization to simple_cleanup_comparison.png")
    
    return clean_audio, noisy_audio, cleaned_audio


def demo_resynthesis_cleanup():
    """Demonstrate advanced resynthesis cleanup."""
    print("\n" + "="*60)
    print("🎼 RESYNTHESIS CLEANUP DEMO")
    print("="*60)
    
    # Get test audio
    clean_audio, noisy_audio = generate_artificial_artifacts(duration=3.0)
    
    # Create cleanup engine
    engine = AudioCleanupEngine()
    
    # Resynthesis cleanup with mode enforcement
    print("\n4. Applying resonant resynthesis cleanup...")
    print("   - Encoding audio to resonant patterns")
    print("   - Filtering in resonant domain")
    print("   - Resynthesizing with mode enforcement")
    
    cleaned_audio = engine.cleanup_resynthesis(
        noisy_audio, 
        mode="ionian",  # Enforce major scale
        preserve_transients=True
    )
    
    # Calculate improvement
    noise_before = np.mean((noisy_audio - clean_audio)**2)
    noise_after = np.mean((cleaned_audio - clean_audio)**2)
    improvement = (1 - noise_after/noise_before) * 100
    
    print(f"\n   ✓ Noise reduced by {improvement:.1f}%")
    print("   ✓ Musical structure preserved through mode enforcement")
    
    # Save audio
    wavfile.write("audio_cleaned_resynthesis.wav", 44100, 
                 (cleaned_audio * 32767).astype(np.int16))
    
    print("   ✓ Saved to audio_cleaned_resynthesis.wav")
    
    # Visualize
    fig = visualize_spectrum_comparison(
        clean_audio, noisy_audio, cleaned_audio, 44100,
        "Resonant Resynthesis Cleanup"
    )
    plt.savefig("resynthesis_cleanup_comparison.png", dpi=150)
    plt.close()
    
    print("   ✓ Saved visualization to resynthesis_cleanup_comparison.png")
    
    return cleaned_audio


def demo_adaptive_cleanup():
    """Demonstrate adaptive cleanup with learning."""
    print("\n" + "="*60)
    print("🤖 ADAPTIVE CLEANUP DEMO")
    print("="*60)
    
    # Generate two similar pieces - one clean, one noisy
    print("\n5. Generating reference and target audio...")
    
    # Clean reference
    composer1 = CompositionEngine(style=MusicalStyle.AMBIENT)
    reference_audio = composer1.render_composition(2.0)
    
    # Noisy target (different but similar style)
    composer2 = CompositionEngine(style=MusicalStyle.AMBIENT)
    target_clean = composer2.render_composition(2.0)
    
    # Add artifacts to target
    time = np.arange(len(target_clean)) / 44100
    whistle = 0.06 * np.sin(2 * np.pi * 4567 * time)
    target_noisy = target_clean + whistle
    
    # Create cleanup engine
    engine = AudioCleanupEngine()
    
    # Adaptive cleanup
    print("\n   Applying adaptive cleanup...")
    print("   - Learning from clean reference")
    print("   - Adapting network to match spectral characteristics")
    
    cleaned_audio = engine.cleanup_adaptive(
        target_noisy,
        reference_audio=reference_audio,
        learning_rate=0.1
    )
    
    # Save
    wavfile.write("audio_cleaned_adaptive.wav", 44100,
                 (cleaned_audio * 32767).astype(np.int16))
    
    print("\n   ✓ Saved to audio_cleaned_adaptive.wav")
    
    return cleaned_audio


def demo_pipeline_usage():
    """Demonstrate using cleanup as a pipeline."""
    print("\n" + "="*60)
    print("🔧 CLEANUP PIPELINE DEMO")
    print("="*60)
    
    # Create different pipeline styles
    print("\n6. Creating cleanup pipelines...")
    
    simple_cleanup = create_cleanup_pipeline("simple")
    resynth_cleanup = create_cleanup_pipeline("resynthesis")
    adaptive_cleanup = create_cleanup_pipeline("adaptive")
    
    # Generate test audio
    _, noisy_audio = generate_artificial_artifacts(duration=2.0)
    
    # Apply different pipelines
    print("\n   Applying different cleanup styles:")
    
    results = {}
    for name, pipeline in [("simple", simple_cleanup), 
                          ("resynthesis", resynth_cleanup),
                          ("adaptive", adaptive_cleanup)]:
        print(f"   - {name} cleanup...")
        if name == "adaptive":
            # Adaptive needs special handling
            cleaned = adaptive_cleanup(noisy_audio, reference_audio=None)
        else:
            cleaned = pipeline(noisy_audio)
        results[name] = cleaned
        
    print("\n   ✓ All pipelines applied successfully")
    
    # Compare results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Cleanup Pipeline Comparison")
    
    time = np.arange(len(noisy_audio)) / 44100
    
    for ax, (name, audio) in zip(axes.flat, 
                                [("Original", noisy_audio)] + list(results.items())):
        ax.plot(time[:4410], audio[:4410], linewidth=0.5)  # First 0.1s
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{name.title()} Cleanup")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("pipeline_comparison.png", dpi=150)
    plt.close()
    
    print("   ✓ Saved comparison to pipeline_comparison.png")


def main():
    """Run all audio cleanup demos."""
    print("\n" + "🧹"*30)
    print("SYNTHNN AUDIO CLEANUP DEMO")
    print("🧹"*30)
    print("\nThis demo shows how to use SynthNN's resonance-based")
    print("audio cleanup to remove artifacts from AI-generated audio")
    print("(like persistent whistling tones from Suno).")
    
    # Create output directory
    os.makedirs("audio_cleanup_output", exist_ok=True)
    os.chdir("audio_cleanup_output")
    
    # Run demos
    demo_simple_cleanup()
    demo_resynthesis_cleanup()
    demo_adaptive_cleanup()
    demo_pipeline_usage()
    
    print("\n" + "✨"*30)
    print("AUDIO CLEANUP DEMO COMPLETE!")
    print("✨"*30)
    print("\nGenerated files in 'audio_cleanup_output' directory:")
    print("- Audio files (.wav) - Compare before/after cleanup")
    print("- Visualizations (.png) - See artifact detection and removal")
    print("\nThe cleanup engine successfully:")
    print("✓ Detected persistent whistling tones")
    print("✓ Removed artifacts while preserving musical content")
    print("✓ Used resonance-based filtering and resynthesis")
    print("✓ Demonstrated adaptive learning from clean references")


if __name__ == "__main__":
    main() 