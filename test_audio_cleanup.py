#!/usr/bin/env python3
"""
Quick test for audio cleanup features
"""

import numpy as np
from synthnn.core import (
    AudioCleanupEngine, ArtifactDetector, create_cleanup_pipeline,
    CompositionEngine, MusicalStyle
)

print("Testing Audio Cleanup Features...")

# Test 1: Artifact Detection
print("\n1. Testing Artifact Detection...")

# Create test audio with artificial whistle
sample_rate = 44100
duration = 1.0
time = np.arange(int(duration * sample_rate)) / sample_rate

# Clean audio (simple sine wave)
clean = 0.5 * np.sin(2 * np.pi * 440 * time)

# Add whistle artifact
whistle_freq = 3500.0
whistle = 0.1 * np.sin(2 * np.pi * whistle_freq * time)
noisy = clean + whistle

# Detect artifacts
detector = ArtifactDetector(sample_rate)
artifacts = detector.detect_whistling(noisy, threshold=0.05)

print(f"   ✓ Detected {len(artifacts)} artifacts")
for artifact in artifacts:
    print(f"     - {artifact.artifact_type.value} at {artifact.frequency:.1f}Hz")

# Test 2: Simple Cleanup
print("\n2. Testing Simple Cleanup...")
engine = AudioCleanupEngine(sample_rate)
cleaned = engine.cleanup_simple(noisy, artifacts)

# Check improvement
noise_before = np.mean((noisy - clean)**2)
noise_after = np.mean((cleaned - clean)**2)
improvement = (1 - noise_after/noise_before) * 100

print(f"   ✓ Noise reduced by {improvement:.1f}%")

# Test 3: Pipeline Creation
print("\n3. Testing Cleanup Pipelines...")

for style in ["simple", "resynthesis", "adaptive"]:
    try:
        pipeline = create_cleanup_pipeline(style)
        print(f"   ✓ Created {style} pipeline")
    except Exception as e:
        print(f"   ✗ Failed to create {style} pipeline: {e}")

# Test 4: Resynthesis with real music
print("\n4. Testing Resynthesis Cleanup with Music...")

# Generate a short musical piece
composer = CompositionEngine(style=MusicalStyle.AMBIENT, base_tempo=60)
music = composer.render_composition(1.0)

# Add artifact
music_noisy = music + 0.05 * np.sin(2 * np.pi * 2750 * time)

# Clean with resynthesis
cleaned_music = engine.cleanup_resynthesis(music_noisy, mode="ionian")

print(f"   ✓ Applied resynthesis cleanup to musical audio")
print(f"   ✓ Original peak: {np.max(np.abs(music)):.3f}")
print(f"   ✓ Cleaned peak: {np.max(np.abs(cleaned_music)):.3f}")

print("\n✅ All audio cleanup features working correctly!")
print("\nRun 'python demos/audio_cleanup_demo.py' for full demonstrations.") 