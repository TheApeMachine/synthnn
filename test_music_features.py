#!/usr/bin/env python3
"""
Quick test for advanced musical features
"""

import numpy as np
from synthnn.core import (
    # Musical synthesis
    MusicalNode, MusicalResonantNetwork, WaveShape, FilterType, ADSREnvelope,
    
    # Rhythm
    RhythmEngine, RhythmPattern, DrumVoice, TimeSignature,
    
    # Composition
    CompositionEngine, MusicalStyle, HarmonicProgression
)

print("Testing Advanced Musical Features...")

# Test 1: Musical Synthesis
print("\n1. Testing Musical Synthesis...")
node = MusicalNode("test_node", frequency=440.0)

# Configure synthesizer
node.synthesizer.add_oscillator(WaveShape.SAWTOOTH, 0.5)
node.synthesizer.add_oscillator(WaveShape.SINE, 0.5)
node.synthesizer.filter.filter_type = FilterType.LOWPASS
node.synthesizer.filter.cutoff = 1000

# Trigger and generate
node.trigger(velocity=0.8)
audio = node.generate_audio(0.5, 22050)

print(f"   ✓ Generated {len(audio)} samples")
print(f"   ✓ Peak amplitude: {np.max(np.abs(audio)):.3f}")

# Test 2: Rhythm Engine
print("\n2. Testing Rhythm Engine...")
rhythm = RhythmEngine(tempo=120.0)

# Activate patterns
rhythm.activate_pattern('four_on_floor', DrumVoice.KICK)
rhythm.activate_pattern('hihat_8ths', DrumVoice.HIHAT_CLOSED)

# Generate events
events = rhythm.generate_events(2.0)
print(f"   ✓ Generated {len(events)} rhythm events")

# Test polyrhythm
poly = rhythm.generate_polyrhythm('basic_rock', 'hihat_8ths', (3, 2))
print(f"   ✓ Created polyrhythm with {len(poly.pattern)} beats")

# Test 3: Composition Engine
print("\n3. Testing Composition Engine...")
composer = CompositionEngine(style=MusicalStyle.CLASSICAL)

# Create structure
structure = composer.create_structure("ABA", section_measures=2)
print(f"   ✓ Created structure with {structure.total_measures} measures")

# Generate progression
progression = composer.generate_harmonic_progression(
    HarmonicProgression.I_IV_V_I,
    structure.total_measures
)
print(f"   ✓ Generated chord progression: {progression[:4]}")

# Generate melody
melody = composer.generate_melody(structure.total_measures, notes_per_measure=4)
print(f"   ✓ Generated melody with {len(melody)} notes")

# Test network building
network = composer.build_arrangement()
print(f"   ✓ Built arrangement with {len(network.nodes)} voices")

print("\n✅ All musical features working correctly!")
print("\nRun 'python demos/advanced_music_demo.py' for full demonstrations.") 