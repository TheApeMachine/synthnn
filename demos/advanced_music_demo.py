#!/usr/bin/env python3
"""
Advanced Music Demo for SynthNN

Demonstrates the enhanced musical capabilities including:
- Advanced synthesis with multiple waveforms, filters, and effects
- Rhythm engine with grooves and humanization
- Complete composition system with structure and emotion
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

from synthnn.core.musical_synthesis import (
    MusicalNode, MusicalResonantNetwork, MusicalSynthesizer,
    WaveShape, FilterType, ADSREnvelope,
)

from synthnn.core.rhythm_engine import (
    RhythmEngine, RhythmPattern, GrooveTemplate, DrumVoice,
    TimeSignature,
)

from synthnn.core.composition_engine import (
    CompositionEngine, MusicalStyle, HarmonicProgression,
)

from synthnn.core.emotional_resonance import (
    EmotionalResonanceEngine, EmotionCategory
)


def demo_synthesis_capabilities():
    """Demonstrate advanced synthesis features."""
    print("\n" + "="*60)
    print("🎹 ADVANCED SYNTHESIS DEMO")
    print("="*60)
    
    # Create a musical node
    print("\n1. Creating musical node with advanced synthesis...")
    node = MusicalNode("synth_demo", frequency=440.0)
    
    # Configure synthesizer
    synth = node.synthesizer
    
    # Add multiple oscillators
    synth.add_oscillator(WaveShape.SAWTOOTH, mix=0.5)
    synth.add_oscillator(WaveShape.SQUARE, mix=0.3)
    synth.add_oscillator(WaveShape.SINE, mix=0.2)
    
    # Configure filter
    synth.filter.filter_type = FilterType.LOWPASS
    synth.filter.cutoff = 1000.0
    synth.filter.resonance = 2.0
    
    # Configure envelopes
    synth.amplitude_envelope = ADSREnvelope(
        attack=0.01,
        decay=0.1,
        sustain=0.7,
        release=0.3
    )
    
    synth.filter_envelope = ADSREnvelope(
        attack=0.05,
        decay=0.2,
        sustain=0.3,
        release=0.5
    )
    synth.filter_envelope_amount = 0.8
    
    # Add effects
    synth.reverb_mix = 0.3
    synth.delay_mix = 0.2
    synth.delay_time = 0.375  # Dotted eighth
    
    # Add modulation
    synth.vibrato_rate = 5.0
    synth.vibrato_depth = 0.02
    synth.tremolo_rate = 3.0
    synth.tremolo_depth = 0.1
    
    # Generate different notes
    print("\n2. Generating synthesized notes...")
    
    notes = {
        'C4': 261.63,
        'E4': 329.63,
        'G4': 392.00,
        'C5': 523.25
    }
    
    duration = 1.0
    sample_rate = 44100
    
    # Generate audio for each note
    audio_samples = []
    
    for note_name, freq in notes.items():
        print(f"   Synthesizing {note_name} ({freq:.2f} Hz)...")
        
        # Update node frequency
        node.frequency = freq
        
        # Trigger note
        node.trigger(velocity=0.8)
        
        # Generate audio
        audio = node.generate_audio(duration, sample_rate)
        audio_samples.append(audio)
        
        # Release note for next one
        node.release()
    
    # Concatenate all notes
    full_audio = np.concatenate(audio_samples)
    
    # Save audio
    output_file = "advanced_synthesis_demo.wav"
    wavfile.write(output_file, sample_rate, (full_audio * 32767).astype(np.int16))
    print(f"\n✓ Saved synthesis demo to {output_file}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Advanced Synthesis Visualization")
    
    # Waveform
    time = np.arange(len(audio_samples[0])) / sample_rate
    axes[0, 0].plot(time[:2000], audio_samples[0][:2000], linewidth=0.5)
    axes[0, 0].set_title("Waveform (C4)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Spectrum
    from scipy.fft import fft, fftfreq
    
    spectrum = np.abs(fft(audio_samples[0]))
    frequencies = fftfreq(len(audio_samples[0]), 1/sample_rate)
    
    axes[0, 1].plot(frequencies[:len(frequencies)//2], 
                    spectrum[:len(spectrum)//2], linewidth=0.5)
    axes[0, 1].set_title("Frequency Spectrum")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Magnitude")
    axes[0, 1].set_xlim(0, 5000)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Envelope visualization
    env_time = np.linspace(0, duration, 1000)
    amp_env = synth.amplitude_envelope.generate(duration, sample_rate)
    filt_env = synth.filter_envelope.generate(duration, sample_rate)
    
    axes[1, 0].plot(np.linspace(0, duration, len(amp_env)), amp_env, 
                    label='Amplitude', linewidth=2)
    axes[1, 0].plot(np.linspace(0, duration, len(filt_env)), filt_env, 
                    label='Filter', linewidth=2, alpha=0.7)
    axes[1, 0].set_title("ADSR Envelopes")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Level")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # All notes comparison
    for i, (note_name, audio) in enumerate(zip(notes.keys(), audio_samples)):
        axes[1, 1].plot(time[:1000], audio[:1000] + i*2, 
                       label=note_name, linewidth=0.8, alpha=0.7)
    
    axes[1, 1].set_title("All Notes Comparison")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Amplitude (offset)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("advanced_synthesis_visualization.png", dpi=150)
    plt.close()
    
    print("✓ Saved visualization to advanced_synthesis_visualization.png")


def demo_rhythm_engine():
    """Demonstrate rhythm engine capabilities."""
    print("\n" + "="*60)
    print("🥁 RHYTHM ENGINE DEMO")
    print("="*60)
    
    # Create rhythm engine
    print("\n1. Creating rhythm engine...")
    rhythm = RhythmEngine(tempo=120.0, time_signature=TimeSignature(4, 4))
    
    # Create custom pattern
    print("\n2. Creating custom rhythm patterns...")
    
    # Afro-Cuban clave pattern
    clave_pattern = RhythmPattern(
        'clave_3_2',
        pattern=[0.0, 0.375, 0.5, 0.75, 0.875],
        velocities=[1.0, 0.8, 0.9, 0.8, 0.7]
    )
    rhythm.add_pattern('clave', clave_pattern)
    
    # Bossa nova pattern
    bossa_kick = RhythmPattern(
        'bossa_kick',
        pattern=[0.0, 0.375, 0.5, 0.875],
        velocities=[1.0, 0.7, 0.8, 0.6]
    )
    rhythm.add_pattern('bossa_kick', bossa_kick)
    
    # Create grooves
    print("\n3. Creating groove templates...")
    
    # Latin groove with swing
    latin_groove = GrooveTemplate(
        'latin',
        base_pattern=clave_pattern,
        swing=0.15,
        humanize=0.2
    )
    rhythm.add_groove('latin', latin_groove)
    
    # Activate patterns
    print("\n4. Building drum arrangement...")
    
    # Basic rock beat
    rhythm.activate_pattern('basic_rock', DrumVoice.KICK)
    rhythm.activate_pattern('snare_backbeat', DrumVoice.SNARE)
    rhythm.activate_pattern('hihat_8ths', DrumVoice.HIHAT_CLOSED)
    
    # Generate events
    duration = 8.0  # 2 measures at 120 BPM
    events = rhythm.generate_events(duration)
    
    print(f"\n   Generated {len(events)} rhythm events")
    
    # Create rhythm network
    print("\n5. Creating rhythm network and generating audio...")
    rhythm_network = rhythm.create_rhythm_network()
    
    # Generate audio
    sample_rate = 44100
    audio = np.zeros(int(duration * sample_rate))
    
    # Process in chunks
    chunk_duration = 0.1  # 100ms chunks
    num_chunks = int(duration / chunk_duration)
    
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_duration
        chunk_end = (chunk_idx + 1) * chunk_duration
        
        # Get events for this chunk
        chunk_events = [
            (t - chunk_start, v, vel) for t, v, vel in events
            if chunk_start <= t < chunk_end
        ]
        
        # Trigger nodes
        for event_time, voice, velocity in chunk_events:
            if abs(event_time) < 0.01:  # At start of chunk
                node_id = f"drum_{voice.value}"
                if node_id in rhythm_network.nodes:
                    node = rhythm_network.nodes[node_id]
                    if isinstance(node, MusicalNode):
                        node.trigger(velocity)
        
        # Generate chunk audio
        chunk_audio = rhythm_network.generate_musical_signals(
            chunk_duration, sample_rate
        )
        
        # Add to full audio
        start_sample = int(chunk_start * sample_rate)
        end_sample = start_sample + len(chunk_audio)
        if end_sample <= len(audio):
            audio[start_sample:end_sample] += chunk_audio
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Save audio
    output_file = "rhythm_engine_demo.wav"
    wavfile.write(output_file, sample_rate, (audio * 32767).astype(np.int16))
    print(f"\n✓ Saved rhythm demo to {output_file}")
    
    # Visualize rhythm pattern
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle("Rhythm Engine Visualization")
    
    # Event plot
    for event_time, voice, velocity in events[:100]:  # First few events
        voice_y = {
            DrumVoice.KICK: 0,
            DrumVoice.SNARE: 1,
            DrumVoice.HIHAT_CLOSED: 2,
            DrumVoice.HIHAT_OPEN: 3
        }.get(voice, 4)
        
        ax1.scatter(event_time, voice_y, s=velocity*100, alpha=0.7)
        ax1.vlines(event_time, voice_y - 0.1, voice_y + 0.1, alpha=0.3)
    
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['Kick', 'Snare', 'HH Closed', 'HH Open'])
    ax1.set_xlabel("Time (s)")
    ax1.set_title("Rhythm Pattern")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 4)
    
    # Waveform
    time = np.arange(len(audio)) / sample_rate
    ax2.plot(time[:sample_rate*2], audio[:sample_rate*2], linewidth=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Generated Drum Audio")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("rhythm_engine_visualization.png", dpi=150)
    plt.close()
    
    print("✓ Saved visualization to rhythm_engine_visualization.png")
    
    # Demo polyrhythm
    print("\n6. Creating polyrhythm...")
    poly = rhythm.generate_polyrhythm('basic_rock', 'clave', ratio=(3, 2))
    print(f"   Created polyrhythm with {len(poly.pattern)} events")


def demo_composition_engine():
    """Demonstrate full composition capabilities."""
    print("\n" + "="*60)
    print("🎼 COMPOSITION ENGINE DEMO")
    print("="*60)
    
    # Try different musical styles
    styles = [
        MusicalStyle.CLASSICAL,
        MusicalStyle.JAZZ,
        MusicalStyle.ELECTRONIC,
        MusicalStyle.AMBIENT
    ]
    
    for style in styles[:2]:  # Demo first two styles
        print(f"\n{'='*40}")
        print(f"Generating {style.value.upper()} composition...")
        print('='*40)
        
        # Create composition engine
        tempo = {
            MusicalStyle.CLASSICAL: 100,
            MusicalStyle.JAZZ: 140,
            MusicalStyle.ELECTRONIC: 128,
            MusicalStyle.AMBIENT: 60
        }[style]
        
        composer = CompositionEngine(
            style=style,
            base_tempo=tempo,
            time_signature=TimeSignature(4, 4)
        )
        
        # Create structure
        form = {
            MusicalStyle.CLASSICAL: "AABA",
            MusicalStyle.JAZZ: "AABA",
            MusicalStyle.ELECTRONIC: "ABAB",
            MusicalStyle.AMBIENT: "ABC"
        }[style]
        
        structure = composer.create_structure(form=form, section_measures=4)
        print(f"\n1. Created structure: {form}")
        print(f"   Total measures: {structure.total_measures}")
        
        # Generate harmonic progression
        progression_type = {
            MusicalStyle.CLASSICAL: HarmonicProgression.I_IV_V_I,
            MusicalStyle.JAZZ: HarmonicProgression.ii_V_I,
            MusicalStyle.ELECTRONIC: HarmonicProgression.I_V_vi_IV,
            MusicalStyle.AMBIENT: HarmonicProgression.MODAL_INTERCHANGE
        }[style]
        
        progression = composer.generate_harmonic_progression(
            progression_type,
            structure.total_measures
        )
        print(f"\n2. Generated progression: {progression_type.value}")
        
        # Generate melody
        melody = composer.generate_melody(
            structure.total_measures,
            notes_per_measure=8
        )
        print(f"\n3. Generated melody with {len(melody)} notes")
        
        # Generate bass line
        bass = composer.generate_bass_line(progression, notes_per_chord=4)
        print(f"\n4. Generated bass line with {len(bass)} notes")
        
        # Build arrangement
        network = composer.build_arrangement()
        print(f"\n5. Built arrangement with {len(network.nodes)} voices")
        
        # Render composition
        duration = 20.0  # 20 seconds
        print(f"\n6. Rendering {duration}s of audio...")
        
        audio = composer.render_composition(duration)
        
        # Save audio
        output_file = f"composition_{style.value}_demo.wav"
        wavfile.write(output_file, 44100, (audio * 32767).astype(np.int16))
        print(f"\n✓ Saved {style.value} composition to {output_file}")
        
        # Visualize structure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{style.value.title()} Composition Analysis")
        
        # Section structure
        section_times = []
        section_names = []
        current_time = 0
        
        for section in structure.sections:
            section_times.append(current_time)
            section_names.append(section.name.split('_')[1])  # Get letter
            current_time += section.duration_measures * (60 / tempo) * 4
            
        colors = {'A': 'blue', 'B': 'red', 'C': 'green'}
        
        for i, (time, name) in enumerate(zip(section_times, section_names)):
            if i < len(section_times) - 1:
                duration = section_times[i+1] - time
            else:
                duration = 20 - time
                
            axes[0, 0].barh(0, duration, left=time, height=0.5,
                          color=colors.get(name, 'gray'), alpha=0.7,
                          label=name if name not in [n for t, n in zip(section_times[:i], section_names[:i])] else '')
            
        axes[0, 0].set_xlim(0, 20)
        axes[0, 0].set_ylim(-0.5, 0.5)
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_title("Song Structure")
        axes[0, 0].legend()
        axes[0, 0].set_yticks([])
        
        # Harmonic rhythm
        chord_changes = []
        for i, (degree, chord_type) in enumerate(progression[:8]):
            chord_changes.append((i, degree))
            
        axes[0, 1].step([c[0] for c in chord_changes], 
                       [c[1] for c in chord_changes], 
                       where='post', linewidth=2)
        axes[0, 1].set_xlabel("Measure")
        axes[0, 1].set_ylabel("Scale Degree")
        axes[0, 1].set_title("Harmonic Progression")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Melody contour
        melody_subset = melody[:32]  # First 4 measures
        axes[1, 0].plot(melody_subset, marker='o', markersize=4, linewidth=1)
        axes[1, 0].set_xlabel("Note Index")
        axes[1, 0].set_ylabel("Frequency (Hz)")
        axes[1, 0].set_title("Melodic Contour")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Audio waveform
        time = np.arange(len(audio)) / 44100
        axes[1, 1].plot(time, audio, linewidth=0.5)
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Amplitude")
        axes[1, 1].set_title("Generated Audio")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"composition_{style.value}_visualization.png", dpi=150)
        plt.close()
        
        print(f"✓ Saved visualization to composition_{style.value}_visualization.png")


def demo_emotional_composition():
    """Demonstrate emotional aspects in composition."""
    print("\n" + "="*60)
    print("💖 EMOTIONAL COMPOSITION DEMO")
    print("="*60)
    
    # Create a composition that transitions through emotions
    print("\n1. Creating emotional journey composition...")
    
    composer = CompositionEngine(
        style=MusicalStyle.AMBIENT,
        base_tempo=80,
        time_signature=TimeSignature(4, 4)
    )
    
    # Create custom structure with emotional progression
    from synthnn.core import SectionStructure, CompositionStructure
    
    sections = [
        SectionStructure(
            name="Intro_Calm",
            duration_measures=4,
            dynamic_level=0.3,
            texture_density=0.2,
            emotion=EmotionCategory.CALM
        ),
        SectionStructure(
            name="Development_Melancholy",
            duration_measures=4,
            dynamic_level=0.5,
            texture_density=0.4,
            emotion=EmotionCategory.MELANCHOLY
        ),
        SectionStructure(
            name="Climax_Joy",
            duration_measures=4,
            dynamic_level=0.8,
            texture_density=0.7,
            emotion=EmotionCategory.JOY
        ),
        SectionStructure(
            name="Resolution_Love",
            duration_measures=4,
            dynamic_level=0.6,
            texture_density=0.5,
            emotion=EmotionCategory.LOVE
        )
    ]
    
    composer.structure = CompositionStructure(sections=sections, form="ABCD")
    
    print("\n2. Emotional journey:")
    for section in sections:
        print(f"   {section.name}: {section.emotion.value} "
              f"(dynamics: {section.dynamic_level:.1f})")
    
    # Generate composition
    print("\n3. Generating emotionally-aware composition...")
    
    # Use modal interchange for emotional variety
    composer.generate_harmonic_progression(
        HarmonicProgression.MODAL_INTERCHANGE,
        composer.structure.total_measures
    )
    
    # Generate expressive melody
    composer.generate_melody(
        composer.structure.total_measures,
        notes_per_measure=6
    )
    
    # Render
    duration = 30.0
    audio = composer.render_composition(duration)
    
    # Save
    output_file = "emotional_composition_demo.wav"
    wavfile.write(output_file, 44100, (audio * 32767).astype(np.int16))
    print(f"\n✓ Saved emotional composition to {output_file}")
    
    # Visualize emotional journey
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot emotional trajectory
    emotions = [s.emotion for s in sections]
    emotion_colors = {
        EmotionCategory.CALM: 'lightblue',
        EmotionCategory.MELANCHOLY: 'purple',
        EmotionCategory.JOY: 'yellow',
        EmotionCategory.LOVE: 'pink'
    }
    
    current_time = 0
    for section in sections:
        duration = section.duration_measures * (60 / 80) * 4  # Convert to seconds
        
        # Background color for emotion
        ax.axvspan(current_time, current_time + duration,
                  color=emotion_colors.get(section.emotion, 'gray'),
                  alpha=0.3)
        
        # Emotion label
        ax.text(current_time + duration/2, 0.9,
               section.emotion.value.title(),
               ha='center', va='center',
               transform=ax.get_xaxis_transform(),
               fontsize=12, fontweight='bold')
        
        # Dynamic level line
        times = np.linspace(current_time, current_time + duration, 100)
        dynamics = np.ones_like(times) * section.dynamic_level
        ax.plot(times, dynamics, 'k-', linewidth=2, alpha=0.7)
        
        current_time += duration
    
    # Add audio waveform
    time = np.arange(len(audio)) / 44100
    ax.plot(time, audio * 0.3 + 0.5, linewidth=0.5, alpha=0.5, color='darkblue')
    
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Dynamic Level / Amplitude")
    ax.set_title("Emotional Journey Through Music")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("emotional_composition_visualization.png", dpi=150)
    plt.close()
    
    print("✓ Saved visualization to emotional_composition_visualization.png")


def main():
    """Run all music demos."""
    print("\n" + "🎵"*30)
    print("SYNTHNN ADVANCED MUSIC DEMO")
    print("🎵"*30)
    print("\nThis demo showcases enhanced musical capabilities:")
    print("1. Advanced Synthesis - Multi-oscillator, filters, effects")
    print("2. Rhythm Engine - Grooves, humanization, polyrhythms")
    print("3. Composition Engine - Complete musical pieces")
    print("4. Emotional Music - Emotion-driven composition")
    
    # Create output directory
    os.makedirs("music_output", exist_ok=True)
    os.chdir("music_output")
    
    # Run demos
    demo_synthesis_capabilities()
    demo_rhythm_engine()
    demo_composition_engine()
    demo_emotional_composition()
    
    print("\n" + "🎉"*30)
    print("ALL MUSIC DEMOS COMPLETE!")
    print("🎉"*30)
    print("\nGenerated files in 'music_output' directory:")
    print("- Audio files (.wav) - Listen to the generated music")
    print("- Visualizations (.png) - See the musical structures")
    print("\nThese demos show SynthNN creating rich, expressive music")
    print("with advanced synthesis, rhythm, and compositional intelligence!")


if __name__ == "__main__":
    main() 