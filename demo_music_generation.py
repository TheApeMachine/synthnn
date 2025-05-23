#!/usr/bin/env python3
"""
Demonstration of the Modal Music Generation System

This script shows various ways to use the modal music generation framework:
1. Basic composition generation
2. Style-based generation
3. Emotional trajectory generation
4. Interactive generation
"""

import numpy as np
import matplotlib.pyplot as plt
from modal_music_generator import ModalMusicGenerator
from context_aware_detector import ContextAwareModeDetector
from scipy.io import wavfile

def demo_basic_generation():
    """Demonstrate basic music generation"""
    print("\n=== Basic Composition Generation ===")
    
    generator = ModalMusicGenerator(base_freq=440.0, sample_rate=44100)
    
    # Generate a simple composition
    print("Generating a 20-second verse-chorus composition...")
    audio = generator.generate_composition(
        duration_seconds=20,
        structure='verse-chorus'
    )
    
    generator.save_composition(audio, 'demo_basic.wav')
    
    # Show the modal progression
    print("\nModal progression:")
    modes = generator.generation_history['modes'][:10]
    print(" -> ".join(modes))
    
    return generator


def demo_style_based_generation():
    """Demonstrate generation with different musical styles"""
    print("\n=== Style-Based Generation ===")
    
    # Define musical styles with semantic tags
    styles = {
        'Classical': {
            'tags': ['peaceful', 'contemplative', 'bright'],
            'structure': 'ABAB',
            'base_freq': 440.0,
            'tempo': 90
        },
        'Dark Ambient': {
            'tags': ['dark', 'mysterious', 'tense'],
            'structure': 'through-composed',
            'base_freq': 220.0,
            'tempo': 60
        },
        'Heroic': {
            'tags': ['heroic', 'bright', 'powerful'],
            'structure': 'verse-chorus',
            'base_freq': 440.0,
            'tempo': 140
        }
    }
    
    for style_name, style_params in styles.items():
        print(f"\nGenerating {style_name} style...")
        
        # Create generator with style-specific parameters
        generator = ModalMusicGenerator(
            base_freq=style_params['base_freq'],
            sample_rate=44100
        )
        generator.tempo = style_params['tempo']
        
        # Override structure generator's semantic tags
        original_create = generator.structure_generator.create_structure
        
        def create_with_style_tags(total_duration, structure_type, tempo):
            structure = original_create(total_duration, structure_type, tempo)
            # Override semantic tags with style-specific ones
            for section in structure:
                section['semantic_tags'] = style_params['tags']
            return structure
        
        generator.structure_generator.create_structure = create_with_style_tags
        
        # Generate composition
        audio = generator.generate_composition(
            duration_seconds=15,
            structure=style_params['structure']
        )
        
        generator.save_composition(audio, f'demo_{style_name.lower().replace(" ", "_")}.wav')
        
        print(f"Generated {style_name} composition")
        print(f"Modes used: {set(generator.generation_history['modes'])}")


def demo_emotional_trajectory():
    """Demonstrate generation following an emotional trajectory"""
    print("\n=== Emotional Trajectory Generation ===")
    
    generator = ModalMusicGenerator()
    
    # Define emotional trajectory
    trajectory = [
        {'time': 0.0, 'emotion': 'peaceful', 'tags': ['peaceful', 'bright']},
        {'time': 0.2, 'emotion': 'contemplative', 'tags': ['contemplative', 'nostalgic']},
        {'time': 0.4, 'emotion': 'tense', 'tags': ['tense', 'mysterious']},
        {'time': 0.6, 'emotion': 'heroic', 'tags': ['heroic', 'bright']},
        {'time': 0.8, 'emotion': 'resolution', 'tags': ['peaceful', 'nostalgic']},
        {'time': 1.0, 'emotion': 'peaceful', 'tags': ['peaceful']}
    ]
    
    # Custom structure following emotional trajectory
    structure = []
    duration = 30  # seconds
    
    for i in range(len(trajectory) - 1):
        current = trajectory[i]
        next_point = trajectory[i + 1]
        
        section_duration = (next_point['time'] - current['time']) * duration
        
        # Determine section type based on emotion
        if current['emotion'] == 'peaceful':
            section_type = 'intro' if i == 0 else 'outro'
        elif current['emotion'] == 'heroic':
            section_type = 'chorus'
        elif current['emotion'] == 'tense':
            section_type = 'bridge'
        else:
            section_type = 'verse'
        
        structure.append({
            'name': f'{current["emotion"]}_{i}',
            'type': section_type,
            'duration': section_duration,
            'position': current['time'],
            'semantic_tags': current['tags']
        })
    
    # Generate with custom structure
    generator.structure_generator.create_structure = lambda d, s, t: structure
    
    print("Generating composition following emotional trajectory...")
    audio = generator.generate_composition(duration_seconds=duration)
    
    generator.save_composition(audio, 'demo_emotional_trajectory.wav')
    
    # Visualize emotional trajectory and modes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot emotional trajectory
    times = [p['time'] for p in trajectory]
    emotions = [p['emotion'] for p in trajectory]
    ax1.plot(times, range(len(emotions)), 'o-')
    ax1.set_yticks(range(len(emotions)))
    ax1.set_yticklabels(emotions)
    ax1.set_xlabel('Time (normalized)')
    ax1.set_title('Emotional Trajectory')
    ax1.grid(True, alpha=0.3)
    
    # Plot mode progression
    modes = generator.generation_history['modes']
    mode_names = list(generator.context_detector.mode_intervals.keys())
    mode_indices = [mode_names.index(m) for m in modes]
    
    ax2.plot(np.linspace(0, 1, len(mode_indices)), mode_indices, 'o-', markersize=4)
    ax2.set_yticks(range(len(mode_names)))
    ax2.set_yticklabels(mode_names)
    ax2.set_xlabel('Time (normalized)')
    ax2.set_title('Modal Progression')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_emotional_trajectory.png')
    plt.show()
    
    return generator


def demo_adaptive_generation():
    """Demonstrate adaptive generation that responds to analysis"""
    print("\n=== Adaptive Generation ===")
    
    generator = ModalMusicGenerator()
    detector = ContextAwareModeDetector()
    
    # Generate initial phrase
    print("Generating initial phrase...")
    initial_audio = generator.generate_composition(duration_seconds=5, structure='through-composed')
    
    # Analyze the generated music
    print("Analyzing generated music...")
    mode_probs = detector.get_mode_probabilities(initial_audio[:44100], generator.sample_rate)
    dominant_mode = max(mode_probs.items(), key=lambda x: x[1])[0]
    
    print(f"Detected dominant mode: {dominant_mode}")
    print("Mode probabilities:")
    for mode, prob in sorted(mode_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {mode}: {prob:.3f}")
    
    # Generate response based on analysis
    print("\nGenerating adaptive response...")
    
    # Choose complementary mode
    if dominant_mode in ['Ionian', 'Lydian']:
        response_tags = ['contemplative', 'nostalgic']
        target_mode = 'Dorian'
    elif dominant_mode in ['Phrygian', 'Locrian']:
        response_tags = ['bright', 'peaceful']
        target_mode = 'Ionian'
    else:
        response_tags = ['mysterious', 'ethereal']
        target_mode = 'Lydian'
    
    # Set initial mode for response
    generator.current_mode = target_mode
    
    # Generate response with specific tags
    generator.structure_generator.create_structure = lambda d, s, t: [{
        'name': 'response',
        'type': 'verse',
        'duration': d,
        'position': 0,
        'semantic_tags': response_tags
    }]
    
    response_audio = generator.generate_composition(duration_seconds=5)
    
    # Combine original and response
    combined = np.concatenate([initial_audio, response_audio])
    generator.save_composition(combined, 'demo_adaptive.wav')
    
    print(f"Generated adaptive response in {target_mode} mode")


def demo_microtonal_generation():
    """Demonstrate generation with microtonal/non-Western modes"""
    print("\n=== Microtonal/Non-Western Generation ===")
    
    # Define custom modes with microtonal intervals
    custom_modes = {
        'Arabic_Maqam': [1, 1.125, 1.3125, 1.5, 1.625, 1.875, 2],  # Quarter-tone intervals
        'Indian_Raga': [1, 1.0535, 1.1892, 1.3348, 1.4983, 1.6818, 1.8877],  # 22-shruti system approximation
        'Bohlen_Pierce': [1, 1.0905, 1.1892, 1.2968, 1.4142, 1.5422, 1.6818],  # Non-octave scale
        'Harmonic_Series': [1, 2, 3, 4, 5, 6, 7],  # Pure harmonic series
    }
    
    for mode_name, intervals in custom_modes.items():
        print(f"\nGenerating with {mode_name} mode...")
        
        # Create custom detector with extended modes
        custom_detector = ContextAwareModeDetector()
        custom_detector.mode_intervals[mode_name] = intervals
        custom_detector.mode_characteristics[mode_name] = {
            'brightness': 0.6,
            'stability': 0.7,
            'complexity': 0.8
        }
        
        # Create generator with custom detector
        generator = ModalMusicGenerator()
        generator.context_detector = custom_detector
        generator.current_mode = mode_name
        
        # Simple structure for demonstration
        generator.structure_generator.create_structure = lambda d, s, t: [{
            'name': mode_name,
            'type': 'verse',
            'duration': d,
            'position': 0,
            'semantic_tags': ['experimental', 'ethereal']
        }]
        
        audio = generator.generate_composition(duration_seconds=10)
        generator.save_composition(audio, f'demo_{mode_name.lower()}.wav')
        
        print(f"Generated {mode_name} composition")


def main():
    """Run all demonstrations"""
    print("Modal Music Generation System - Demonstrations")
    print("=" * 50)
    
    # Run demos
    demo_basic_generation()
    demo_style_based_generation()
    demo_emotional_trajectory()
    demo_adaptive_generation()
    demo_microtonal_generation()
    
    print("\n" + "=" * 50)
    print("All demonstrations complete!")
    print("Generated audio files:")
    print("  - demo_basic.wav")
    print("  - demo_classical.wav")
    print("  - demo_dark_ambient.wav") 
    print("  - demo_heroic.wav")
    print("  - demo_emotional_trajectory.wav")
    print("  - demo_adaptive.wav")
    print("  - demo_arabic_maqam.wav")
    print("  - demo_indian_raga.wav")
    print("  - demo_bohlen_pierce.wav")
    print("  - demo_harmonic_series.wav")


if __name__ == "__main__":
    main() 