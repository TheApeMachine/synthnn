import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from collections import deque
import json

from adaptive import AdaptiveModalNetwork
from hierarchical_modal import HierarchicalModalProcessor
from context_aware_detector import ContextAwareModeDetector
from abstract import ResonantNetwork


class ModalMusicGenerator:
    """
    Generates music using adaptive modal networks with intelligent progression
    and multi-scale temporal structure.
    """
    
    def __init__(self, base_freq=440.0, sample_rate=44100):
        self.base_freq = base_freq
        self.sample_rate = sample_rate
        
        # Core components
        self.context_detector = ContextAwareModeDetector(context_window=20)
        self.hierarchical_processor = HierarchicalModalProcessor(
            time_scales=[0.25, 1.0, 4.0],  # Note, phrase, section levels
            num_nodes=7
        )
        self.adaptive_network = AdaptiveModalNetwork(num_nodes_per_network=7)
        
        # Generation parameters
        self.current_mode = 'Ionian'
        self.tempo = 120  # BPM
        self.beat_duration = 60.0 / self.tempo  # seconds per beat
        
        # Musical structure
        self.structure_generator = MusicalStructureGenerator()
        self.phrase_generator = ModalPhraseGenerator(self.context_detector)
        self.rhythm_generator = RhythmGenerator()
        
        # Generation history
        self.generation_history = {
            'modes': [],
            'frequencies': [],
            'dynamics': [],
            'structure': []
        }
        
    def generate_composition(self, duration_seconds=30, structure='verse-chorus'):
        """
        Generate a complete musical composition.
        
        Args:
            duration_seconds: Total duration of the composition
            structure: Musical form ('verse-chorus', 'ABAB', 'through-composed')
            
        Returns:
            np.array: Generated audio signal
        """
        # Create musical structure
        structure_plan = self.structure_generator.create_structure(
            duration_seconds, 
            structure,
            self.tempo
        )
        
        # Initialize output
        total_samples = int(duration_seconds * self.sample_rate)
        composition = np.zeros(total_samples)
        
        current_sample = 0
        
        for section in structure_plan:
            print(f"Generating {section['name']} section...")
            
            # Generate section with appropriate context
            section_audio = self._generate_section(
                section['duration'],
                section['type'],
                section['position'],
                section.get('semantic_tags', [])
            )
            
            # Apply section-level processing
            section_audio = self._apply_section_processing(
                section_audio,
                section['type']
            )
            
            # Add to composition
            end_sample = min(current_sample + len(section_audio), total_samples)
            composition[current_sample:end_sample] = section_audio[:end_sample-current_sample]
            current_sample = end_sample
            
        # Apply final mastering
        composition = self._master_audio(composition)
        
        return composition
    
    def _generate_section(self, duration, section_type, position, semantic_tags):
        """Generate a musical section with coherent modal progression"""
        samples_needed = int(duration * self.sample_rate)
        section_audio = np.zeros(samples_needed)
        
        # Determine initial mode based on section type and position
        structural_hints = {
            'section': section_type,
            'position': position
        }
        
        # Generate phrases
        phrase_duration = 4 * self.beat_duration  # 4-beat phrases
        phrase_samples = int(phrase_duration * self.sample_rate)
        
        current_pos = 0
        phrase_count = 0
        
        while current_pos < samples_needed:
            # Update context for phrase generation
            phrase_position = current_pos / samples_needed
            
            # Generate phrase
            phrase_audio, phrase_mode = self._generate_phrase(
                phrase_duration,
                section_type,
                phrase_position,
                semantic_tags,
                phrase_count
            )
            
            # Blend with previous phrase for smooth transitions
            if current_pos > 0:
                overlap_samples = int(0.1 * self.sample_rate)  # 100ms overlap
                if current_pos >= overlap_samples:
                    # Crossfade
                    fade_in = np.linspace(0, 1, overlap_samples)
                    fade_out = np.linspace(1, 0, overlap_samples)
                    
                    section_audio[current_pos-overlap_samples:current_pos] *= fade_out
                    section_audio[current_pos-overlap_samples:current_pos] += phrase_audio[:overlap_samples] * fade_in
                    
                    # Add rest of phrase
                    end_pos = min(current_pos + len(phrase_audio) - overlap_samples, samples_needed)
                    section_audio[current_pos:end_pos] = phrase_audio[overlap_samples:end_pos-current_pos+overlap_samples]
                else:
                    end_pos = min(current_pos + len(phrase_audio), samples_needed)
                    section_audio[current_pos:end_pos] = phrase_audio[:end_pos-current_pos]
            else:
                end_pos = min(current_pos + len(phrase_audio), samples_needed)
                section_audio[current_pos:end_pos] = phrase_audio[:end_pos-current_pos]
            
            current_pos = end_pos
            phrase_count += 1
            
            # Update generation history
            self.generation_history['modes'].append(phrase_mode)
        
        return section_audio
    
    def _generate_phrase(self, duration, section_type, position, semantic_tags, phrase_index):
        """Generate a musical phrase using modal networks"""
        # Determine mode for this phrase
        if phrase_index == 0 or np.random.random() < 0.3:  # Mode change probability
            # Get intelligent mode suggestion
            test_signal = self._create_test_signal(self.current_mode)
            
            suggested_mode = self.context_detector.analyze_with_context(
                test_signal,
                self.sample_rate,
                semantic_tags=semantic_tags,
                structural_hints={'section': section_type, 'position': position}
            )
            
            # Consider mode transition path
            if suggested_mode != self.current_mode:
                transition_path = self.context_detector.suggest_mode_transition(
                    self.current_mode,
                    test_signal,
                    self.sample_rate
                )
                
                # Take next step in transition
                if len(transition_path) > 1:
                    self.current_mode = transition_path[1]
                else:
                    self.current_mode = suggested_mode
            else:
                self.current_mode = suggested_mode
        
        # Generate melodic contour
        melody_contour = self.phrase_generator.generate_contour(
            self.current_mode,
            phrase_length=int(duration * 4),  # Quarter note resolution
            section_type=section_type
        )
        
        # Generate rhythm pattern
        rhythm_pattern = self.rhythm_generator.generate_pattern(
            duration,
            section_type,
            complexity=0.5 + 0.3 * position  # Increase complexity over time
        )
        
        # Convert to audio using resonant networks
        phrase_audio = self._render_phrase(
            melody_contour,
            rhythm_pattern,
            self.current_mode,
            duration
        )
        
        return phrase_audio, self.current_mode
    
    def _render_phrase(self, melody_contour, rhythm_pattern, mode, duration):
        """Render a phrase using the resonant networks"""
        samples = int(duration * self.sample_rate)
        audio = np.zeros(samples)
        
        # Get mode intervals
        mode_intervals = self.context_detector.mode_intervals[mode]
        
        # Create resonant network for this mode
        mode_network = ResonantNetwork(
            num_nodes=len(mode_intervals),
            base_freq=self.base_freq,
            harmonic_ratios=mode_intervals,
            target_phase=0,
            target_amplitude=0.5
        )
        
        # Render each note
        note_index = 0
        current_sample = 0
        
        for i, (note_degree, duration_beats) in enumerate(zip(melody_contour, rhythm_pattern)):
            note_duration = duration_beats * self.beat_duration
            note_samples = int(note_duration * self.sample_rate)
            
            if note_degree > 0:  # Not a rest
                # Calculate frequency based on mode degree
                if note_degree <= len(mode_intervals):
                    freq_ratio = mode_intervals[note_degree - 1]
                else:
                    # Wrap around for extended range
                    octave = (note_degree - 1) // len(mode_intervals)
                    degree_in_octave = (note_degree - 1) % len(mode_intervals)
                    freq_ratio = mode_intervals[degree_in_octave] * (2 ** octave)
                
                note_freq = self.base_freq * freq_ratio
                
                # Generate note using resonant synthesis
                t = np.linspace(0, note_duration, note_samples)
                
                # Create rich harmonic content
                note_audio = np.zeros(note_samples)
                
                # Fundamental
                note_audio += 0.6 * np.sin(2 * np.pi * note_freq * t)
                
                # Add harmonics based on mode character
                if mode in ['Ionian', 'Lydian']:  # Bright modes
                    note_audio += 0.2 * np.sin(2 * np.pi * note_freq * 2 * t)
                    note_audio += 0.1 * np.sin(2 * np.pi * note_freq * 3 * t)
                elif mode in ['Phrygian', 'Locrian']:  # Dark modes
                    note_audio += 0.15 * np.sin(2 * np.pi * note_freq * 1.5 * t)
                    note_audio += 0.1 * np.sin(2 * np.pi * note_freq * 2.5 * t)
                else:  # Neutral modes
                    note_audio += 0.15 * np.sin(2 * np.pi * note_freq * 2 * t)
                    note_audio += 0.1 * np.sin(2 * np.pi * note_freq * 4 * t)
                
                # Apply envelope
                envelope = self._create_envelope(note_samples, note_duration)
                note_audio *= envelope
                
                # Add to output
                end_sample = min(current_sample + note_samples, samples)
                audio[current_sample:end_sample] = note_audio[:end_sample-current_sample]
            
            current_sample += note_samples
            if current_sample >= samples:
                break
        
        # Process through adaptive network for modal coherence
        audio_processed = np.zeros_like(audio)
        chunk_size = 1024
        
        for i in range(0, len(audio) - chunk_size, chunk_size):
            chunk = audio[i:i+chunk_size]
            processed_value = self.adaptive_network.process(chunk, self.sample_rate)
            # Use processed value to modulate the chunk
            audio_processed[i:i+chunk_size] = chunk * (0.8 + 0.2 * np.tanh(processed_value))
        
        return audio_processed
    
    def _create_envelope(self, num_samples, duration):
        """Create an ADSR envelope for a note"""
        attack_time = min(0.02, duration * 0.1)
        decay_time = min(0.05, duration * 0.2)
        sustain_level = 0.7
        release_time = min(0.1, duration * 0.3)
        
        attack_samples = int(attack_time * self.sample_rate)
        decay_samples = int(decay_time * self.sample_rate)
        release_samples = int(release_time * self.sample_rate)
        sustain_samples = num_samples - attack_samples - decay_samples - release_samples
        
        envelope = np.concatenate([
            np.linspace(0, 1, attack_samples),
            np.linspace(1, sustain_level, decay_samples),
            np.ones(max(0, sustain_samples)) * sustain_level,
            np.linspace(sustain_level, 0, release_samples)
        ])
        
        return envelope[:num_samples]
    
    def _create_test_signal(self, current_mode):
        """Create a test signal representing the current musical context"""
        # Generate a short signal with characteristics of the current mode
        duration = 0.5
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        mode_intervals = self.context_detector.mode_intervals[current_mode]
        signal = np.zeros_like(t)
        
        # Add key frequencies from the mode
        for i, interval in enumerate(mode_intervals[:3]):
            freq = self.base_freq * interval
            signal += (0.5 / (i + 1)) * np.sin(2 * np.pi * freq * t)
        
        return signal
    
    def _apply_section_processing(self, audio, section_type):
        """Apply section-specific audio processing"""
        if section_type == 'intro':
            # Fade in
            fade_samples = int(0.5 * self.sample_rate)
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        
        elif section_type == 'chorus':
            # Add slight compression/excitement
            audio = np.tanh(audio * 1.2) * 0.9
        
        elif section_type == 'bridge':
            # Add some filtering effect
            # Simple high-pass effect
            from scipy.signal import butter, filtfilt
            b, a = butter(2, 200 / (self.sample_rate / 2), 'high')
            audio = filtfilt(b, a, audio) * 0.8 + audio * 0.2
        
        elif section_type == 'outro':
            # Fade out
            fade_samples = int(2.0 * self.sample_rate)
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return audio
    
    def _master_audio(self, audio):
        """Apply final mastering to the composition"""
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8
        
        # Soft limiting
        audio = np.tanh(audio * 0.9) * 0.95
        
        return audio
    
    def save_composition(self, audio, filename):
        """Save the generated composition to a WAV file"""
        # Convert to 16-bit integer
        audio_int = (audio * 32767).astype(np.int16)
        wavfile.write(filename, self.sample_rate, audio_int)
        print(f"Composition saved to {filename}")
    
    def plot_generation_history(self):
        """Visualize the generation history"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot mode progression
        mode_names = self.generation_history['modes']
        mode_indices = [list(self.context_detector.mode_intervals.keys()).index(m) 
                       for m in mode_names]
        
        ax1.plot(mode_indices, 'o-')
        ax1.set_yticks(range(len(self.context_detector.mode_intervals)))
        ax1.set_yticklabels(list(self.context_detector.mode_intervals.keys()))
        ax1.set_xlabel('Phrase Index')
        ax1.set_ylabel('Mode')
        ax1.set_title('Modal Progression')
        ax1.grid(True, alpha=0.3)
        
        # Plot structure
        if 'structure' in self.generation_history and self.generation_history['structure']:
            sections = self.generation_history['structure']
            colors = {'intro': 'blue', 'verse': 'green', 'chorus': 'red', 
                     'bridge': 'purple', 'outro': 'orange'}
            
            for i, section in enumerate(sections):
                color = colors.get(section['type'], 'gray')
                ax2.barh(0, section['duration'], left=section['start_time'], 
                        color=color, alpha=0.6, label=section['type'])
            
            ax2.set_ylim(-0.5, 0.5)
            ax2.set_xlabel('Time (seconds)')
            ax2.set_title('Song Structure')
            ax2.legend()
        
        plt.tight_layout()
        plt.show()


class MusicalStructureGenerator:
    """Generates high-level musical structure"""
    
    def create_structure(self, total_duration, structure_type, tempo):
        """Create a musical structure plan"""
        structures = {
            'verse-chorus': [
                {'name': 'intro', 'type': 'intro', 'duration_beats': 8},
                {'name': 'verse1', 'type': 'verse', 'duration_beats': 16},
                {'name': 'chorus1', 'type': 'chorus', 'duration_beats': 16},
                {'name': 'verse2', 'type': 'verse', 'duration_beats': 16},
                {'name': 'chorus2', 'type': 'chorus', 'duration_beats': 16},
                {'name': 'bridge', 'type': 'bridge', 'duration_beats': 8},
                {'name': 'chorus3', 'type': 'chorus', 'duration_beats': 16},
                {'name': 'outro', 'type': 'outro', 'duration_beats': 8}
            ],
            'ABAB': [
                {'name': 'intro', 'type': 'intro', 'duration_beats': 4},
                {'name': 'A1', 'type': 'verse', 'duration_beats': 16},
                {'name': 'B1', 'type': 'chorus', 'duration_beats': 16},
                {'name': 'A2', 'type': 'verse', 'duration_beats': 16},
                {'name': 'B2', 'type': 'chorus', 'duration_beats': 16},
                {'name': 'outro', 'type': 'outro', 'duration_beats': 4}
            ],
            'through-composed': [
                {'name': 'section1', 'type': 'intro', 'duration_beats': 8},
                {'name': 'section2', 'type': 'verse', 'duration_beats': 12},
                {'name': 'section3', 'type': 'bridge', 'duration_beats': 8},
                {'name': 'section4', 'type': 'verse', 'duration_beats': 12},
                {'name': 'section5', 'type': 'outro', 'duration_beats': 8}
            ]
        }
        
        structure = structures.get(structure_type, structures['verse-chorus'])
        
        # Convert beats to seconds and calculate positions
        beat_duration = 60.0 / tempo
        current_time = 0
        
        processed_structure = []
        for section_template in structure:
            # Make a copy to avoid modifying the original template
            section = section_template.copy()
            
            section_duration_seconds = section['duration_beats'] * beat_duration
            
            # If this section would exceed total_duration, cap it
            if current_time + section_duration_seconds > total_duration and current_time < total_duration:
                section_duration_seconds = total_duration - current_time
            elif current_time >= total_duration:
                break # We have already filled the total duration
                
            section['duration'] = section_duration_seconds
            section['start_time'] = current_time
            section['position'] = current_time / total_duration if total_duration > 0 else 0
            
            # Add semantic tags based on section type
            if section['type'] == 'intro':
                section['semantic_tags'] = ['peaceful', 'bright']
            elif section['type'] == 'verse':
                section['semantic_tags'] = ['contemplative', 'nostalgic']
            elif section['type'] == 'chorus':
                section['semantic_tags'] = ['heroic', 'bright']
            elif section['type'] == 'bridge':
                section['semantic_tags'] = ['mysterious', 'tense']
            elif section['type'] == 'outro':
                section['semantic_tags'] = ['peaceful', 'nostalgic']
            else:
                section['semantic_tags'] = [] # Default empty list
            
            processed_structure.append(section)
            current_time += section_duration_seconds
        
        return processed_structure


class ModalPhraseGenerator:
    """Generates melodic phrases based on modal characteristics"""
    
    def __init__(self, mode_detector):
        self.mode_detector = mode_detector
        
        # Define melodic patterns for different modes
        self.mode_patterns = {
            'Ionian': {
                'tendencies': [1, 3, 5, 8],  # Strong notes
                'avoid': [],
                'cadence': [5, 4, 3, 2, 1]  # Typical cadence
            },
            'Dorian': {
                'tendencies': [1, 3, 6, 5],
                'avoid': [],
                'cadence': [6, 5, 4, 1]
            },
            'Phrygian': {
                'tendencies': [1, 2, 5],
                'avoid': [],
                'cadence': [2, 1]
            },
            'Lydian': {
                'tendencies': [1, 3, 4, 5],
                'avoid': [],
                'cadence': [4, 3, 2, 1]
            },
            'Mixolydian': {
                'tendencies': [1, 3, 5, 7],
                'avoid': [],
                'cadence': [7, 1]
            },
            'Aeolian': {
                'tendencies': [1, 3, 5, 6],
                'avoid': [],
                'cadence': [6, 5, 1]
            },
            'Locrian': {
                'tendencies': [1, 5],
                'avoid': [3],
                'cadence': [5, 1]
            }
        }
    
    def generate_contour(self, mode, phrase_length=8, section_type='verse'):
        """Generate a melodic contour for a phrase"""
        pattern_info = self.mode_patterns.get(mode, self.mode_patterns['Ionian'])
        contour = []
        
        # Starting note
        start_options = pattern_info['tendencies']
        current_degree = np.random.choice(start_options)
        contour.append(current_degree)
        
        # Generate phrase
        for i in range(1, phrase_length - 1):
            # Determine next note based on musical rules
            if i == phrase_length // 2:  # Midpoint - create some tension
                # Jump to create interest
                jump_options = [d for d in range(1, 8) 
                              if abs(d - current_degree) > 2 and 
                              d not in pattern_info['avoid']]
                if jump_options:
                    current_degree = np.random.choice(jump_options)
                else:
                    current_degree = np.random.choice(pattern_info['tendencies'])
            else:
                # Stepwise motion with occasional jumps
                if np.random.random() < 0.7:  # Stepwise
                    step = np.random.choice([-1, 1])
                    current_degree = max(1, min(8, current_degree + step))
                else:  # Jump
                    jump = np.random.choice([-3, -2, 2, 3])
                    current_degree = max(1, min(8, current_degree + jump))
            
            # Avoid notes if specified
            if current_degree in pattern_info['avoid']:
                current_degree = np.random.choice(pattern_info['tendencies'])
            
            contour.append(current_degree)
        
        # Ending - use cadence pattern
        if section_type in ['verse', 'chorus', 'outro']:
            # Strong cadence
            cadence = pattern_info['cadence']
            if len(cadence) <= phrase_length - len(contour):
                contour.extend(cadence[:phrase_length - len(contour)])
            else:
                contour.append(1)  # Return to tonic
        else:
            # Weak/open ending for bridges
            contour.append(np.random.choice([2, 5, 7]))
        
        return contour


class RhythmGenerator:
    """Generates rhythm patterns"""
    
    def generate_pattern(self, duration, section_type, complexity=0.5):
        """Generate a rhythm pattern"""
        # Basic rhythm patterns (in beats)
        patterns = {
            'simple': [1, 1, 1, 1],
            'syncopated': [0.5, 1, 0.5, 1, 1],
            'complex': [0.5, 0.5, 0.25, 0.25, 0.5, 1],
            'driving': [0.5, 0.5, 0.5, 0.5],
            'sparse': [2, 1, 1]
        }
        
        # Select pattern based on section and complexity
        if section_type == 'intro' or section_type == 'outro':
            pattern_choice = 'sparse'
        elif section_type == 'chorus':
            pattern_choice = 'driving' if complexity > 0.6 else 'simple'
        elif section_type == 'bridge':
            pattern_choice = 'syncopated'
        else:  # verse
            pattern_choice = 'simple' if complexity < 0.5 else 'syncopated'
        
        base_pattern = patterns[pattern_choice]
        
        # Repeat pattern to fill duration
        total_beats = duration / (60.0 / 120.0)  # Assuming 120 BPM base
        rhythm_pattern = []
        
        while sum(rhythm_pattern) < total_beats:
            rhythm_pattern.extend(base_pattern)
        
        # Trim to exact duration
        cumsum = 0
        trimmed_pattern = []
        for beat in rhythm_pattern:
            if cumsum + beat <= total_beats:
                trimmed_pattern.append(beat)
                cumsum += beat
            else:
                # Add partial beat
                remaining = total_beats - cumsum
                if remaining > 0.1:
                    trimmed_pattern.append(remaining)
                break
        
        return trimmed_pattern


def demonstrate_music_generation():
    """Demonstrate the music generation system"""
    print("Modal Music Generator Demo")
    print("=" * 50)
    
    # Create generator
    generator = ModalMusicGenerator(base_freq=440.0, sample_rate=44100)
    
    # Generate a short composition
    print("\nGenerating a 30-second composition...")
    audio = generator.generate_composition(
        duration_seconds=30,
        structure='verse-chorus'
    )
    
    # Save the composition
    generator.save_composition(audio, 'modal_composition.wav')
    
    # Plot generation history
    generator.plot_generation_history()
    
    # Generate another piece with different structure
    print("\nGenerating a through-composed piece...")
    audio2 = generator.generate_composition(
        duration_seconds=20,
        structure='through-composed'
    )
    
    generator.save_composition(audio2, 'modal_throughcomposed.wav')
    
    print("\nGeneration complete!")
    print(f"Mode progression: {' -> '.join(generator.generation_history['modes'][:10])}...")


if __name__ == "__main__":
    demonstrate_music_generation() 