"""
Composition Engine for SynthNN

High-level musical composition system that combines synthesis, rhythm,
harmony, and structure to create complete musical pieces.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import random

from .resonant_network import ResonantNetwork
from .musical_synthesis import (
    MusicalNode, MusicalResonantNetwork, MusicalSynthesizer,
    WaveShape, FilterType, ADSREnvelope
)
from .rhythm_engine import (
    RhythmEngine, RhythmPattern, GrooveTemplate,
    DrumVoice, TimeSignature
)
from .emotional_resonance import EmotionalResonanceEngine, EmotionCategory


class MusicalStyle(Enum):
    """Musical style presets."""
    CLASSICAL = "classical"
    JAZZ = "jazz"
    ELECTRONIC = "electronic"
    AMBIENT = "ambient"
    ROCK = "rock"
    EXPERIMENTAL = "experimental"
    WORLD = "world"
    MINIMALIST = "minimalist"


class HarmonicProgression(Enum):
    """Common chord progression types."""
    I_IV_V_I = "I-IV-V-I"
    ii_V_I = "ii-V-I"
    I_V_vi_IV = "I-V-vi-IV"
    I_vi_IV_V = "I-vi-IV-V"
    BLUES_12_BAR = "blues_12_bar"
    MODAL_INTERCHANGE = "modal_interchange"
    CHROMATIC = "chromatic"
    CUSTOM = "custom"


@dataclass
class SectionStructure:
    """Defines a section of a composition."""
    name: str
    duration_measures: int
    tempo_change: Optional[float] = None
    key_change: Optional[str] = None
    dynamic_level: float = 0.7  # 0-1
    texture_density: float = 0.5  # 0-1
    emotion: Optional[EmotionCategory] = None


@dataclass
class CompositionStructure:
    """Overall structure of a composition."""
    sections: List[SectionStructure]
    form: str = "AABA"  # Song form
    total_measures: int = 0
    
    def __post_init__(self):
        """Calculate total measures."""
        self.total_measures = sum(s.duration_measures for s in self.sections)


class VoiceLeading:
    """Handles voice leading between chords."""
    
    @staticmethod
    def smooth_voice_leading(current_chord: List[float], 
                           next_chord: List[float]) -> List[float]:
        """
        Apply smooth voice leading between chords.
        
        Args:
            current_chord: Current chord frequencies
            next_chord: Target chord frequencies
            
        Returns:
            Rearranged next_chord for minimal movement
        """
        if len(current_chord) != len(next_chord):
            return next_chord
            
        # Find optimal voice assignment to minimize movement
        from itertools import permutations
        
        best_arrangement = next_chord
        min_movement = float('inf')
        
        for perm in permutations(next_chord):
            movement = sum(abs(c - n) for c, n in zip(current_chord, perm))
            if movement < min_movement:
                min_movement = movement
                best_arrangement = list(perm)
                
        return best_arrangement
        
    @staticmethod
    def apply_voice_leading_rules(melody_note: float, 
                                chord_notes: List[float]) -> List[float]:
        """
        Apply classical voice leading rules.
        
        Args:
            melody_note: Current melody note frequency
            chord_notes: Chord frequencies
            
        Returns:
            Adjusted chord notes
        """
        adjusted = chord_notes.copy()
        
        # Avoid parallel fifths and octaves
        for i in range(len(adjusted)):
            ratio = melody_note / adjusted[i]
            
            # If perfect fifth (3:2) or octave (2:1), slightly detune
            if abs(ratio - 1.5) < 0.01 or abs(ratio - 2.0) < 0.01:
                adjusted[i] *= 1.01  # Slight detuning
                
        return adjusted


class MelodicGenerator:
    """Generates melodic lines based on various algorithms."""
    
    def __init__(self, scale_frequencies: List[float]):
        self.scale_frequencies = scale_frequencies
        
    def generate_stepwise_melody(self, length: int, 
                                range_octaves: float = 1.5) -> List[float]:
        """Generate melody with mostly stepwise motion."""
        melody = []
        current_idx = len(self.scale_frequencies) // 2
        
        for _ in range(length):
            melody.append(self.scale_frequencies[current_idx])
            
            # Stepwise motion with occasional leaps
            if random.random() < 0.8:  # 80% stepwise
                step = random.choice([-1, 1])
            else:  # 20% leaps
                step = random.choice([-3, -2, 2, 3])
                
            # Keep within range
            new_idx = current_idx + step
            if 0 <= new_idx < len(self.scale_frequencies):
                current_idx = new_idx
                
        return melody
        
    def generate_motivic_melody(self, motif: List[float], 
                              length: int) -> List[float]:
        """Generate melody based on a motif with variations."""
        melody = []
        
        while len(melody) < length:
            # Use motif with variations
            variation = random.choice(['exact', 'transpose', 'invert', 'retrograde'])
            
            if variation == 'exact':
                melody.extend(motif)
            elif variation == 'transpose':
                # Transpose by scale degree
                transpose_steps = random.choice([-2, -1, 1, 2])
                transposed = []
                for note in motif:
                    if note in self.scale_frequencies:
                        idx = self.scale_frequencies.index(note)
                        new_idx = (idx + transpose_steps) % len(self.scale_frequencies)
                        transposed.append(self.scale_frequencies[new_idx])
                    else:
                        transposed.append(note)
                melody.extend(transposed)
            elif variation == 'invert':
                # Invert intervals
                if motif:
                    first_note = motif[0]
                    inverted = [first_note]
                    for i in range(1, len(motif)):
                        interval = motif[i] / motif[i-1]
                        inverted.append(inverted[-1] / interval)
                    melody.extend(inverted)
            elif variation == 'retrograde':
                # Reverse
                melody.extend(reversed(motif))
                
        return melody[:length]


class CompositionEngine:
    """
    Main composition engine for creating complete musical pieces.
    """
    
    def __init__(self, style: MusicalStyle = MusicalStyle.CLASSICAL,
                 base_tempo: float = 120.0,
                 time_signature: TimeSignature = None):
        """
        Initialize composition engine.
        
        Args:
            style: Musical style preset
            base_tempo: Base tempo in BPM
            time_signature: Time signature
        """
        self.style = style
        self.base_tempo = base_tempo
        self.time_signature = time_signature or TimeSignature(4, 4)
        
        # Core components
        self.musical_network = MusicalResonantNetwork(
            name="composition_network",
            mode="ionian",
            base_freq=440.0
        )
        
        self.rhythm_engine = RhythmEngine(
            tempo=base_tempo,
            time_signature=self.time_signature
        )
        
        self.emotional_engine = EmotionalResonanceEngine()
        
        # Composition state
        self.structure: Optional[CompositionStructure] = None
        self.current_section_idx = 0
        self.current_measure = 0
        
        # Musical elements
        self.harmonic_progression: List[Tuple[int, str]] = []
        self.melody_line: List[float] = []
        self.bass_line: List[float] = []
        
        # Initialize style-specific settings
        self._init_style_settings()
        
    def _init_style_settings(self):
        """Initialize settings based on musical style."""
        style_settings = {
            MusicalStyle.CLASSICAL: {
                'modes': ['ionian', 'dorian', 'aeolian'],
                'rhythm_patterns': ['basic_rock', 'snare_backbeat'],
                'tempo_range': (60, 140),
                'dynamics_range': (0.3, 0.9),
                'preferred_progression': HarmonicProgression.I_IV_V_I
            },
            MusicalStyle.JAZZ: {
                'modes': ['dorian', 'mixolydian', 'lydian'],
                'rhythm_patterns': ['jazz_ride', 'swing'],
                'tempo_range': (80, 200),
                'dynamics_range': (0.4, 0.8),
                'preferred_progression': HarmonicProgression.ii_V_I
            },
            MusicalStyle.ELECTRONIC: {
                'modes': ['phrygian', 'locrian', 'chromatic'],
                'rhythm_patterns': ['four_on_floor', 'hihat_16ths'],
                'tempo_range': (110, 160),
                'dynamics_range': (0.6, 1.0),
                'preferred_progression': HarmonicProgression.I_V_vi_IV
            },
            MusicalStyle.AMBIENT: {
                'modes': ['lydian', 'ionian', 'dorian'],
                'rhythm_patterns': [],  # No drums
                'tempo_range': (50, 90),
                'dynamics_range': (0.1, 0.5),
                'preferred_progression': HarmonicProgression.MODAL_INTERCHANGE
            }
        }
        
        # Apply style settings
        if self.style in style_settings:
            settings = style_settings[self.style]
            
            # Set mode
            if settings['modes']:
                mode = random.choice(settings['modes'])
                self.musical_network.mode = mode
                self.musical_network.mode_intervals = self.musical_network._get_mode_intervals(mode)
                
            # Activate rhythm patterns
            for pattern in settings['rhythm_patterns']:
                if pattern in self.rhythm_engine.patterns:
                    # Activate for appropriate drums
                    if 'four_on_floor' in pattern or 'kick' in pattern:
                        self.rhythm_engine.activate_pattern(pattern, DrumVoice.KICK)
                    elif 'snare' in pattern:
                        self.rhythm_engine.activate_pattern(pattern, DrumVoice.SNARE)
                    elif 'hihat' in pattern:
                        self.rhythm_engine.activate_pattern(pattern, DrumVoice.HIHAT_CLOSED)
                    elif 'ride' in pattern or 'jazz' in pattern:
                        self.rhythm_engine.activate_pattern(pattern, DrumVoice.RIDE)
                        
    def create_structure(self, form: str = "AABA", 
                        section_measures: int = 8) -> CompositionStructure:
        """
        Create composition structure.
        
        Args:
            form: Musical form (e.g., "AABA", "ABAB", "ABCBA")
            section_measures: Measures per section
            
        Returns:
            Composition structure
        """
        sections = []
        
        # Create sections based on form
        for i, section_letter in enumerate(form):
            # Determine section properties
            if section_letter == 'A':
                emotion = EmotionCategory.JOY
                dynamic = 0.7
                texture = 0.6
            elif section_letter == 'B':
                emotion = EmotionCategory.CALM
                dynamic = 0.5
                texture = 0.4
            elif section_letter == 'C':
                emotion = EmotionCategory.EXCITEMENT
                dynamic = 0.9
                texture = 0.8
            else:
                emotion = EmotionCategory.MELANCHOLY
                dynamic = 0.6
                texture = 0.5
                
            section = SectionStructure(
                name=f"Section_{section_letter}_{i}",
                duration_measures=section_measures,
                dynamic_level=dynamic,
                texture_density=texture,
                emotion=emotion
            )
            
            sections.append(section)
            
        self.structure = CompositionStructure(sections=sections, form=form)
        return self.structure
        
    def generate_harmonic_progression(self, 
                                    progression_type: HarmonicProgression,
                                    length_measures: int) -> List[Tuple[int, str]]:
        """
        Generate chord progression.
        
        Args:
            progression_type: Type of progression
            length_measures: Length in measures
            
        Returns:
            List of (root_degree, chord_type) tuples
        """
        progressions = {
            HarmonicProgression.I_IV_V_I: [
                (0, 'triad'), (3, 'triad'), (4, 'seventh'), (0, 'triad')
            ],
            HarmonicProgression.ii_V_I: [
                (1, 'seventh'), (4, 'seventh'), (0, 'triad')
            ],
            HarmonicProgression.I_V_vi_IV: [
                (0, 'triad'), (4, 'triad'), (5, 'triad'), (3, 'triad')
            ],
            HarmonicProgression.I_vi_IV_V: [
                (0, 'triad'), (5, 'triad'), (3, 'triad'), (4, 'seventh')
            ],
            HarmonicProgression.BLUES_12_BAR: [
                (0, 'seventh'), (0, 'seventh'), (0, 'seventh'), (0, 'seventh'),
                (3, 'seventh'), (3, 'seventh'), (0, 'seventh'), (0, 'seventh'),
                (4, 'seventh'), (3, 'seventh'), (0, 'seventh'), (4, 'seventh')
            ]
        }
        
        base_progression = progressions.get(
            progression_type,
            [(0, 'triad')]  # Default to tonic
        )
        
        # Extend to desired length
        progression = []
        for i in range(length_measures):
            chord = base_progression[i % len(base_progression)]
            progression.append(chord)
            
        self.harmonic_progression = progression
        return progression
        
    def generate_melody(self, length_measures: int,
                       notes_per_measure: int = 4) -> List[float]:
        """
        Generate a melodic line.
        
        Args:
            length_measures: Length in measures
            notes_per_measure: Notes per measure
            
        Returns:
            List of frequencies
        """
        # Create scale frequencies
        scale_frequencies = []
        for octave in range(-1, 2):  # 3 octaves
            for i, ratio in enumerate(self.musical_network.mode_intervals[:-1]):
                freq = self.musical_network.base_freq * ratio * (2 ** octave)
                scale_frequencies.append(freq)
                
        # Create melodic generator
        melodic_gen = MelodicGenerator(scale_frequencies)
        
        # Generate based on style
        total_notes = length_measures * notes_per_measure
        
        if self.style == MusicalStyle.MINIMALIST:
            # Simple repeating pattern
            pattern = melodic_gen.generate_stepwise_melody(4)
            melody = pattern * (total_notes // 4)
        elif self.style == MusicalStyle.JAZZ:
            # Motivic development
            motif = melodic_gen.generate_stepwise_melody(3)
            melody = melodic_gen.generate_motivic_melody(motif, total_notes)
        else:
            # General melodic line
            melody = melodic_gen.generate_stepwise_melody(total_notes)
            
        self.melody_line = melody
        return melody
        
    def generate_bass_line(self, harmonic_progression: List[Tuple[int, str]],
                          notes_per_chord: int = 2) -> List[float]:
        """
        Generate bass line following the harmony.
        
        Args:
            harmonic_progression: Chord progression
            notes_per_chord: Notes per chord
            
        Returns:
            List of frequencies
        """
        bass_line = []
        
        for root_degree, chord_type in harmonic_progression:
            # Get root frequency (one octave below)
            root_ratio = self.musical_network.mode_intervals[root_degree]
            root_freq = self.musical_network.base_freq * root_ratio / 2
            
            # Create pattern based on style
            if self.style == MusicalStyle.ROCK:
                # Repeated root
                pattern = [root_freq] * notes_per_chord
            elif self.style == MusicalStyle.JAZZ:
                # Walking bass
                pattern = [root_freq]
                for i in range(1, notes_per_chord):
                    # Walk up or down scale
                    if random.random() < 0.5:
                        pattern.append(root_freq * self.musical_network.mode_intervals[1])
                    else:
                        pattern.append(root_freq / self.musical_network.mode_intervals[1])
            else:
                # Root and fifth
                fifth_ratio = self.musical_network.mode_intervals[4]
                pattern = [root_freq, root_freq * fifth_ratio] * (notes_per_chord // 2)
                
            bass_line.extend(pattern[:notes_per_chord])
            
        self.bass_line = bass_line
        return bass_line
        
    def build_arrangement(self) -> MusicalResonantNetwork:
        """
        Build complete arrangement with all voices.
        
        Returns:
            Configured musical network
        """
        # Clear existing nodes
        self.musical_network.nodes.clear()
        
        # Add melodic voice
        melody_node = MusicalNode("melody", frequency=440.0)
        melody_node.synthesizer.add_oscillator(WaveShape.SAWTOOTH, 0.7)
        melody_node.synthesizer.add_oscillator(WaveShape.SINE, 0.3)
        melody_node.synthesizer.filter.filter_type = FilterType.LOWPASS
        melody_node.synthesizer.filter.cutoff = 2000
        melody_node.synthesizer.reverb_mix = 0.3
        self.musical_network.add_node(melody_node)
        
        # Add harmonic voices (chord)
        for i in range(4):  # 4-voice harmony
            harmony_node = MusicalNode(f"harmony_{i}", frequency=440.0)
            harmony_node.synthesizer.add_oscillator(WaveShape.TRIANGLE, 0.8)
            harmony_node.synthesizer.add_oscillator(WaveShape.SINE, 0.2)
            harmony_node.synthesizer.amplitude_envelope = ADSREnvelope(
                attack=0.05, decay=0.2, sustain=0.6, release=0.5
            )
            harmony_node.synthesizer.reverb_mix = 0.4
            self.musical_network.add_node(harmony_node)
            
        # Add bass voice
        bass_node = MusicalNode("bass", frequency=220.0)
        bass_node.synthesizer.add_oscillator(WaveShape.SINE, 0.6)
        bass_node.synthesizer.add_oscillator(WaveShape.SAWTOOTH, 0.4)
        bass_node.synthesizer.filter.filter_type = FilterType.LOWPASS
        bass_node.synthesizer.filter.cutoff = 400
        bass_node.synthesizer.amplitude_envelope = ADSREnvelope(
            attack=0.01, decay=0.1, sustain=0.8, release=0.2
        )
        self.musical_network.add_node(bass_node)
        
        # Add pad/atmosphere for ambient styles
        if self.style in [MusicalStyle.AMBIENT, MusicalStyle.ELECTRONIC]:
            pad_node = MusicalNode("pad", frequency=440.0)
            pad_node.synthesizer.add_oscillator(WaveShape.SINE, 0.3)
            pad_node.synthesizer.add_oscillator(WaveShape.TRIANGLE, 0.3)
            pad_node.synthesizer.add_oscillator(WaveShape.SAWTOOTH, 0.4)
            pad_node.synthesizer.filter.filter_type = FilterType.LOWPASS
            pad_node.synthesizer.filter.cutoff = 800
            pad_node.synthesizer.filter_envelope_amount = 0.7
            pad_node.synthesizer.amplitude_envelope = ADSREnvelope(
                attack=2.0, decay=1.0, sustain=0.7, release=3.0
            )
            pad_node.synthesizer.reverb_mix = 0.6
            pad_node.synthesizer.delay_mix = 0.3
            self.musical_network.add_node(pad_node)
            
        # Create connections for resonance
        # Melody influences harmony
        for i in range(4):
            self.musical_network.connect("melody", f"harmony_{i}", weight=0.3)
            
        # Bass influences all
        for node_id in self.musical_network.nodes:
            if node_id != "bass":
                self.musical_network.connect("bass", node_id, weight=0.2)
                
        # Harmony voices influence each other
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.musical_network.connect(
                        f"harmony_{i}", f"harmony_{j}", 
                        weight=0.1
                    )
                    
        return self.musical_network
        
    def render_composition(self, duration_seconds: float = 30.0,
                          sample_rate: int = 44100) -> np.ndarray:
        """
        Render the complete composition to audio.
        
        Args:
            duration_seconds: Total duration
            sample_rate: Sample rate
            
        Returns:
            Audio array
        """
        if not self.structure:
            self.create_structure()
            
        if not self.harmonic_progression:
            self.generate_harmonic_progression(
                HarmonicProgression.I_IV_V_I,
                self.structure.total_measures
            )
            
        if not self.melody_line:
            self.generate_melody(self.structure.total_measures)
            
        if not self.bass_line:
            self.generate_bass_line(self.harmonic_progression)
            
        # Build arrangement
        network = self.build_arrangement()
        
        # Create rhythm network
        rhythm_network = self.rhythm_engine.create_rhythm_network()
        
        # Generate rhythm events
        rhythm_events = self.rhythm_engine.generate_events(duration_seconds)
        
        # Render audio in chunks
        chunk_duration = self.rhythm_engine.measure_duration
        num_chunks = int(np.ceil(duration_seconds / chunk_duration))
        
        full_audio = np.zeros(int(duration_seconds * sample_rate))
        
        for chunk_idx in range(num_chunks):
            chunk_start_time = chunk_idx * chunk_duration
            chunk_end_time = min((chunk_idx + 1) * chunk_duration, duration_seconds)
            actual_chunk_duration = chunk_end_time - chunk_start_time
            
            # Update section if needed
            measure_idx = chunk_idx
            current_section = None
            measure_in_composition = 0
            
            for section in self.structure.sections:
                if measure_in_composition <= measure_idx < measure_in_composition + section.duration_measures:
                    current_section = section
                    break
                measure_in_composition += section.duration_measures
                
            # Apply section settings
            if current_section:
                # Dynamic level
                for node in network.nodes.values():
                    if isinstance(node, MusicalNode):
                        node.velocity = current_section.dynamic_level
                        
                # Tempo change
                if current_section.tempo_change:
                    self.rhythm_engine.tempo = current_section.tempo_change
                    
                # Emotional coloring
                if current_section.emotion:
                    # Apply emotional parameters to synthesis
                    emotion_sig = self.emotional_engine.emotion_signatures[current_section.emotion]
                    
                    for node in network.nodes.values():
                        if isinstance(node, MusicalNode):
                            # Modulate filter based on emotion
                            node.synthesizer.filter.cutoff = (
                                1000 + emotion_sig.energy_level * 2000
                            )
                            # Modulate reverb
                            node.synthesizer.reverb_mix = (
                                0.2 + (1 - emotion_sig.energy_level) * 0.4
                            )
                            
            # Set current harmony
            if chunk_idx < len(self.harmonic_progression):
                root_degree, chord_type = self.harmonic_progression[chunk_idx]
                
                # Get chord frequencies
                chord_frequencies = []
                chord_degrees = {'triad': [0, 2, 4], 'seventh': [0, 2, 4, 6]}
                
                for degree_offset in chord_degrees.get(chord_type, [0, 2, 4]):
                    degree = (root_degree + degree_offset) % len(self.musical_network.mode_intervals)
                    ratio = self.musical_network.mode_intervals[degree]
                    freq = self.musical_network.base_freq * ratio
                    chord_frequencies.append(freq)
                    
                # Apply to harmony nodes with voice leading
                harmony_nodes = [f"harmony_{i}" for i in range(min(4, len(chord_frequencies)))]
                
                if hasattr(self, 'previous_chord_frequencies'):
                    chord_frequencies = VoiceLeading.smooth_voice_leading(
                        self.previous_chord_frequencies,
                        chord_frequencies
                    )
                    
                for i, (node_id, freq) in enumerate(zip(harmony_nodes, chord_frequencies)):
                    if node_id in network.nodes:
                        network.nodes[node_id].frequency = freq
                        network.nodes[node_id].trigger(0.6)
                        
                self.previous_chord_frequencies = chord_frequencies
                
            # Set melody note
            melody_idx = chunk_idx * 4  # 4 notes per measure
            if melody_idx < len(self.melody_line):
                network.nodes["melody"].frequency = self.melody_line[melody_idx]
                network.nodes["melody"].trigger(0.8)
                
            # Set bass note
            bass_idx = chunk_idx * 2  # 2 notes per measure
            if bass_idx < len(self.bass_line):
                network.nodes["bass"].frequency = self.bass_line[bass_idx]
                network.nodes["bass"].trigger(0.9)
                
            # Apply rhythm events
            current_rhythm_events = [
                (t, v, vel) for t, v, vel in rhythm_events
                if chunk_start_time <= t < chunk_end_time
            ]
            
            # Render chunk
            chunk_samples = int(actual_chunk_duration * sample_rate)
            
            # Musical network audio
            musical_audio = network.generate_musical_signals(
                actual_chunk_duration, sample_rate
            )
            
            # Rhythm network audio
            rhythm_audio = np.zeros(chunk_samples)
            if self.style != MusicalStyle.AMBIENT:  # Skip drums for ambient
                for event_time, voice, velocity in current_rhythm_events:
                    relative_time = event_time - chunk_start_time
                    self.rhythm_engine.apply_events_to_network(
                        [(relative_time, voice, velocity)],
                        rhythm_network,
                        0.0
                    )
                    
                rhythm_audio = rhythm_network.generate_musical_signals(
                    actual_chunk_duration, sample_rate
                )
                
            # Mix musical and rhythm
            chunk_audio = musical_audio * 0.7 + rhythm_audio * 0.3
            
            # Apply to full audio
            start_sample = int(chunk_start_time * sample_rate)
            end_sample = start_sample + len(chunk_audio)
            
            if end_sample <= len(full_audio):
                full_audio[start_sample:end_sample] = chunk_audio
                
        # Final mastering
        # Normalize
        max_val = np.max(np.abs(full_audio))
        if max_val > 0:
            full_audio = full_audio / max_val * 0.8
            
        # Simple limiting/compression
        full_audio = np.tanh(full_audio * 0.7) / 0.7
        
        return full_audio 