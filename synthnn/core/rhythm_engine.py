"""
Rhythm Engine for SynthNN

Provides rhythmic capabilities including beat patterns, time signatures,
groove templates, and synchronization with musical networks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import random

from .resonant_node import ResonantNode
from .resonant_network import ResonantNetwork
from .musical_synthesis import MusicalNode, MusicalResonantNetwork, WaveShape, FilterType


class TimeSignature:
    """Represents a musical time signature."""
    
    def __init__(self, numerator: int = 4, denominator: int = 4):
        self.numerator = numerator  # Beats per measure
        self.denominator = denominator  # Note value of beat
        
    @property
    def beats_per_measure(self) -> int:
        return self.numerator
        
    @property
    def beat_duration(self) -> float:
        """Duration of one beat relative to whole note."""
        return 4.0 / self.denominator
        
    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


class RhythmPattern:
    """Represents a rhythmic pattern."""
    
    def __init__(self, name: str, pattern: List[float], 
                 velocities: Optional[List[float]] = None):
        """
        Initialize rhythm pattern.
        
        Args:
            name: Pattern name
            pattern: List of beat times (0-1 within a measure)
            velocities: Optional velocity for each beat
        """
        self.name = name
        self.pattern = pattern
        
        if velocities is None:
            self.velocities = [1.0] * len(pattern)
        else:
            self.velocities = velocities
            
    def get_events(self, measure_duration: float) -> List[Tuple[float, float]]:
        """
        Get pattern events.
        
        Returns:
            List of (time, velocity) tuples
        """
        events = []
        for beat_time, velocity in zip(self.pattern, self.velocities):
            actual_time = beat_time * measure_duration
            events.append((actual_time, velocity))
        return events


class GrooveTemplate:
    """Template for musical grooves with swing and humanization."""
    
    def __init__(self, name: str, base_pattern: RhythmPattern,
                 swing: float = 0.0, humanize: float = 0.0):
        """
        Initialize groove template.
        
        Args:
            name: Groove name
            base_pattern: Base rhythm pattern
            swing: Swing amount (0-1, where 0.67 is triplet swing)
            humanize: Random timing variation (0-1)
        """
        self.name = name
        self.base_pattern = base_pattern
        self.swing = swing
        self.humanize = humanize
        
    def apply_swing(self, events: List[Tuple[float, float]], 
                   measure_duration: float) -> List[Tuple[float, float]]:
        """Apply swing to events."""
        if self.swing == 0:
            return events
            
        swung_events = []
        beat_duration = measure_duration / 4  # Assume 4/4 for now
        
        for time, velocity in events:
            # Find which beat subdivision this falls on
            beat_pos = (time % beat_duration) / beat_duration
            
            # Apply swing to off-beats (roughly 0.5)
            if 0.4 < beat_pos < 0.6:
                # Delay off-beats based on swing amount
                swing_delay = beat_duration * 0.5 * self.swing
                new_time = time + swing_delay
            else:
                new_time = time
                
            swung_events.append((new_time, velocity))
            
        return swung_events
        
    def apply_humanization(self, events: List[Tuple[float, float]],
                          measure_duration: float) -> List[Tuple[float, float]]:
        """Add human-like timing variations."""
        if self.humanize == 0:
            return events
            
        humanized_events = []
        
        for time, velocity in events:
            # Random timing offset
            max_offset = 0.02 * measure_duration * self.humanize
            time_offset = random.uniform(-max_offset, max_offset)
            
            # Random velocity variation
            velocity_var = random.uniform(1 - 0.2 * self.humanize, 
                                        1 + 0.1 * self.humanize)
            
            new_time = max(0, time + time_offset)
            new_velocity = np.clip(velocity * velocity_var, 0, 1)
            
            humanized_events.append((new_time, new_velocity))
            
        return humanized_events
        
    def generate_events(self, measure_duration: float) -> List[Tuple[float, float]]:
        """Generate groove events with all modifications."""
        events = self.base_pattern.get_events(measure_duration)
        events = self.apply_swing(events, measure_duration)
        events = self.apply_humanization(events, measure_duration)
        return events


class DrumVoice(Enum):
    """Standard drum voices."""
    KICK = "kick"
    SNARE = "snare"
    HIHAT_CLOSED = "hihat_closed"
    HIHAT_OPEN = "hihat_open"
    RIDE = "ride"
    CRASH = "crash"
    TOM_HIGH = "tom_high"
    TOM_MID = "tom_mid"
    TOM_LOW = "tom_low"
    CLAP = "clap"
    RIMSHOT = "rimshot"
    COWBELL = "cowbell"


@dataclass
class DrumKit:
    """Collection of drum sounds mapped to voices."""
    voice_frequencies: Dict[DrumVoice, float] = field(default_factory=dict)
    voice_envelopes: Dict[DrumVoice, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default drum kit."""
        if not self.voice_frequencies:
            self.voice_frequencies = {
                DrumVoice.KICK: 60.0,
                DrumVoice.SNARE: 200.0,
                DrumVoice.HIHAT_CLOSED: 8000.0,
                DrumVoice.HIHAT_OPEN: 6000.0,
                DrumVoice.RIDE: 5000.0,
                DrumVoice.CRASH: 4000.0,
                DrumVoice.TOM_HIGH: 400.0,
                DrumVoice.TOM_MID: 250.0,
                DrumVoice.TOM_LOW: 150.0,
                DrumVoice.CLAP: 1500.0,
                DrumVoice.RIMSHOT: 500.0,
                DrumVoice.COWBELL: 800.0
            }
            
        if not self.voice_envelopes:
            # Default ADSR for each voice
            self.voice_envelopes = {
                DrumVoice.KICK: {'attack': 0.001, 'decay': 0.1, 'sustain': 0.3, 'release': 0.2},
                DrumVoice.SNARE: {'attack': 0.001, 'decay': 0.05, 'sustain': 0.1, 'release': 0.1},
                DrumVoice.HIHAT_CLOSED: {'attack': 0.001, 'decay': 0.02, 'sustain': 0.0, 'release': 0.02},
                DrumVoice.HIHAT_OPEN: {'attack': 0.001, 'decay': 0.1, 'sustain': 0.2, 'release': 0.3},
                # ... etc for other voices
            }


class RhythmEngine:
    """
    Main rhythm engine for beat generation and synchronization.
    """
    
    def __init__(self, tempo: float = 120.0, 
                 time_signature: TimeSignature = None):
        """
        Initialize rhythm engine.
        
        Args:
            tempo: Beats per minute
            time_signature: Time signature (default 4/4)
        """
        self.tempo = tempo
        self.time_signature = time_signature or TimeSignature(4, 4)
        
        # Timing
        self.current_time = 0.0
        self.current_beat = 0
        self.current_measure = 0
        
        # Pattern storage
        self.patterns: Dict[str, RhythmPattern] = {}
        self.grooves: Dict[str, GrooveTemplate] = {}
        self.active_patterns: Dict[str, bool] = {}
        
        # Drum kit
        self.drum_kit = DrumKit()
        
        # Event queue
        self.event_queue: List[Tuple[float, str, float]] = []  # (time, voice, velocity)
        
        # Initialize built-in patterns
        self._init_builtin_patterns()
        
    def _init_builtin_patterns(self):
        """Initialize built-in rhythm patterns."""
        # Basic patterns
        self.patterns['four_on_floor'] = RhythmPattern(
            'four_on_floor',
            [0.0, 0.25, 0.5, 0.75],
            [1.0, 0.8, 0.9, 0.8]
        )
        
        self.patterns['basic_rock'] = RhythmPattern(
            'basic_rock',
            [0.0, 0.5],  # Kick on 1 and 3
            [1.0, 0.9]
        )
        
        self.patterns['snare_backbeat'] = RhythmPattern(
            'snare_backbeat',
            [0.25, 0.75],  # Snare on 2 and 4
            [0.9, 1.0]
        )
        
        self.patterns['hihat_8ths'] = RhythmPattern(
            'hihat_8ths',
            [i * 0.125 for i in range(8)],
            [0.7 if i % 2 == 0 else 0.5 for i in range(8)]
        )
        
        self.patterns['hihat_16ths'] = RhythmPattern(
            'hihat_16ths',
            [i * 0.0625 for i in range(16)],
            [0.6 if i % 4 == 0 else 0.3 for i in range(16)]
        )
        
        # Complex patterns
        self.patterns['funk_kick'] = RhythmPattern(
            'funk_kick',
            [0.0, 0.1875, 0.5, 0.5625, 0.75],
            [1.0, 0.7, 0.8, 0.6, 0.9]
        )
        
        self.patterns['jazz_ride'] = RhythmPattern(
            'jazz_ride',
            [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875],
            [0.8, 0.4, 0.7, 0.3, 0.8, 0.4, 0.7, 0.3]
        )
        
        # Create grooves
        self.grooves['straight'] = GrooveTemplate(
            'straight',
            self.patterns['four_on_floor'],
            swing=0.0,
            humanize=0.1
        )
        
        self.grooves['swing'] = GrooveTemplate(
            'swing',
            self.patterns['jazz_ride'],
            swing=0.67,  # Triplet swing
            humanize=0.2
        )
        
        self.grooves['shuffle'] = GrooveTemplate(
            'shuffle',
            self.patterns['hihat_8ths'],
            swing=0.5,
            humanize=0.15
        )
        
    @property
    def beat_duration(self) -> float:
        """Duration of one beat in seconds."""
        return 60.0 / self.tempo
        
    @property
    def measure_duration(self) -> float:
        """Duration of one measure in seconds."""
        return self.beat_duration * self.time_signature.beats_per_measure
        
    def add_pattern(self, name: str, pattern: RhythmPattern):
        """Add a rhythm pattern."""
        self.patterns[name] = pattern
        
    def add_groove(self, name: str, groove: GrooveTemplate):
        """Add a groove template."""
        self.grooves[name] = groove
        
    def activate_pattern(self, pattern_name: str, voice: DrumVoice):
        """Activate a pattern for a specific drum voice."""
        key = f"{pattern_name}_{voice.value}"
        self.active_patterns[key] = True
        
    def deactivate_pattern(self, pattern_name: str, voice: DrumVoice):
        """Deactivate a pattern."""
        key = f"{pattern_name}_{voice.value}"
        self.active_patterns[key] = False
        
    def update(self, dt: float):
        """
        Update rhythm engine state.
        
        Args:
            dt: Time step in seconds
        """
        self.current_time += dt
        
        # Update beat and measure
        total_beats = self.current_time / self.beat_duration
        self.current_beat = int(total_beats) % self.time_signature.beats_per_measure
        self.current_measure = int(total_beats / self.time_signature.beats_per_measure)
        
    def get_current_position(self) -> Tuple[int, float]:
        """
        Get current position in the rhythm.
        
        Returns:
            (measure, beat_fraction) where beat_fraction is 0-1
        """
        beat_in_measure = (self.current_time / self.beat_duration) % self.time_signature.beats_per_measure
        return (self.current_measure, beat_in_measure / self.time_signature.beats_per_measure)
        
    def generate_events(self, duration: float) -> List[Tuple[float, DrumVoice, float]]:
        """
        Generate rhythm events for a duration.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            List of (time, voice, velocity) events
        """
        events = []
        
        # Calculate how many measures we need
        num_measures = int(np.ceil(duration / self.measure_duration))
        
        # Generate events for each active pattern
        for pattern_key, is_active in self.active_patterns.items():
            if not is_active:
                continue
                
            # Parse pattern name and voice
            parts = pattern_key.rsplit('_', 1)
            if len(parts) != 2:
                continue
                
            pattern_name, voice_str = parts
            
            # Get pattern and voice
            if pattern_name not in self.patterns:
                continue
                
            try:
                voice = DrumVoice(voice_str)
            except ValueError:
                continue
                
            pattern = self.patterns[pattern_name]
            
            # Generate events for each measure
            for measure in range(num_measures):
                measure_start = measure * self.measure_duration
                
                # Get pattern events
                pattern_events = pattern.get_events(self.measure_duration)
                
                # Add to event list
                for event_time, velocity in pattern_events:
                    absolute_time = measure_start + event_time
                    if absolute_time < duration:
                        events.append((absolute_time, voice, velocity))
                        
        # Sort by time
        events.sort(key=lambda x: x[0])
        
        return events
        
    def sync_with_network(self, network: MusicalResonantNetwork):
        """
        Synchronize rhythm engine with a musical network.
        
        Args:
            network: Musical network to sync with
        """
        # Match tempo
        if hasattr(network, 'tempo'):
            self.tempo = network.tempo
            
        # Match time signature
        if hasattr(network, 'time_signature'):
            self.time_signature = TimeSignature(*network.time_signature)
            
    def create_rhythm_network(self) -> MusicalResonantNetwork:
        """
        Create a musical network for rhythm generation.
        
        Returns:
            Network configured for rhythm
        """
        network = MusicalResonantNetwork(
            name="rhythm_network",
            mode="chromatic",  # Use all frequencies
            base_freq=100.0
        )
        
        # Add nodes for each drum voice
        for voice, freq in self.drum_kit.voice_frequencies.items():
            node = MusicalNode(
                node_id=f"drum_{voice.value}",
                frequency=freq
            )
            
            # Configure synthesizer for drum sounds
            if voice == DrumVoice.KICK:
                node.synthesizer.add_oscillator(WaveShape.SINE, 1.0)
                node.synthesizer.add_oscillator(WaveShape.NOISE, 0.1)
                node.synthesizer.filter.filter_type = FilterType.LOWPASS
                node.synthesizer.filter.cutoff = 150
                
            elif voice == DrumVoice.SNARE:
                node.synthesizer.add_oscillator(WaveShape.TRIANGLE, 0.5)
                node.synthesizer.add_oscillator(WaveShape.NOISE, 0.5)
                node.synthesizer.filter.filter_type = FilterType.HIGHPASS
                node.synthesizer.filter.cutoff = 200
                
            elif voice in [DrumVoice.HIHAT_CLOSED, DrumVoice.HIHAT_OPEN]:
                node.synthesizer.add_oscillator(WaveShape.NOISE, 1.0)
                node.synthesizer.filter.filter_type = FilterType.HIGHPASS
                node.synthesizer.filter.cutoff = 5000
                
            # Set envelope if available
            if voice in self.drum_kit.voice_envelopes:
                env_params = self.drum_kit.voice_envelopes[voice]
                node.synthesizer.amplitude_envelope.attack = env_params.get('attack', 0.001)
                node.synthesizer.amplitude_envelope.decay = env_params.get('decay', 0.1)
                node.synthesizer.amplitude_envelope.sustain = env_params.get('sustain', 0.0)
                node.synthesizer.amplitude_envelope.release = env_params.get('release', 0.1)
                
            network.add_node(node)
            
        return network
        
    def apply_events_to_network(self, events: List[Tuple[float, DrumVoice, float]],
                               network: MusicalResonantNetwork,
                               current_time: float = 0.0):
        """
        Apply rhythm events to a network.
        
        Args:
            events: List of (time, voice, velocity) events
            network: Network to apply events to
            current_time: Current time in the sequence
        """
        # Process events that should trigger now
        for event_time, voice, velocity in events:
            if abs(event_time - current_time) < 0.01:  # Within 10ms
                node_id = f"drum_{voice.value}"
                
                if node_id in network.nodes:
                    node = network.nodes[node_id]
                    if isinstance(node, MusicalNode):
                        node.trigger(velocity)
                        
    def generate_polyrhythm(self, pattern1: str, pattern2: str,
                          ratio: Tuple[int, int] = (3, 2)) -> RhythmPattern:
        """
        Generate a polyrhythm from two patterns.
        
        Args:
            pattern1: First pattern name
            pattern2: Second pattern name  
            ratio: Polyrhythmic ratio (e.g., 3:2)
            
        Returns:
            Combined polyrhythmic pattern
        """
        if pattern1 not in self.patterns or pattern2 not in self.patterns:
            raise ValueError("Pattern not found")
            
        p1 = self.patterns[pattern1]
        p2 = self.patterns[pattern2]
        
        # Scale patterns to fit ratio
        events = []
        
        # Pattern 1 events
        for i in range(ratio[0]):
            base_time = i / ratio[0]
            for beat_time, vel in zip(p1.pattern, p1.velocities):
                scaled_time = base_time + beat_time / ratio[0]
                if scaled_time < 1.0:
                    events.append((scaled_time, vel))
                    
        # Pattern 2 events  
        for i in range(ratio[1]):
            base_time = i / ratio[1]
            for beat_time, vel in zip(p2.pattern, p2.velocities):
                scaled_time = base_time + beat_time / ratio[1]
                if scaled_time < 1.0:
                    events.append((scaled_time, vel * 0.8))  # Slightly quieter
                    
        # Sort and extract
        events.sort(key=lambda x: x[0])
        pattern_times = [e[0] for e in events]
        pattern_vels = [e[1] for e in events]
        
        return RhythmPattern(
            f"poly_{pattern1}_{pattern2}_{ratio[0]}_{ratio[1]}",
            pattern_times,
            pattern_vels
        ) 