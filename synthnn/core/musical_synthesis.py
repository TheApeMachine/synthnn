"""
Advanced Musical Synthesis Engine for SynthNN

Provides rich synthesis capabilities including wavetable, FM, additive synthesis,
ADSR envelopes, filters, and effects for creating complex, musical sounds.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from enum import Enum
import scipy.signal as signal
from scipy import interpolate

from .resonant_node import ResonantNode
from .resonant_network import ResonantNetwork
from .musical_constants import MODE_INTERVALS


class WaveShape(Enum):
    """Available wave shapes for synthesis."""
    SINE = "sine"
    SAWTOOTH = "sawtooth"
    SQUARE = "square"
    TRIANGLE = "triangle"
    PULSE = "pulse"
    NOISE = "noise"
    CUSTOM = "custom"


class FilterType(Enum):
    """Filter types for sound shaping."""
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    NOTCH = "notch"
    ALLPASS = "allpass"


@dataclass
class ADSREnvelope:
    """ADSR Envelope for amplitude and filter modulation."""
    attack: float = 0.01    # Time to reach peak (seconds)
    decay: float = 0.1      # Time to reach sustain level
    sustain: float = 0.7    # Sustain level (0-1)
    release: float = 0.3    # Time to fade out after note off
    
    def generate(self, duration: float, sample_rate: int, 
                 note_off_time: Optional[float] = None) -> np.ndarray:
        """
        Generate ADSR envelope.
        
        Args:
            duration: Total duration in seconds
            sample_rate: Sample rate
            note_off_time: When note is released (None = held for full duration)
            
        Returns:
            Envelope array
        """
        num_samples = int(duration * sample_rate)
        envelope = np.zeros(num_samples)
        
        # Sample counts for each stage
        attack_samples = int(self.attack * sample_rate)
        decay_samples = int(self.decay * sample_rate)
        
        # Note off time
        if note_off_time is None:
            note_off_sample = num_samples
        else:
            note_off_sample = int(note_off_time * sample_rate)
            
        release_samples = int(self.release * sample_rate)
        
        # Generate envelope stages
        for i in range(num_samples):
            if i < attack_samples:
                # Attack stage
                envelope[i] = i / attack_samples
            elif i < attack_samples + decay_samples:
                # Decay stage
                decay_progress = (i - attack_samples) / decay_samples
                envelope[i] = 1.0 - decay_progress * (1.0 - self.sustain)
            elif i < note_off_sample:
                # Sustain stage
                envelope[i] = self.sustain
            elif i < note_off_sample + release_samples:
                # Release stage
                release_progress = (i - note_off_sample) / release_samples
                start_level = self.sustain if note_off_sample > attack_samples + decay_samples else envelope[note_off_sample - 1]
                envelope[i] = start_level * (1.0 - release_progress)
            else:
                # After release
                envelope[i] = 0.0
                
        return envelope


class Oscillator:
    """
    Advanced oscillator with multiple wave shapes and modulation.
    """
    
    def __init__(self, wave_shape: WaveShape = WaveShape.SINE):
        self.wave_shape = wave_shape
        self.phase = 0.0
        self.phase_modulation = 0.0
        self.frequency_modulation = 0.0
        self.pulse_width = 0.5  # For pulse wave
        
        # Wavetable for custom waveforms
        self.wavetable: Optional[np.ndarray] = None
        self.wavetable_size = 2048
        
    def generate(self, frequency: float, duration: float, 
                 sample_rate: int) -> np.ndarray:
        """Generate oscillator output."""
        num_samples = int(duration * sample_rate)
        output = np.zeros(num_samples)
        
        # Phase increment per sample
        phase_increment = 2 * np.pi * frequency / sample_rate
        
        for i in range(num_samples):
            # Apply frequency modulation
            current_freq = frequency * (1 + self.frequency_modulation)
            phase_increment = 2 * np.pi * current_freq / sample_rate
            
            # Update phase with modulation
            self.phase += phase_increment
            current_phase = self.phase + self.phase_modulation
            
            # Wrap phase
            while current_phase >= 2 * np.pi:
                current_phase -= 2 * np.pi
                
            # Generate sample based on wave shape
            if self.wave_shape == WaveShape.SINE:
                output[i] = np.sin(current_phase)
                
            elif self.wave_shape == WaveShape.SAWTOOTH:
                output[i] = 2 * (current_phase / (2 * np.pi)) - 1
                
            elif self.wave_shape == WaveShape.SQUARE:
                output[i] = 1.0 if current_phase < np.pi else -1.0
                
            elif self.wave_shape == WaveShape.TRIANGLE:
                if current_phase < np.pi:
                    output[i] = -1 + (2 * current_phase / np.pi)
                else:
                    output[i] = 3 - (2 * current_phase / np.pi)
                    
            elif self.wave_shape == WaveShape.PULSE:
                threshold = 2 * np.pi * self.pulse_width
                output[i] = 1.0 if current_phase < threshold else -1.0
                
            elif self.wave_shape == WaveShape.NOISE:
                output[i] = np.random.uniform(-1, 1)
                
            elif self.wave_shape == WaveShape.CUSTOM and self.wavetable is not None:
                # Interpolate from wavetable
                table_index = (current_phase / (2 * np.pi)) * self.wavetable_size
                output[i] = np.interp(table_index, 
                                    range(self.wavetable_size), 
                                    self.wavetable)
                
        return output
        
    def set_wavetable(self, waveform: np.ndarray):
        """Set custom wavetable."""
        self.wavetable = np.array(waveform)
        self.wavetable_size = len(waveform)
        self.wave_shape = WaveShape.CUSTOM


class Filter:
    """
    Multi-mode filter with resonance.
    """
    
    def __init__(self, filter_type: FilterType = FilterType.LOWPASS,
                 cutoff: float = 1000.0, resonance: float = 1.0):
        self.filter_type = filter_type
        self.cutoff = cutoff
        self.resonance = max(1.0, resonance)  # Q factor
        
    def process(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply filter to signal."""
        if len(signal) == 0:
            return signal
            
        # Normalize cutoff frequency
        nyquist = sample_rate / 2
        normalized_cutoff = self.cutoff / nyquist
        
        # Clamp to valid range
        normalized_cutoff = np.clip(normalized_cutoff, 0.001, 0.999)
        
        # Import scipy.signal for filter design
        from scipy import signal as scipy_signal
        
        # Design filter
        if self.filter_type == FilterType.LOWPASS:
            b, a = scipy_signal.butter(4, normalized_cutoff, btype='low')
        elif self.filter_type == FilterType.HIGHPASS:
            b, a = scipy_signal.butter(4, normalized_cutoff, btype='high')
        elif self.filter_type == FilterType.BANDPASS:
            # For bandpass, use cutoff as center frequency
            bandwidth = normalized_cutoff / self.resonance
            low = max(0.001, normalized_cutoff - bandwidth/2)
            high = min(0.999, normalized_cutoff + bandwidth/2)
            b, a = scipy_signal.butter(4, [low, high], btype='band')
        elif self.filter_type == FilterType.NOTCH:
            # Notch filter (band-stop)
            bandwidth = normalized_cutoff / self.resonance
            low = max(0.001, normalized_cutoff - bandwidth/2)
            high = min(0.999, normalized_cutoff + bandwidth/2)
            b, a = scipy_signal.butter(4, [low, high], btype='bandstop')
        else:  # ALLPASS
            # Simple allpass approximation
            return signal
            
        # Apply filter
        return scipy_signal.filtfilt(b, a, signal)


class MusicalSynthesizer:
    """
    Complete musical synthesizer for a resonant node.
    """
    
    def __init__(self):
        # Oscillators
        self.oscillators: List[Oscillator] = []
        self.oscillator_mix: List[float] = []
        
        # Envelopes
        self.amplitude_envelope = ADSREnvelope()
        self.filter_envelope = ADSREnvelope(attack=0.05, decay=0.2, 
                                          sustain=0.3, release=0.5)
        
        # Filter
        self.filter = Filter()
        self.filter_envelope_amount = 0.5  # How much envelope affects filter
        
        # Effects
        self.reverb_mix = 0.0
        self.delay_mix = 0.0
        self.delay_time = 0.25  # seconds
        self.delay_feedback = 0.4
        
        # Modulation
        self.vibrato_rate = 5.0  # Hz
        self.vibrato_depth = 0.02  # Pitch modulation depth
        self.tremolo_rate = 4.0  # Hz
        self.tremolo_depth = 0.1  # Amplitude modulation depth
        
    def add_oscillator(self, wave_shape: WaveShape, mix: float = 1.0):
        """Add an oscillator to the synth."""
        osc = Oscillator(wave_shape)
        self.oscillators.append(osc)
        self.oscillator_mix.append(mix)
        
    def synthesize(self, frequency: float, duration: float, 
                   sample_rate: int = 44100,
                   note_off_time: Optional[float] = None) -> np.ndarray:
        """
        Synthesize a note with all processing.
        
        Args:
            frequency: Note frequency in Hz
            duration: Total duration in seconds
            sample_rate: Sample rate
            note_off_time: When to release the note
            
        Returns:
            Synthesized audio
        """
        num_samples = int(duration * sample_rate)
        output = np.zeros(num_samples)
        
        # If no oscillators, add a default sine
        if not self.oscillators:
            self.add_oscillator(WaveShape.SINE)
            
        # Generate oscillator outputs
        for osc, mix in zip(self.oscillators, self.oscillator_mix):
            # Add vibrato
            if self.vibrato_depth > 0:
                vibrato = np.sin(2 * np.pi * self.vibrato_rate * 
                               np.arange(num_samples) / sample_rate)
                osc.frequency_modulation = self.vibrato_depth * vibrato[0]
                
            # Generate oscillator signal
            osc_output = osc.generate(frequency, duration, sample_rate)
            output += osc_output * mix
            
        # Normalize oscillator mix
        if len(self.oscillator_mix) > 0:
            output /= sum(self.oscillator_mix)
            
        # Apply amplitude envelope
        amp_env = self.amplitude_envelope.generate(duration, sample_rate, note_off_time)
        output *= amp_env
        
        # Apply filter with envelope
        if self.filter.cutoff < sample_rate / 2:
            # Generate filter envelope
            filt_env = self.filter_envelope.generate(duration, sample_rate, note_off_time)
            
            # Modulate filter cutoff with envelope
            base_cutoff = self.filter.cutoff
            for i in range(num_samples):
                # Update filter cutoff based on envelope
                env_mod = filt_env[i] * self.filter_envelope_amount
                self.filter.cutoff = base_cutoff * (1 + env_mod * 4)  # Up to 5x
                
            # Apply filter
            output = self.filter.process(output, sample_rate)
            
            # Restore base cutoff
            self.filter.cutoff = base_cutoff
            
        # Apply tremolo
        if self.tremolo_depth > 0:
            tremolo = 1 + self.tremolo_depth * np.sin(
                2 * np.pi * self.tremolo_rate * np.arange(num_samples) / sample_rate
            )
            output *= tremolo
            
        # Apply effects
        if self.delay_mix > 0:
            output = self._apply_delay(output, sample_rate)
            
        if self.reverb_mix > 0:
            output = self._apply_reverb(output, sample_rate)
            
        return output
        
    def _apply_delay(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply delay effect."""
        delay_samples = int(self.delay_time * sample_rate)
        delayed = np.zeros_like(signal)
        
        # Simple delay with feedback
        for i in range(len(signal)):
            delayed[i] = signal[i]
            if i >= delay_samples:
                delayed[i] += signal[i - delay_samples] * self.delay_feedback
                
        # Mix with original
        return signal * (1 - self.delay_mix) + delayed * self.delay_mix
        
    def _apply_reverb(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply simple reverb effect."""
        # Simple reverb using multiple delays
        reverb = np.zeros_like(signal)
        
        # Early reflections
        delays = [0.013, 0.027, 0.037, 0.055, 0.071, 0.089]
        gains = [0.8, 0.6, 0.5, 0.4, 0.3, 0.2]
        
        for delay, gain in zip(delays, gains):
            delay_samples = int(delay * sample_rate)
            for i in range(delay_samples, len(signal)):
                reverb[i] += signal[i - delay_samples] * gain
                
        # Mix with original
        return signal * (1 - self.reverb_mix) + reverb * self.reverb_mix


class MusicalNode(ResonantNode):
    """
    Extended ResonantNode with musical synthesis capabilities.
    """
    
    def __init__(self, node_id: str, frequency: float = 440.0,
                 phase: float = 0.0, amplitude: float = 1.0,
                 damping: float = 0.1):
        super().__init__(node_id, frequency, phase, amplitude, damping)
        
        # Create synthesizer
        self.synthesizer = MusicalSynthesizer()
        
        # Musical properties
        self.note_on = False
        self.note_start_time = 0.0
        self.velocity = 1.0  # MIDI-style velocity
        
    def trigger(self, velocity: float = 1.0):
        """Trigger a note."""
        self.note_on = True
        self.note_start_time = 0.0
        self.velocity = velocity
        self.amplitude = velocity  # Link amplitude to velocity
        
    def release(self):
        """Release the note."""
        self.note_on = False
        
    def generate_audio(self, duration: float, sample_rate: int = 44100) -> np.ndarray:
        """Generate audio using the musical synthesizer."""
        # Determine note off time based on node state
        if self.note_on:
            note_off_time = None  # Note is held
        else:
            # Use a reasonable release time
            note_off_time = duration * 0.7
            
        # Generate audio with synthesizer
        audio = self.synthesizer.synthesize(
            self.frequency,
            duration,
            sample_rate,
            note_off_time
        )
        
        # Apply node amplitude and velocity
        audio *= self.amplitude * self.velocity
        
        return audio


class MusicalResonantNetwork(ResonantNetwork):
    """
    Musical extension of ResonantNetwork with synthesis capabilities.
    """
    
    def __init__(self, name: str = "musical_network",
                 mode: str = "ionian",
                 base_freq: float = 440.0,
                 mode_detector: Optional[Any] = None):
        super().__init__(name)

        self.mode = mode
        self.base_freq = base_freq
        self.mode_detector = mode_detector
        
        # Musical timing
        self.tempo = 120.0  # BPM
        self.time_signature = (4, 4)
        self.current_beat = 0.0
        
        # Chord progression
        self.chord_progression = []
        self.current_chord_index = 0
        
        # Initialize mode intervals
        self.mode_intervals = self._get_mode_intervals(mode)
        
    def _get_mode_intervals(self, mode: str) -> List[float]:
        """Return frequency ratios for a musical mode."""
        return MODE_INTERVALS.get(mode.lower(), MODE_INTERVALS['ionian'])
        
    def add_musical_node(self, degree: int, octave: int = 0) -> MusicalNode:
        """Add a musical node at a specific scale degree."""
        # Calculate frequency based on mode and degree
        interval_index = degree % len(self.mode_intervals)
        octave_multiplier = 2 ** octave
        
        frequency = self.base_freq * self.mode_intervals[interval_index] * octave_multiplier
        
        node = MusicalNode(
            node_id=f"note_{degree}_oct_{octave}",
            frequency=frequency
        )
        
        self.add_node(node)
        return node
        
    def create_scale(self, num_octaves: int = 2):
        """Create nodes for a musical scale."""
        degrees_per_octave = len(self.mode_intervals) - 1  # Exclude octave
        
        for octave in range(num_octaves):
            for degree in range(degrees_per_octave):
                self.add_musical_node(degree, octave)
                
    def play_chord(self, root_degree: int, chord_type: str = "triad"):
        """Play a chord starting from a root degree."""
        chord_intervals = {
            'triad': [0, 2, 4],
            'seventh': [0, 2, 4, 6],
            'ninth': [0, 2, 4, 6, 8],
            'sus2': [0, 1, 4],
            'sus4': [0, 3, 4],
            'dim': [0, 2, 4],  # Will be modified by mode
            'aug': [0, 2, 4]   # Will be modified
        }
        
        intervals = chord_intervals.get(chord_type, chord_intervals['triad'])
        
        # Trigger notes in chord
        for interval in intervals:
            degree = (root_degree + interval) % (len(self.mode_intervals) - 1)
            node_id = f"note_{degree}_oct_0"
            
            if node_id in self.nodes and isinstance(self.nodes[node_id], MusicalNode):
                self.nodes[node_id].trigger()
                
    def generate_musical_signals(self, duration: float, 
                               sample_rate: int = 44100) -> np.ndarray:
        """Generate musical audio from all nodes."""
        num_samples = int(duration * sample_rate)
        output = np.zeros(num_samples)
        
        # Generate audio from each musical node
        for node in self.nodes.values():
            if isinstance(node, MusicalNode):
                node_audio = node.generate_audio(duration, sample_rate)
                
                # Apply any network-level processing
                node_audio *= np.exp(-node.damping * np.linspace(0, duration, num_samples))
                
                output += node_audio
                
        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.8
            
        return output
        
    def create_chord_progression(self, progression: List[Tuple[int, str]]):
        """
        Create a chord progression.
        
        Args:
            progression: List of (root_degree, chord_type) tuples
        """
        self.chord_progression = progression
        self.current_chord_index = 0
        
    def advance_progression(self):
        """Move to the next chord in the progression."""
        if self.chord_progression:
            # Release current chord
            for node in self.nodes.values():
                if isinstance(node, MusicalNode):
                    node.release()
                    
            # Move to next chord
            self.current_chord_index = (self.current_chord_index + 1) % len(self.chord_progression)
            
            # Play new chord
            root, chord_type = self.chord_progression[self.current_chord_index]
            self.play_chord(root, chord_type) 