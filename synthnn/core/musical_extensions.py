"""
Musical extensions for the SynthNN core framework.

This module bridges the gap between the core ResonantNetwork and the music-specific
features from the original implementation, enabling mode-aware musical networks
with GPU acceleration support.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict

from .resonant_network import ResonantNetwork
from .resonant_node import ResonantNode
from .signal_processor import SignalProcessor
from .musical_constants import ROMAN_CHORD_MAP


class MusicalResonantNetwork(ResonantNetwork):
    """
    Extended ResonantNetwork with music-specific capabilities.
    
    This class combines the sophisticated infrastructure of synthnn.core
    with the musical intelligence from the original abstract.py implementation.
    """
    
    def __init__(self, name: str = "musical_network", base_freq: float = 440.0,
                 mode: str = "Ionian", mode_detector: Optional[Any] = None):
        """
        Initialize a musical resonant network.
        
        Args:
            name: Network name
            base_freq: Base frequency (A4 = 440Hz by default)
            mode: Musical mode name
            mode_detector: Optional ModeDetector instance for analysis
        """
        super().__init__(name)
        
        self.base_freq = base_freq
        self.mode = mode
        self.mode_detector = mode_detector
        
        # Musical state tracking (from abstract.py)
        self.harmonic_outputs = []
        self.dissonant_outputs = []
        self.retuned_outputs = []
        
        # Additional musical parameters
        self.tuning_system = "equal_temperament"  # or "just_intonation"
        self.pitch_bend_range = 2.0  # semitones
        
        # Signal processor for analysis
        self.signal_processor = SignalProcessor()
        
    def create_harmonic_nodes(self, harmonic_ratios: List[float], 
                            amplitude: float = 1.0, phase: float = 0.0) -> None:
        """
        Create nodes based on harmonic ratios relative to base frequency.
        
        Args:
            harmonic_ratios: List of frequency ratios (e.g., [1, 1.5, 2])
            amplitude: Initial amplitude for all nodes
            phase: Initial phase for all nodes
        """
        # Clear existing nodes
        self.nodes.clear()
        self.connections.clear()
        
        # Create nodes with harmonic frequencies
        for i, ratio in enumerate(harmonic_ratios):
            freq = self.base_freq * ratio
            node = ResonantNode(
                node_id=f"harmonic_{i}",
                frequency=freq,
                phase=phase,
                amplitude=amplitude / np.sqrt(len(harmonic_ratios))
            )
            self.add_node(node)
            
    def compute_harmonic_state(self, duration: float, sample_rate: float = 44100) -> np.ndarray:
        """
        Compute the pure harmonic output by stepping through the simulation.
        
        Returns:
            An array representing the summed, real-valued network output over time.
        """
        num_samples = int(duration * sample_rate)
        dt = 1.0 / sample_rate
        
        output_signal = np.zeros(num_samples)
        
        for i in range(num_samples):
            # Advance the network by one time step
            self.step(dt)
            
            # Get the complex signals from all nodes
            signals = self.get_signals()
            
            # Sum the signals and store the real part as the audio output
            total_signal = sum(signals.values())
            output_signal[i] = total_signal.real
            
        self.harmonic_outputs = output_signal.tolist()
        return output_signal
    
    def compute_dissonant_state(self, duration: float, foreign_signal: np.ndarray,
                               sample_rate: float = 44100) -> np.ndarray:
        """
        Compute network output with a foreign signal driving the network.
        """
        num_samples = int(duration * sample_rate)
        dt = 1.0 / sample_rate
        
        output_signal = np.zeros(num_samples)

        # Ensure foreign_signal matches our time steps
        if len(foreign_signal) != num_samples:
            foreign_signal = self.signal_processor.resample(
                foreign_signal, len(foreign_signal), num_samples
            )
        
        for i in range(num_samples):
            # The foreign signal acts as an external input to all nodes
            # We can model this as a complex input, but a real-valued one is simpler
            external_inputs = {
                node_id: complex(foreign_signal[i], 0) for node_id in self.nodes
            }
            
            self.step(dt, external_inputs=external_inputs)
            
            signals = self.get_signals()
            total_signal = sum(signals.values())
            output_signal[i] = total_signal.real
            
        self.dissonant_outputs = output_signal.tolist()
        return output_signal
    
    def analyze_and_retune(self, foreign_signal: np.ndarray, 
                          sample_rate: float = 44100) -> str:
        """
        Analyze foreign signal and retune network to harmonize.
        
        Args:
            foreign_signal: External signal to analyze
            sample_rate: Sample rate of the signal
            
        Returns:
            Detected mode name
        """
        if self.mode_detector is None:
            raise ValueError("Mode detector required for analysis")
            
        # Set signal processor sample rate
        self.signal_processor.sample_rate = sample_rate
        
        # Extract features from foreign signal
        features = self.signal_processor.analyze_spectrum(foreign_signal)
        fundamental = self.signal_processor.extract_fundamental(foreign_signal, freq_range=(50, 2000))
        
        # Detect mode using mode detector
        detected_mode = self.mode_detector.analyze(foreign_signal, sample_rate)
        self.mode = detected_mode
        
        # Get mode intervals
        mode_intervals = self.mode_detector.mode_intervals.get(detected_mode, [1])
        
        # Retune nodes based on detected fundamental and mode
        self._retune_to_mode(fundamental, mode_intervals)
        
        return detected_mode
    
    def _retune_to_mode(self, new_base_freq: float, mode_intervals: Union[str, list[float]]) -> None:
        """
        Retune all nodes to new base frequency with mode intervals.
        """
        self.base_freq = new_base_freq

        if isinstance(mode_intervals, str):
            mode_intervals = ROMAN_CHORD_MAP.get(mode_intervals.upper(), [1])
        # Ensure we have the right number of intervals
        num_nodes = len(self.nodes)
        if len(mode_intervals) < num_nodes:
            mode_intervals = mode_intervals + [mode_intervals[-1]] * (num_nodes - len(mode_intervals))
        
        # Retune each node
        for i, (node_id, node) in enumerate(self.nodes.items()):
            if i < len(mode_intervals):
                new_freq = new_base_freq * mode_intervals[i]
                node.retune(new_freq)
                
    def apply_pitch_bend(self, node_id: str, bend_amount: float) -> None:
        """
        Apply pitch bend to a specific node.
        
        Args:
            node_id: ID of the node to bend
            bend_amount: Pitch bend in semitones (-pitch_bend_range to +pitch_bend_range)
        """
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        bend_amount = np.clip(bend_amount, -self.pitch_bend_range, self.pitch_bend_range)
        
        # Convert semitones to frequency ratio
        ratio = 2 ** (bend_amount / 12.0)
        # Use the frequency property for backward compatibility
        new_freq = node.frequency * ratio
        node.retune(new_freq)
        
    def create_modal_connections(self, connection_pattern: str = "nearest_neighbor",
                                weight_scale: float = 0.5) -> None:
        """
        Create connections between nodes based on musical relationships.
        
        Args:
            connection_pattern: Type of connection pattern
            weight_scale: Base weight for connections
        """
        node_ids = list(self.nodes.keys())
        
        if connection_pattern == "nearest_neighbor":
            # Connect adjacent harmonics
            for i in range(len(node_ids) - 1):
                self.connect(node_ids[i], node_ids[i+1], weight=weight_scale)
                self.connect(node_ids[i+1], node_ids[i], weight=weight_scale)
                
        elif connection_pattern == "harmonic_series":
            # Connect based on harmonic relationships
            for i, src_id in enumerate(node_ids):
                src_freq = self.nodes[src_id].frequency
                for j, tgt_id in enumerate(node_ids):
                    if i != j:
                        tgt_freq = self.nodes[tgt_id].frequency
                        # Weight based on harmonic relationship
                        ratio = tgt_freq / src_freq
                        if abs(ratio - round(ratio)) < 0.1:  # Close to integer ratio
                            weight = weight_scale * (1.0 / abs(i - j))
                            self.connect(src_id, tgt_id, weight=weight)
                            
        elif connection_pattern == "full":
            # Fully connected with distance-based weights
            for i, src_id in enumerate(node_ids):
                for j, tgt_id in enumerate(node_ids):
                    if i != j:
                        weight = weight_scale / (1 + abs(i - j))
                        self.connect(src_id, tgt_id, weight=weight)
                        
    def generate_chord_progression(self, chords: list[Union[str, list[float]]],
                                 duration_per_chord: float = 1.0,
                                 sample_rate: float = 44100) -> np.ndarray:
        """
        Generate a chord progression by retuning the network.
        
        Args:
            chords: List of chord definitions (frequency ratios)
            duration_per_chord: Duration of each chord in seconds
            sample_rate: Output sample rate
            
        Returns:
            Audio signal of the chord progression
        """
        output = []
        
        dt = 1.0 / sample_rate
        num_steps_per_chord = int(duration_per_chord * sample_rate)

        for chord in chords:
            self._retune_to_mode(self.base_freq, chord)
            
            for _ in range(num_steps_per_chord):
                self.step(dt)
                signals = self.get_signals()
                total_signal = sum(signals.values())
                output.append(total_signal.real)
                
        self.retuned_outputs = output
        return np.array(output)
    
    def morph_between_modes(self, target_mode: str, morph_time: float = 1.0,
                           sample_rate: float = 44100) -> np.ndarray:
        """
        Smoothly morph from current mode to target mode.
        
        Args:
            target_mode: Name of target musical mode
            morph_time: Time to morph in seconds
            sample_rate: Output sample rate
            
        Returns:
            Audio signal of the morphing process
        """
        if self.mode_detector is None:
            raise ValueError("Mode detector is required for mode morphing.")

        start_intervals = self.mode_detector.get_intervals(self.mode)
        end_intervals = self.mode_detector.get_intervals(target_mode)
        
        # Ensure interval lists have same length
        num_nodes = len(self.nodes)
        if len(start_intervals) < num_nodes:
            start_intervals.extend([start_intervals[-1]] * (num_nodes - len(start_intervals)))
        if len(end_intervals) < num_nodes:
            end_intervals.extend([end_intervals[-1]] * (num_nodes - len(end_intervals)))
            
        start_freqs = np.array([self.base_freq * i for i in start_intervals[:num_nodes]])
        end_freqs = np.array([self.base_freq * i for i in end_intervals[:num_nodes]])

        output = []
        num_steps = int(morph_time * sample_rate)
        dt = 1.0 / sample_rate
        
        for i in range(num_steps):
            morph_ratio = i / num_steps
            
            # Interpolate frequencies
            current_freqs = start_freqs * (1 - morph_ratio) + end_freqs * morph_ratio
            
            for j, node_id in enumerate(self.nodes.keys()):
                self.nodes[node_id].retune(current_freqs[j], rate=1.0) # Instant retune
                
            self.step(dt)
            signals = self.get_signals()
            total_signal = sum(signals.values())
            output.append(total_signal.real)

        # Update final mode
        self.mode = target_mode
        
        return np.array(output)
    
    def save_musical_state(self) -> Dict[str, Any]:
        """Save musical network state."""
        state = super().save_state()
        
        state['musical_parameters'] = {
            'base_freq': self.base_freq,
            'mode': self.mode,
            'tuning_system': self.tuning_system,
            'pitch_bend_range': self.pitch_bend_range
        }
        return state
    
    @classmethod
    def load_musical_state(cls, state: Dict[str, Any], 
                          mode_detector: Optional[Any] = None) -> 'MusicalResonantNetwork':
        """Load network from saved state including musical parameters."""
        # Create base network
        network = cls(
            name=state['name'],
            base_freq=state['musical_parameters']['base_freq'],
            mode=state['musical_parameters']['mode'],
            mode_detector=mode_detector
        )
        
        # Load base network state
        network.time = state['time']
        network.global_damping = state['parameters']['global_damping']
        network.coupling_strength = state['parameters']['coupling_strength']
        network.adaptation_rate = state['parameters']['adaptation_rate']
        
        # Load nodes
        for node_data in state['nodes'].values():
            node = ResonantNode.from_dict(node_data)
            network.add_node(node)
        
        # Load connections
        for conn_str, conn_data in state['connections'].items():
            src, tgt = conn_str.split('->')
            network.connect(src, tgt, conn_data['weight'], conn_data['delay'])
        
        # Load musical parameters
        musical_params = state['musical_parameters']
        network.tuning_system = musical_params['tuning_system']
        network.pitch_bend_range = musical_params['pitch_bend_range']
        
        return network 