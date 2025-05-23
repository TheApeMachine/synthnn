"""
Musical extensions for the SynthNN core framework.

This module bridges the gap between the core ResonantNetwork and the music-specific
features from the original implementation, enabling mode-aware musical networks
with GPU acceleration support.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from .resonant_network import ResonantNetwork
from .resonant_node import ResonantNode
from .signal_processor import SignalProcessor


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
        Compute the pure harmonic output of the network.
        
        Returns:
            Array of summed network output
        """
        num_samples = int(duration * sample_rate)
        time_steps = np.linspace(0, duration, num_samples)
        
        self.harmonic_outputs = []
        
        for t in time_steps:
            # Get signals from all nodes
            signals = self.get_signals()
            node_outputs = list(signals.values())
            self.harmonic_outputs.append(node_outputs)
            
        # Return summed output
        return np.sum(self.harmonic_outputs, axis=1)
    
    def compute_dissonant_state(self, duration: float, foreign_signal: np.ndarray,
                               sample_rate: float = 44100) -> np.ndarray:
        """
        Compute network output with foreign signal interference.
        """
        num_samples = int(duration * sample_rate)
        time_steps = np.linspace(0, duration, num_samples)
        
        self.dissonant_outputs = []
        
        # Ensure foreign_signal matches our time steps
        if len(foreign_signal) != num_samples:
            # Resample if needed
            foreign_signal = self.signal_processor.resample(
                foreign_signal, len(foreign_signal), num_samples
            )
        
        for i, t in enumerate(time_steps):
            signals = self.get_signals()
            node_outputs = [sig + foreign_signal[i] for sig in signals.values()]
            self.dissonant_outputs.append(node_outputs)
            
        return np.sum(self.dissonant_outputs, axis=1)
    
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
    
    def _retune_to_mode(self, new_base_freq: float, mode_intervals: List[float]) -> None:
        """
        Retune all nodes to new base frequency with mode intervals.
        """
        self.base_freq = new_base_freq
        
        # Ensure we have the right number of intervals
        num_nodes = len(self.nodes)
        if len(mode_intervals) < num_nodes:
            # Repeat last interval or generate octaves
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
                        
    def generate_chord_progression(self, chords: List[List[float]], 
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
        
        for chord_ratios in chords:
            # Retune network to chord
            self._retune_to_mode(self.base_freq, chord_ratios)
            
            # Generate chord audio
            chord_audio = self.compute_harmonic_state(
                duration_per_chord, sample_rate
            )
            output.extend(chord_audio)
            
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
            raise ValueError("Mode detector required for mode morphing")
            
        # Get current and target intervals
        current_intervals = self.mode_detector.mode_intervals[self.mode]
        target_intervals = self.mode_detector.mode_intervals[target_mode]
        
        # Ensure same length
        max_len = max(len(current_intervals), len(target_intervals))
        current_intervals = current_intervals[:max_len] + [1.0] * (max_len - len(current_intervals))
        target_intervals = target_intervals[:max_len] + [1.0] * (max_len - len(target_intervals))
        
        # Generate morph
        num_samples = int(morph_time * sample_rate)
        output = np.zeros(num_samples)
        
        for i in range(num_samples):
            # Calculate morph factor (0 to 1)
            morph_factor = i / num_samples
            
            # Interpolate intervals
            morphed_intervals = [
                current * (1 - morph_factor) + target * morph_factor
                for current, target in zip(current_intervals, target_intervals)
            ]
            
            # Retune and compute single sample
            self._retune_to_mode(self.base_freq, morphed_intervals)
            
            # Step network
            self.step(1.0 / sample_rate)
            
            # Sum outputs
            signals = self.get_signals()
            output[i] = sum(signals.values())
            
        self.mode = target_mode
        return output
    
    def save_musical_state(self) -> Dict[str, Any]:
        """Save network state including musical parameters."""
        base_state = super().save_state()
        
        musical_state = {
            'base_freq': self.base_freq,
            'mode': self.mode,
            'tuning_system': self.tuning_system,
            'pitch_bend_range': self.pitch_bend_range,
            'harmonic_outputs': self.harmonic_outputs,
            'dissonant_outputs': self.dissonant_outputs,
            'retuned_outputs': self.retuned_outputs
        }
        
        base_state['musical_state'] = musical_state
        return base_state
    
    @classmethod
    def load_musical_state(cls, state: Dict[str, Any], 
                          mode_detector: Optional[Any] = None) -> 'MusicalResonantNetwork':
        """Load network from saved state including musical parameters."""
        # Create base network
        network = cls(
            name=state['name'],
            base_freq=state['musical_state']['base_freq'],
            mode=state['musical_state']['mode'],
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
        
        # Load musical state
        musical_state = state['musical_state']
        network.tuning_system = musical_state['tuning_system']
        network.pitch_bend_range = musical_state['pitch_bend_range']
        network.harmonic_outputs = musical_state['harmonic_outputs']
        network.dissonant_outputs = musical_state['dissonant_outputs']
        network.retuned_outputs = musical_state['retuned_outputs']
        
        return network 