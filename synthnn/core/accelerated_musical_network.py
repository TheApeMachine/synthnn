"""
GPU-accelerated musical resonant network.

This module combines the musical intelligence of MusicalResonantNetwork
with the performance benefits of the acceleration backends.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from .musical_extensions import MusicalResonantNetwork
from ..performance import BackendManager, BackendType


class AcceleratedMusicalNetwork(MusicalResonantNetwork):
    """
    Musical resonant network with automatic GPU/Metal acceleration.
    
    This class provides all the musical features while leveraging
    hardware acceleration for computationally intensive operations.
    """
    
    def __init__(self, name: str = "accelerated_musical", 
                 base_freq: float = 440.0,
                 mode: str = "Ionian", 
                 mode_detector: Optional[Any] = None,
                 backend: Optional[BackendType] = None):
        """
        Initialize accelerated musical network.
        
        Args:
            name: Network name
            base_freq: Base frequency (A4 = 440Hz)
            mode: Initial musical mode
            mode_detector: ModeDetector instance
            backend: Preferred backend, or None for auto-selection
        """
        super().__init__(name, base_freq, mode, mode_detector)
        
        # Initialize backend
        self.backend_manager = BackendManager(preferred_backend=backend)
        self.backend = self.backend_manager.get_backend()
        
        # Device arrays for acceleration
        self._device_phases = None
        self._device_frequencies = None
        self._device_amplitudes = None
        self._device_weights = None
        self._device_connections = None
        
        print(f"AcceleratedMusicalNetwork initialized with {type(self.backend).__name__}")
        
    def _sync_to_device(self):
        """Synchronize node data to device memory."""
        nodes = self.nodes
        n_nodes = len(nodes)
        
        if n_nodes == 0:
            return
            
        # Extract node data
        node_ids = list(nodes.keys())
        phases = np.array([nodes[nid].phase for nid in node_ids], dtype=np.float32)
        frequencies = np.array([nodes[nid].frequency for nid in node_ids], dtype=np.float32)
        amplitudes = np.array([nodes[nid].amplitude for nid in node_ids], dtype=np.float32)
        
        # Build weight and connection matrices
        weights = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        connections = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        for i, src_id in enumerate(node_ids):
            for j, tgt_id in enumerate(node_ids):
                if (src_id, tgt_id) in self.connections:
                    conn = self.connections[(src_id, tgt_id)]
                    weights[i, j] = conn.weight
                    connections[i, j] = 1.0
                    
        # Transfer to device
        self._device_phases = self.backend.to_device(phases)
        self._device_frequencies = self.backend.to_device(frequencies)
        self._device_amplitudes = self.backend.to_device(amplitudes)
        self._device_weights = self.backend.to_device(weights)
        self._device_connections = self.backend.to_device(connections)
        
    def _sync_from_device(self):
        """Synchronize device data back to nodes."""
        if self._device_phases is None:
            return
            
        # Transfer back
        phases = self.backend.to_host(self._device_phases)
        amplitudes = self.backend.to_host(self._device_amplitudes)
        frequencies = self.backend.to_host(self._device_frequencies)
        
        # Update nodes
        node_ids = list(self.nodes.keys())
        for i, nid in enumerate(node_ids):
            self.nodes[nid].phase = float(phases[i])
            self.nodes[nid].amplitude = float(amplitudes[i])
            self.nodes[nid].frequency = float(frequencies[i])
            
    def add_node(self, node):
        """Add node and sync to device."""
        super().add_node(node)
        self._sync_to_device()
        
    def connect(self, source_id: str, target_id: str, weight: float = 1.0, delay: float = 0.0):
        """Create connection and sync to device."""
        super().connect(source_id, target_id, weight, delay)
        self._sync_to_device()
        
    def step(self, dt: float, external_inputs: Optional[Dict[str, float]] = None):
        """
        Accelerated network step using compute backend.
        """
        if len(self.nodes) == 0:
            return
            
        # Ensure device sync
        self._sync_to_device()
        
        # Apply external inputs if any
        if external_inputs:
            node_ids = list(self.nodes.keys())
            amp_host = self.backend.to_host(self._device_amplitudes)
            
            for node_id, signal in external_inputs.items():
                if node_id in node_ids:
                    idx = node_ids.index(node_id)
                    # Update amplitude based on stimulus
                    amp_host[idx] *= (1 + np.tanh(signal))
                    
            self._device_amplitudes = self.backend.to_device(amp_host)
        
        # Compute phase coupling on device
        coupling = self.backend.phase_coupling(
            self._device_phases,
            self._device_weights,
            self._device_connections
        )
        
        # Update phases
        coupling_scaled = self.backend.multiply(
            coupling,
            self.backend.to_device(np.array(self.coupling_strength * dt, dtype=np.float32))
        )
        
        # Natural frequency advance
        freq_advance = self.backend.multiply(
            self._device_frequencies,
            self.backend.to_device(np.array(2 * np.pi * dt, dtype=np.float32))
        )
        
        # Total phase change
        phase_change = self.backend.add(freq_advance, coupling_scaled)
        
        # Update phases
        self._device_phases = self.backend.add(self._device_phases, phase_change)
        
        # Apply damping to amplitudes
        damping_factor = 1 - self.global_damping * dt
        self._device_amplitudes = self.backend.multiply(
            self._device_amplitudes,
            self.backend.to_device(np.array(damping_factor, dtype=np.float32))
        )
        
        # Sync back to CPU
        self._sync_from_device()
        
        # Record history
        self._record_state()
        
        # Increment time
        self.time += dt
        
    def generate_audio_accelerated(self, duration: float, sample_rate: float = 44100) -> np.ndarray:
        """
        Generate audio using accelerated oscillator bank.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Audio signal array
        """
        if len(self.nodes) == 0:
            return np.zeros(int(duration * sample_rate))
            
        # Ensure device sync
        self._sync_to_device()
        
        # Generate time points
        time_points = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
        time_device = self.backend.to_device(time_points)
        
        # Compute oscillator bank on device
        signals_device = self.backend.oscillator_bank(
            self._device_frequencies,
            self._device_phases,
            self._device_amplitudes,
            time_device
        )
        
        # Sum all oscillators
        summed_device = self.backend.sum(signals_device, axis=0)
        
        # Transfer back and normalize
        audio = self.backend.to_host(summed_device)
        
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
            
        return audio
        
    def compute_harmonic_state(self, duration: float, sample_rate: float = 44100) -> np.ndarray:
        """
        Override to use accelerated generation.
        """
        # Clear outputs
        self.harmonic_outputs = []
        
        # Generate using accelerated method
        audio = self.generate_audio_accelerated(duration, sample_rate)
        
        # Store for compatibility
        num_samples = len(audio)
        for i in range(num_samples):
            # Approximate individual node outputs
            node_outputs = [audio[i] / len(self.nodes)] * len(self.nodes)
            self.harmonic_outputs.append(node_outputs)
            
        return audio
        
    def morph_between_modes_accelerated(self, target_mode: str, 
                                       morph_time: float = 1.0,
                                       sample_rate: float = 44100) -> np.ndarray:
        """
        Accelerated mode morphing using device computation.
        """
        if self.mode_detector is None:
            raise ValueError("Mode detector required for mode morphing")
            
        # Get intervals
        current_intervals = self.mode_detector.mode_intervals[self.mode]
        target_intervals = self.mode_detector.mode_intervals[target_mode]
        
        # Prepare intervals
        max_len = max(len(current_intervals), len(target_intervals))
        current_intervals = np.array(current_intervals[:max_len] + [1.0] * (max_len - len(current_intervals)), dtype=np.float32)
        target_intervals = np.array(target_intervals[:max_len] + [1.0] * (max_len - len(target_intervals)), dtype=np.float32)
        
        # Transfer to device
        current_device = self.backend.to_device(current_intervals)
        target_device = self.backend.to_device(target_intervals)
        
        # Generate morph steps
        num_steps = 100  # Number of interpolation steps
        step_duration = morph_time / num_steps
        output_segments = []
        
        for step in range(num_steps):
            # Calculate morph factor
            morph_factor = step / num_steps
            morph_factor_device = self.backend.to_device(np.array(morph_factor, dtype=np.float32))
            
            # Interpolate on device
            one_minus_morph = self.backend.to_device(np.array(1 - morph_factor, dtype=np.float32))
            
            current_scaled = self.backend.multiply(current_device, one_minus_morph)
            target_scaled = self.backend.multiply(target_device, morph_factor_device)
            morphed_intervals_device = self.backend.add(current_scaled, target_scaled)
            
            # Get intervals back
            morphed_intervals = self.backend.to_host(morphed_intervals_device)
            
            # Retune network
            self._retune_to_mode(self.base_freq, morphed_intervals)
            
            # Generate audio segment
            segment = self.generate_audio_accelerated(step_duration, sample_rate)
            output_segments.append(segment)
            
        self.mode = target_mode
        return np.concatenate(output_segments)
        
    def analyze_spectrum_accelerated(self, signal: np.ndarray, 
                                   sample_rate: float = 44100) -> Dict[str, Any]:
        """
        Perform accelerated spectrum analysis using device FFT.
        """
        # Transfer signal to device
        signal_device = self.backend.to_device(signal.astype(np.float32))
        
        # Compute FFT on device
        fft_device = self.backend.rfft(signal_device)
        
        # Get magnitude spectrum
        fft_host = self.backend.to_host(fft_device)
        magnitudes = np.abs(fft_host)
        
        # Frequency bins
        freqs = np.fft.rfftfreq(len(signal), 1/sample_rate)
        
        # Find peaks using device operations
        magnitudes_device = self.backend.to_device(magnitudes)
        max_mag = self.backend.max(magnitudes_device)
        threshold = self.backend.multiply(max_mag, self.backend.to_device(np.array(0.1, dtype=np.float32)))
        
        # Transfer back for peak finding (could be optimized further)
        threshold_host = self.backend.to_host(threshold)
        peaks = magnitudes > threshold_host
        
        return {
            'frequencies': freqs,
            'magnitudes': magnitudes,
            'peaks': peaks,
            'fundamental': freqs[np.argmax(magnitudes[:len(magnitudes)//2])]
        }
        
    def batch_process_signals(self, signals: List[np.ndarray], 
                            operation: str = "analyze") -> List[Any]:
        """
        Process multiple signals in parallel using device acceleration.
        
        Args:
            signals: List of input signals
            operation: Type of operation ("analyze", "harmonize", etc.)
            
        Returns:
            List of results
        """
        # Stack signals for batch processing
        max_len = max(len(sig) for sig in signals)
        padded_signals = np.zeros((len(signals), max_len), dtype=np.float32)
        
        for i, sig in enumerate(signals):
            padded_signals[i, :len(sig)] = sig
            
        # Transfer batch to device
        batch_device = self.backend.to_device(padded_signals)
        
        results = []
        
        if operation == "analyze":
            # Batch FFT
            fft_batch = self.backend.fft(batch_device, axis=1)
            fft_host = self.backend.to_host(fft_batch)
            
            for i in range(len(signals)):
                spectrum = np.abs(fft_host[i, :len(signals[i])//2])
                results.append({
                    'spectrum': spectrum,
                    'fundamental': np.argmax(spectrum) * 44100 / len(signals[i])
                })
                
        elif operation == "harmonize":
            # Process each signal through the network
            for i, sig in enumerate(signals):
                self.analyze_and_retune(sig)
                harmonized = self.generate_audio_accelerated(len(sig) / 44100)
                results.append(harmonized)
                
        return results
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the current backend."""
        return {
            'backend': type(self.backend).__name__,
            'device_info': self.backend.get_device_info(),
            'capabilities': self.backend.get_capabilities(),
            'num_nodes': len(self.nodes),
            'num_connections': len(self.connections),
            'memory_usage': {
                'phases': self._device_phases.nbytes if self._device_phases is not None else 0,
                'frequencies': self._device_frequencies.nbytes if self._device_frequencies is not None else 0,
                'amplitudes': self._device_amplitudes.nbytes if self._device_amplitudes is not None else 0,
                'weights': self._device_weights.nbytes if self._device_weights is not None else 0,
            }
        } 