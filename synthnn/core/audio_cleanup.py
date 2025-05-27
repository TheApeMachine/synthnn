"""
Audio Cleanup Engine for SynthNN

Provides resonance-based audio cleanup capabilities for removing artifacts
like persistent whistling tones from AI-generated audio (e.g., from Suno).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import scipy.signal as signal
from scipy.fft import fft, fftfreq, ifft

from .resonant_node import ResonantNode
from .resonant_network import ResonantNetwork
from .signal_processor import SignalProcessor
from .pattern_codec import AudioPatternEncoder, AudioPatternDecoder
from .musical_extensions import MusicalResonantNetwork
from applications.detector import ModeDetector 
from .accelerated_musical_network import AcceleratedMusicalNetwork
from ..performance import BackendManager # Added import


class ArtifactType(Enum):
    """Types of audio artifacts to clean."""
    WHISTLING = "whistling"
    HUM = "hum"
    NOISE = "noise"
    DISTORTION = "distortion"
    PHASE_ISSUES = "phase_issues"
    RESONANCE_PEAKS = "resonance_peaks"


@dataclass
class ArtifactProfile:
    """Profile of a detected artifact."""
    artifact_type: ArtifactType
    frequency: Optional[float] = None
    frequency_range: Optional[Tuple[float, float]] = None
    strength: float = 0.0
    time_range: Optional[Tuple[float, float]] = None
    phase_coherence: float = 1.0


class ArtifactDetector:
    """Detects various types of artifacts in audio."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.signal_processor = SignalProcessor()
        
    def detect_whistling(self, audio: np.ndarray, 
                        threshold: float = 0.1) -> List[ArtifactProfile]:
        """
        Detect persistent whistling tones.
        
        Args:
            audio: Audio signal
            threshold: Detection threshold
            
        Returns:
            List of detected whistling artifacts
        """
        artifacts = []
        
        # Analyze spectrum
        spectrum = np.abs(fft(audio))
        frequencies = fftfreq(len(audio), 1/self.sample_rate)
        
        # Find persistent narrow peaks
        # Look for frequencies with high energy but narrow bandwidth
        for i in range(1, len(spectrum)//2 - 1):
            if frequencies[i] > 0:
                # Check if this is a narrow peak
                local_energy = spectrum[i]
                neighbor_energy = (spectrum[i-1] + spectrum[i+1]) / 2
                
                if local_energy > threshold and local_energy > neighbor_energy * 2:
                    # Check persistence over time
                    persistence = self._check_frequency_persistence(
                        audio, frequencies[i]
                    )
                    
                    if persistence > 0.7:  # Present in >70% of time
                        artifacts.append(ArtifactProfile(
                            artifact_type=ArtifactType.WHISTLING,
                            frequency=frequencies[i],
                            strength=float(local_energy),
                            phase_coherence=persistence
                        ))
                        
        return artifacts
        
    def _check_frequency_persistence(self, audio: np.ndarray, 
                                   target_freq: float,
                                   window_size: int = 2048) -> float:
        """Check how persistently a frequency appears over time."""
        hop_size = window_size // 2
        num_windows = (len(audio) - window_size) // hop_size
        
        presence_count = 0
        
        for i in range(num_windows):
            start = i * hop_size
            window = audio[start:start + window_size]
            
            # Check if frequency is present in this window
            spectrum = np.abs(fft(window))
            frequencies = fftfreq(len(window), 1/self.sample_rate)
            
            # Find closest frequency bin
            freq_idx = np.argmin(np.abs(frequencies - target_freq))
            
            # Check if significant energy at this frequency
            if spectrum[freq_idx] > np.mean(spectrum) * 2:
                presence_count += 1
                
        return presence_count / num_windows if num_windows > 0 else 0
        
    def detect_harmonic_anomalies(self, audio: np.ndarray,
                                expected_mode: Optional[str] = None) -> List[ArtifactProfile]:
        """Detect frequencies that don't fit the expected harmonic structure."""
        artifacts = []
        
        # Use mode detector if available
        if expected_mode:
            mode_detector = ModeDetector()
            detected_mode, confidence = mode_detector.detect_mode(audio, self.sample_rate)
            
            if detected_mode != expected_mode and confidence > 0.5:
                # Find conflicting frequencies
                mode_frequencies = mode_detector._get_mode_frequencies(expected_mode, 440.0)
                spectrum = np.abs(fft(audio))
                frequencies = fftfreq(len(audio), 1/self.sample_rate)
                
                for i, (freq, energy) in enumerate(zip(frequencies[:len(frequencies)//2], 
                                                      spectrum[:len(spectrum)//2])):
                    if freq > 0 and energy > np.mean(spectrum) * 3:
                        # Check if this frequency fits the mode
                        is_harmonic = any(
                            abs(freq - mf) < 10 or  # Within 10Hz
                            abs(freq / mf - round(freq / mf)) < 0.05  # Harmonic ratio
                            for mf in mode_frequencies
                        )
                        
                        if not is_harmonic:
                            artifacts.append(ArtifactProfile(
                                artifact_type=ArtifactType.RESONANCE_PEAKS,
                                frequency=freq,
                                strength=float(energy)
                            ))
                            
        return artifacts


class ResonanceFilter:
    """Resonance-based filtering for artifact removal."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.network = ResonantNetwork("filter_network")
        
    def create_notch_resonator(self, frequency: float, 
                             damping: float = 0.99) -> ResonantNode:
        """Create a highly damped resonator for notch filtering."""
        node = ResonantNode(
            node_id=f"notch_{frequency:.1f}Hz",
            frequency=frequency,
            damping=damping  # High damping = narrow notch
        )
        return node
        
    def create_harmonic_enhancer(self, fundamental: float,
                               num_harmonics: int = 8) -> ResonantNetwork:
        """Create a network that enhances harmonic content."""
        network = ResonantNetwork("harmonic_enhancer")
        
        # Add nodes for fundamental and harmonics
        for h in range(1, num_harmonics + 1):
            freq = fundamental * h
            node = ResonantNode(
                node_id=f"harmonic_{h}",
                frequency=freq,
                damping=0.1  # Low damping = resonance
            )
            network.add_node(node)
            
            # Connect to previous harmonic
            if h > 1:
                network.connect(f"harmonic_{h-1}", f"harmonic_{h}", 
                              weight=0.5 / h)  # Decreasing weight
                              
        return network
        
    def apply_resonant_filter(self, audio: np.ndarray,
                            artifacts: List[ArtifactProfile]) -> np.ndarray:
        """Apply resonance-based filtering to remove artifacts."""
        filtered_audio = audio.copy()
        
        # Create notch filters for each artifact
        for artifact in artifacts:
            if artifact.frequency:
                # Create anti-resonance at artifact frequency
                notch_node = self.create_notch_resonator(
                    artifact.frequency,
                    damping=0.99
                )
                
                # Process audio through notch
                # This is a simplified version - in practice we'd implement
                # proper digital filtering
                filtered_audio = self._apply_notch(
                    filtered_audio,
                    artifact.frequency,
                    bandwidth=50.0  # Hz
                )
                
        return filtered_audio
        
    def _apply_notch(self, audio: np.ndarray, frequency: float,
                    bandwidth: float) -> np.ndarray:
        """Apply notch filter at specific frequency."""
        # Design notch filter
        nyquist = self.sample_rate / 2
        freq_norm = frequency / nyquist
        bandwidth_norm = bandwidth / nyquist
        
        # Use scipy to create notch filter
        from scipy.signal import iirnotch, filtfilt
        
        if 0 < freq_norm < 1:
            b, a = iirnotch(freq_norm, freq_norm / 10)  # Q = 10
            filtered = filtfilt(b, a, audio)
            return filtered
        else:
            return audio


class AudioCleanupEngine:
    """
    Main audio cleanup engine combining detection, filtering, and resynthesis.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.detector = ArtifactDetector(sample_rate)
        self.filter = ResonanceFilter(sample_rate)
        self.encoder = AudioPatternEncoder()
        self.decoder = AudioPatternDecoder()
        
    def analyze_artifacts(self, audio: np.ndarray,
                        expected_mode: Optional[str] = None) -> List[ArtifactProfile]:
        """
        Analyze audio for artifacts.
        
        Args:
            audio: Audio signal to analyze
            expected_mode: Expected musical mode (if known)
            
        Returns:
            List of detected artifacts
        """
        artifacts = []
        
        # Detect whistling tones
        whistles = self.detector.detect_whistling(audio)
        artifacts.extend(whistles)
        
        # Detect harmonic anomalies if mode is specified
        if expected_mode:
            anomalies = self.detector.detect_harmonic_anomalies(audio, expected_mode)
            artifacts.extend(anomalies)
            
        # Sort by strength
        artifacts.sort(key=lambda a: a.strength, reverse=True)
        
        return artifacts
        
    def cleanup_simple(self, audio: np.ndarray,
                      artifacts: Optional[List[ArtifactProfile]] = None) -> np.ndarray:
        """
        Simple cleanup using notch filtering.
        
        Args:
            audio: Audio to clean
            artifacts: Pre-detected artifacts (or None to auto-detect)
            
        Returns:
            Cleaned audio
        """
        if artifacts is None:
            artifacts = self.analyze_artifacts(audio)
            
        # Apply notch filters
        cleaned = self.filter.apply_resonant_filter(audio, artifacts)
        
        return cleaned
        
    def cleanup_resynthesis(self, audio: np.ndarray,
                          mode: Optional[str] = None,
                          preserve_transients: bool = True) -> np.ndarray:
        """
        Advanced cleanup using resonant resynthesis.
        
        This method:
        1. Encodes audio into resonant patterns
        2. Filters patterns in resonant domain
        3. Resynthesizes clean audio
        
        Args:
            audio: Audio to clean
            mode: Musical mode to enforce
            preserve_transients: Whether to preserve transient sounds
            
        Returns:
            Cleaned audio
        """
        # Detect artifacts
        artifacts = self.analyze_artifacts(audio, mode)
        
        # Encode to resonant representation
        pattern_dict = self.encoder.encode(audio)
        pattern = [
            (p['frequency'], p['amplitude'], p['phase'])
            for p in pattern_dict.values()
        ]
        
        # Filter in resonant domain
        filtered_pattern = self._filter_resonant_pattern(pattern, artifacts, mode)
        
        # Decode back to audio
        # Create musical network for decoding
        # Conditionally use AcceleratedMusicalNetwork if backend is available
        # Note: 'from ..performance import BackendManager' is now at the module level.
        
        backend_manager = BackendManager()
        use_accelerated = False # Default to false
        try:
            # Check if the preferred backend is available and usable
            if backend_manager.get_backend().is_available():
                use_accelerated = True
        except Exception:
            # Intentionally doing nothing if backend check fails,
            # use_accelerated will remain False.
            pass

        if use_accelerated:
            network = AcceleratedMusicalNetwork(
                name="cleanup_network_accelerated",
                mode=mode or "ionian"
                # base_freq will use default from AcceleratedMusicalNetwork constructor
            )
            # print("Using AcceleratedMusicalNetwork for resynthesis cleanup.") # Optional: for debugging
        else:
            network = MusicalResonantNetwork(
                name="cleanup_network",
                mode=mode or "ionian"
            )
            # print("Using MusicalResonantNetwork for resynthesis cleanup.") # Optional: for debugging
        
        # Configure network from filtered pattern
        for i, (freq, amp, phase) in enumerate(filtered_pattern): # Loop correctly placed
            if amp > 0.01:  # Threshold for including component
                node = ResonantNode(
                    node_id=f"component_{i}",
                    frequency=freq,
                    amplitude=amp,
                    phase=phase,
                    damping=0.1
                )
                network.add_node(node)
                
        # Generate cleaned audio
        duration = len(audio) / self.sample_rate
        cleaned = network.compute_harmonic_state(duration, self.sample_rate)
        
        # Preserve transients if requested
        if preserve_transients:
            transients = self._extract_transients(audio)
            cleaned = self._add_transients(cleaned, transients)
            
        return cleaned
        
    def _filter_resonant_pattern(self, pattern: List[Tuple[float, float, float]],
                               artifacts: List[ArtifactProfile],
                               mode: Optional[str] = None) -> List[Tuple[float, float, float]]:
        """Filter resonant pattern to remove artifacts."""
        filtered = []
        
        # Get artifact frequencies
        artifact_freqs = [a.frequency for a in artifacts if a.frequency]
        
        # Get mode frequencies if specified
        mode_freqs = []
        if mode:
            mode_detector = ModeDetector()
            mode_freqs = mode_detector._get_mode_frequencies(mode, 440.0)
            
        for freq, amp, phase in pattern:
            # Check if this component is an artifact
            is_artifact = any(
                abs(freq - af) < 10  # Within 10Hz
                for af in artifact_freqs
            )
            
            if is_artifact:
                # Heavily attenuate artifact
                filtered.append((freq, amp * 0.1, phase))
            elif mode_freqs:
                # Check if component fits the mode
                is_harmonic = any(
                    abs(freq - mf) < 20 or  # Close to mode frequency
                    abs(freq / mf - round(freq / mf)) < 0.1  # Harmonic
                    for mf in mode_freqs
                )
                
                if is_harmonic:
                    # Keep harmonic components
                    filtered.append((freq, amp, phase))
                else:
                    # Attenuate non-harmonic components
                    filtered.append((freq, amp * 0.3, phase))
            else:
                # No mode specified, keep all non-artifact components
                filtered.append((freq, amp, phase))
                
        return filtered
        
    def _extract_transients(self, audio: np.ndarray) -> np.ndarray:
        """Extract transient components from audio."""
        # Simple transient extraction using onset detection
        # This is a placeholder - real implementation would use
        # more sophisticated onset detection
        
        # High-pass filter to isolate transients
        from scipy.signal import butter, filtfilt
        
        nyquist = self.sample_rate / 2
        cutoff = 1000 / nyquist
        
        if cutoff < 1:
            b, a = butter(4, cutoff, btype='high')
            transients = filtfilt(b, a, audio)
            
            # Apply envelope follower
            envelope = np.abs(transients)
            
            # Threshold to keep only strong transients
            threshold = np.mean(envelope) * 3
            transients[envelope < threshold] = 0
            
            return transients
        else:
            return np.zeros_like(audio)
            
    def _add_transients(self, audio: np.ndarray, 
                       transients: np.ndarray) -> np.ndarray:
        """Add transients back to cleaned audio."""
        # Simple mixing - could be more sophisticated
        return audio + transients * 0.5
        
    def cleanup_adaptive(self, audio: np.ndarray,
                       reference_audio: Optional[np.ndarray] = None,
                       learning_rate: float = 0.01) -> np.ndarray:
        """
        Adaptive cleanup that learns from reference.
        
        Args:
            audio: Audio to clean
            reference_audio: Clean reference (if available)
            learning_rate: Adaptation rate
            
        Returns:
            Cleaned audio
        """
        # Create adaptive resonant network
        network = AcceleratedMusicalNetwork(
            "adaptive_cleanup",
            num_nodes=50,
            frequency_range=(20, 20000)
        )
        
        if reference_audio is not None:
            # Learn from reference
            ref_pattern = self.encoder.encode(reference_audio, self.sample_rate)
            
            # Configure network to match reference spectrum
            for freq, amp, phase in ref_pattern[:50]:  # Top 50 components
                # Find closest node
                distances = [abs(node.frequency - freq) for node in network.nodes.values()]
                closest_idx = np.argmin(distances)
                closest_node = list(network.nodes.values())[closest_idx]
                
                # Adapt node
                closest_node.frequency = freq * (1 - learning_rate) + closest_node.frequency * learning_rate
                closest_node.amplitude = amp
                
        # Process audio through adaptive network
        # The network will resonate with "good" frequencies and dampen artifacts
        network_input = audio / np.max(np.abs(audio))  # Normalize
        
        # Step through audio
        chunk_size = 1024
        output = np.zeros_like(audio)
        
        for i in range(0, len(audio) - chunk_size, chunk_size):
            chunk = network_input[i:i + chunk_size]
            
            # Process the entire chunk using the new method in AcceleratedMusicalNetwork
            # The process_audio_chunk method handles per-sample iteration on the device
            # and returns the calculated state for the chunk.
            # dt=0.001 was the value used in the original per-sample step call.
            dt_per_sample = 0.001 
            current_output_state = network.process_audio_chunk(chunk, dt_per_sample)
            
            output[i:i + chunk_size] = current_output_state
            
        return output * np.max(np.abs(audio))  # Restore scale


def create_cleanup_pipeline(style: str = "resynthesis") -> Callable:
    """
    Create a cleanup pipeline function.
    
    Args:
        style: Cleanup style ('simple', 'resynthesis', 'adaptive')
        
    Returns:
        Cleanup function
    """
    engine = AudioCleanupEngine()
    
    if style == "simple":
        return engine.cleanup_simple
    elif style == "resynthesis":
        return engine.cleanup_resynthesis
    elif style == "adaptive":
        return engine.cleanup_adaptive
    else:
        raise ValueError(f"Unknown cleanup style: {style}") 