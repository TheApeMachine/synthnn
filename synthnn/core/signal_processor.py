"""
Signal processing utilities for the SynthNN framework.

This module provides tools for analyzing, transforming, and processing
signals in the resonant network context.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from scipy import signal, fft
from scipy.signal import hilbert, find_peaks


class SignalProcessor:
    """
    Comprehensive signal processing toolkit for resonant networks.
    
    Handles frequency analysis, filtering, feature extraction, and
    signal transformations specific to resonant network dynamics.
    """
    
    def __init__(self, sample_rate: float = 44100.0):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
    
    def analyze_spectrum(self, signal_data: np.ndarray, 
                        window: str = 'hann') -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform frequency spectrum analysis on a signal.
        
        Args:
            signal_data: Input signal array
            window: Window function name for FFT
            
        Returns:
            frequencies: Array of frequency bins
            magnitudes: Array of magnitude values
        """
        # Apply window
        if window:
            window_func = signal.get_window(window, len(signal_data))
            windowed = signal_data * window_func
        else:
            windowed = signal_data
        
        # Compute FFT
        fft_vals = fft.fft(windowed)
        freqs = fft.fftfreq(len(windowed), 1/self.sample_rate)
        
        # Get positive frequencies only
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        magnitudes = np.abs(fft_vals[pos_mask])
        
        return freqs, magnitudes
    
    def extract_fundamental(self, signal_data: np.ndarray,
                           freq_range: Optional[Tuple[float, float]] = None) -> float:
        """
        Extract the fundamental frequency from a signal.
        
        Args:
            signal_data: Input signal
            freq_range: Optional frequency range to search within
            
        Returns:
            Estimated fundamental frequency
        """
        freqs, mags = self.analyze_spectrum(signal_data)
        
        # Apply frequency range filter if specified
        if freq_range:
            mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            freqs = freqs[mask]
            mags = mags[mask]
        
        # Find peaks in spectrum
        peaks, properties = find_peaks(mags, height=np.max(mags) * 0.1)
        
        if len(peaks) == 0:
            return 0.0
        
        # Return frequency of highest peak
        max_peak_idx = peaks[np.argmax(mags[peaks])]
        return freqs[max_peak_idx]
    
    def compute_phase_coherence(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute phase coherence matrix between multiple signals.
        
        Args:
            signals: Dictionary of signal arrays
            
        Returns:
            Coherence matrix (N x N) where N is number of signals
        """
        signal_list = list(signals.values())
        n_signals = len(signal_list)
        coherence_matrix = np.zeros((n_signals, n_signals))
        
        for i in range(n_signals):
            for j in range(i, n_signals):
                # Extract instantaneous phase using Hilbert transform
                phase_i = np.angle(hilbert(signal_list[i]))
                phase_j = np.angle(hilbert(signal_list[j]))
                
                # Compute phase locking value (PLV)
                phase_diff = phase_i - phase_j
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                
                coherence_matrix[i, j] = plv
                coherence_matrix[j, i] = plv
        
        return coherence_matrix
    
    def bandpass_filter(self, signal_data: np.ndarray,
                       low_freq: float, high_freq: float,
                       order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to signal.
        
        Args:
            signal_data: Input signal
            low_freq: Lower cutoff frequency
            high_freq: Upper cutoff frequency
            order: Filter order
            
        Returns:
            Filtered signal
        """
        # Normalize frequencies
        low_norm = low_freq / self.nyquist
        high_norm = high_freq / self.nyquist
        
        # Design filter
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        
        # Apply filter (forward-backward to preserve phase)
        filtered = signal.filtfilt(b, a, signal_data)
        
        return filtered
    
    def extract_envelope(self, signal_data: np.ndarray,
                        smooth_window: int = 100) -> np.ndarray:
        """
        Extract the amplitude envelope of a signal.
        
        Args:
            signal_data: Input signal
            smooth_window: Window size for smoothing
            
        Returns:
            Amplitude envelope
        """
        # Get analytic signal
        analytic = hilbert(signal_data)
        envelope = np.abs(analytic)
        
        # Smooth if requested
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            envelope = np.convolve(envelope, kernel, mode='same')
        
        return envelope
    
    def compute_spectral_centroid(self, signal_data: np.ndarray) -> float:
        """
        Compute the spectral centroid (center of mass of spectrum).
        
        Returns:
            Spectral centroid frequency
        """
        freqs, mags = self.analyze_spectrum(signal_data)
        
        # Compute weighted average
        centroid = np.sum(freqs * mags) / np.sum(mags)
        
        return centroid
    
    def compute_zero_crossing_rate(self, signal_data: np.ndarray) -> float:
        """
        Compute the zero-crossing rate of a signal.
        
        Returns:
            Zero crossings per second
        """
        # Count zero crossings
        zero_crossings = np.sum(np.abs(np.diff(np.sign(signal_data))) > 0)
        
        # Convert to rate
        duration = len(signal_data) / self.sample_rate
        zcr = zero_crossings / duration
        
        return zcr
    
    def adaptive_threshold(self, signal_data: np.ndarray,
                          window_size: int = 1000,
                          threshold_factor: float = 1.5) -> np.ndarray:
        """
        Apply adaptive thresholding for event detection.
        
        Args:
            signal_data: Input signal
            window_size: Size of moving window
            threshold_factor: Multiplier for threshold above mean
            
        Returns:
            Binary array indicating events
        """
        # Compute envelope
        envelope = self.extract_envelope(signal_data)
        
        # Compute moving statistics
        kernel = np.ones(window_size) / window_size
        moving_mean = np.convolve(envelope, kernel, mode='same')
        moving_std = np.sqrt(np.convolve((envelope - moving_mean)**2, kernel, mode='same'))
        
        # Adaptive threshold
        threshold = moving_mean + threshold_factor * moving_std
        
        # Detect events
        events = envelope > threshold
        
        return events

    def detect_anomalies(self, signal_data: np.ndarray,
                          window_size: int = 1000,
                          threshold_factor: float = 3.0) -> np.ndarray:
        """Detect anomalies in a time series using adaptive thresholding.

        Args:
            signal_data (np.ndarray): Input signal array.
            window_size (int): Size of the moving window for adaptive thresholding.
            threshold_factor (float): Multiplier for the threshold above the moving mean.

        Returns:
            np.ndarray: Binary array indicating detected anomalies (True where an anomaly is present).
        """
        events = self.adaptive_threshold(
            signal_data,
            window_size=window_size,
            threshold_factor=threshold_factor,
        )
        return events
    
    def resample(self, signal_data: np.ndarray,
                target_rate: float) -> np.ndarray:
        """
        Resample signal to a different sample rate.
        
        Args:
            signal_data: Input signal
            target_rate: Target sample rate
            
        Returns:
            Resampled signal
        """
        # Calculate resampling ratio
        ratio = target_rate / self.sample_rate
        
        # Resample
        new_length = int(len(signal_data) * ratio)
        resampled = signal.resample(signal_data, new_length)
        
        return resampled
    
    def compute_phase_velocity(self, signals: Dict[str, np.ndarray],
                              positions: Dict[str, float]) -> float:
        """
        Compute phase velocity across spatially distributed signals.
        
        Args:
            signals: Dictionary of signals keyed by node ID
            positions: Dictionary of spatial positions keyed by node ID
            
        Returns:
            Estimated phase velocity
        """
        # Extract phases
        phases = {}
        for node_id, sig in signals.items():
            analytic = hilbert(sig)
            phases[node_id] = np.angle(analytic)
        
        # Compute phase gradients
        velocities = []
        node_ids = list(signals.keys())
        
        for i in range(len(node_ids) - 1):
            node1, node2 = node_ids[i], node_ids[i + 1]
            
            # Spatial distance
            distance = abs(positions[node2] - positions[node1])
            
            # Phase difference
            phase_diff = np.mean(phases[node2] - phases[node1])
            
            # Velocity estimate
            if distance > 0:
                velocity = phase_diff / distance
                velocities.append(velocity)
        
        return np.mean(velocities) if velocities else 0.0 