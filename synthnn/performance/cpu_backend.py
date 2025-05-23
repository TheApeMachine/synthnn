"""
CPU backend implementation for SynthNN.

This backend uses NumPy with optimizations like vectorization, 
multi-threading via NumPy's BLAS/LAPACK, and efficient memory access patterns.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from scipy import fft as sp_fft
from scipy.signal import hilbert
import multiprocessing
import os

from .backend import ComputeBackend


class CPUBackend(ComputeBackend):
    """CPU backend using optimized NumPy operations."""
    
    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self.num_threads = multiprocessing.cpu_count()
        
    def initialize(self) -> None:
        """Initialize the CPU backend."""
        # Set optimal number of threads for NumPy
        os.environ['OMP_NUM_THREADS'] = str(self.num_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.num_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.num_threads)
        
        self._capabilities = {
            'backend_type': 'cpu',
            'num_threads': self.num_threads,
            'simd_support': self._detect_simd(),
            'max_memory': self._get_available_memory(),
            'preferred_dtype': np.float32
        }
        self.is_initialized = True
    
    def is_available(self) -> bool:
        """CPU backend is always available."""
        return True
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        import platform
        return {
            'device_name': platform.processor(),
            'device_type': 'CPU',
            'num_cores': multiprocessing.cpu_count(),
            'architecture': platform.machine(),
            'system': platform.system()
        }
    
    def _detect_simd(self) -> Dict[str, bool]:
        """Detect SIMD capabilities."""
        # This is a simplified detection
        return {
            'sse': True,  # Most modern CPUs have SSE
            'avx': True,  # Assume AVX is available
            'avx2': True,  # Common in recent CPUs
            'avx512': False  # Less common
        }
    
    def _get_available_memory(self) -> int:
        """Get available system memory in bytes."""
        import psutil
        return psutil.virtual_memory().available if hasattr(psutil, 'virtual_memory') else 8 * 1024**3
    
    # Memory management
    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Allocate aligned memory for better performance."""
        # Use np.empty for faster allocation (no initialization)
        return np.empty(shape, dtype=dtype)
    
    def free(self, array: np.ndarray) -> None:
        """Free memory (handled by garbage collector)."""
        del array
    
    def to_device(self, host_array: np.ndarray) -> np.ndarray:
        """No-op for CPU backend."""
        return np.asarray(host_array)
    
    def to_host(self, device_array: np.ndarray) -> np.ndarray:
        """No-op for CPU backend."""
        return np.asarray(device_array)
    
    # Basic operations (vectorized)
    def add(self, a: np.ndarray, b: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized addition."""
        return np.add(a, b, out=out)
    
    def multiply(self, a: np.ndarray, b: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized multiplication."""
        return np.multiply(a, b, out=out)
    
    def sin(self, x: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized sine."""
        return np.sin(x, out=out)
    
    def cos(self, x: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized cosine."""
        return np.cos(x, out=out)
    
    def exp(self, x: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized exponential."""
        return np.exp(x, out=out)
    
    # Reductions
    def sum(self, x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """Sum reduction."""
        return np.sum(x, axis=axis)
    
    def mean(self, x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """Mean reduction."""
        return np.mean(x, axis=axis)
    
    def max(self, x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """Maximum reduction."""
        return np.max(x, axis=axis)
    
    # Linear algebra
    def dot(self, a: np.ndarray, b: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Matrix multiplication using BLAS."""
        return np.dot(a, b, out=out)
    
    def norm(self, x: np.ndarray, ord: Optional[int] = None, axis: Optional[int] = None) -> np.ndarray:
        """Vector/matrix norm."""
        return np.linalg.norm(x, ord=ord, axis=axis)
    
    # FFT operations (using scipy for better performance)
    def fft(self, x: np.ndarray, n: Optional[int] = None, axis: int = -1) -> np.ndarray:
        """Fast Fourier Transform."""
        return sp_fft.fft(x, n=n, axis=axis)
    
    def ifft(self, x: np.ndarray, n: Optional[int] = None, axis: int = -1) -> np.ndarray:
        """Inverse Fast Fourier Transform."""
        return sp_fft.ifft(x, n=n, axis=axis)
    
    def rfft(self, x: np.ndarray, n: Optional[int] = None, axis: int = -1) -> np.ndarray:
        """Real FFT."""
        return sp_fft.rfft(x, n=n, axis=axis)
    
    # Specialized operations for resonant networks
    def phase_coupling(self, phases: np.ndarray, weights: np.ndarray, 
                      connections: np.ndarray) -> np.ndarray:
        """
        Vectorized phase coupling computation.
        
        Uses broadcasting for efficient computation of Kuramoto-style coupling.
        """
        # Compute phase differences using broadcasting
        phase_diff = phases[:, np.newaxis] - phases[np.newaxis, :]
        
        # Apply sine for Kuramoto coupling
        coupling_matrix = np.sin(phase_diff)
        
        # Apply weights and connections
        weighted_coupling = coupling_matrix * weights * connections
        
        # Sum over inputs for each node
        return np.sum(weighted_coupling, axis=1)
    
    def oscillator_bank(self, frequencies: np.ndarray, phases: np.ndarray, 
                       amplitudes: np.ndarray, time_points: np.ndarray) -> np.ndarray:
        """
        Vectorized computation of multiple oscillators.
        
        Returns matrix of shape (n_oscillators, n_time_points).
        """
        # Reshape for broadcasting
        freqs = frequencies[:, np.newaxis]
        phases_reshaped = phases[:, np.newaxis]
        amps = amplitudes[:, np.newaxis]
        times = time_points[np.newaxis, :]
        
        # Compute all oscillators at once
        phase_matrix = 2 * np.pi * freqs * times + phases_reshaped
        signals = amps * np.sin(phase_matrix)
        
        return signals
    
    def batch_hilbert(self, signals: np.ndarray) -> np.ndarray:
        """
        Batch Hilbert transform using scipy.
        
        For better performance, processes multiple signals at once.
        """
        if signals.ndim == 1:
            return hilbert(signals)
        else:
            # Process each signal in the batch
            analytic_signals = np.empty_like(signals, dtype=np.complex128)
            for i in range(signals.shape[0]):
                analytic_signals[i] = hilbert(signals[i])
            return analytic_signals 