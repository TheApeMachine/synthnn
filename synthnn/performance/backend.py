"""
Abstract backend interface for SynthNN performance optimization.

This module defines the common interface that all compute backends must implement,
allowing seamless switching between CPU, CUDA, and Metal implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from enum import Enum


class BackendType(Enum):
    """Available backend types."""
    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"


class ComputeBackend(ABC):
    """Abstract base class for compute backends."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.is_initialized = False
        self._capabilities = {}
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend and detect capabilities."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the compute device."""
        pass
    
    # Memory management
    @abstractmethod
    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype) -> Any:
        """Allocate memory on the device."""
        pass
    
    @abstractmethod
    def free(self, array: Any) -> None:
        """Free memory on the device."""
        pass
    
    @abstractmethod
    def to_device(self, host_array: np.ndarray) -> Any:
        """Transfer array from host to device."""
        pass
    
    @abstractmethod
    def to_host(self, device_array: Any) -> np.ndarray:
        """Transfer array from device to host."""
        pass
    
    # Basic operations
    @abstractmethod
    def add(self, a: Any, b: Any, out: Optional[Any] = None) -> Any:
        """Element-wise addition."""
        pass
    
    @abstractmethod
    def multiply(self, a: Any, b: Any, out: Optional[Any] = None) -> Any:
        """Element-wise multiplication."""
        pass
    
    @abstractmethod
    def sin(self, x: Any, out: Optional[Any] = None) -> Any:
        """Element-wise sine."""
        pass
    
    @abstractmethod
    def cos(self, x: Any, out: Optional[Any] = None) -> Any:
        """Element-wise cosine."""
        pass
    
    @abstractmethod
    def exp(self, x: Any, out: Optional[Any] = None) -> Any:
        """Element-wise exponential."""
        pass
    
    # Reductions
    @abstractmethod
    def sum(self, x: Any, axis: Optional[int] = None) -> Any:
        """Sum reduction."""
        pass
    
    @abstractmethod
    def mean(self, x: Any, axis: Optional[int] = None) -> Any:
        """Mean reduction."""
        pass
    
    @abstractmethod
    def max(self, x: Any, axis: Optional[int] = None) -> Any:
        """Maximum reduction."""
        pass
    
    # Linear algebra
    @abstractmethod
    def dot(self, a: Any, b: Any, out: Optional[Any] = None) -> Any:
        """Matrix multiplication."""
        pass
    
    @abstractmethod
    def norm(self, x: Any, ord: Optional[int] = None, axis: Optional[int] = None) -> Any:
        """Vector/matrix norm."""
        pass
    
    # FFT operations
    @abstractmethod
    def fft(self, x: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """Fast Fourier Transform."""
        pass
    
    @abstractmethod
    def ifft(self, x: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """Inverse Fast Fourier Transform."""
        pass
    
    @abstractmethod
    def rfft(self, x: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """Real FFT."""
        pass
    
    # Specialized operations for resonant networks
    @abstractmethod
    def phase_coupling(self, phases: Any, weights: Any, connections: Any) -> Any:
        """
        Compute phase coupling for resonant network.
        
        Args:
            phases: Array of node phases
            weights: Connection weight matrix
            connections: Adjacency matrix
            
        Returns:
            Phase coupling values for each node
        """
        pass
    
    @abstractmethod
    def oscillator_bank(self, frequencies: Any, phases: Any, amplitudes: Any, 
                       time_points: Any) -> Any:
        """
        Compute signals from a bank of oscillators in parallel.
        
        Args:
            frequencies: Array of frequencies for each oscillator
            phases: Array of phases
            amplitudes: Array of amplitudes
            time_points: Time points to evaluate
            
        Returns:
            Signal matrix (oscillators x time_points)
        """
        pass
    
    @abstractmethod
    def batch_hilbert(self, signals: Any) -> Any:
        """
        Compute Hilbert transform for multiple signals in parallel.
        
        Args:
            signals: 2D array (batch x time)
            
        Returns:
            Analytic signals
        """
        pass
    
    # Synchronization utilities
    def synchronize(self) -> None:
        """Synchronize device operations (if needed)."""
        pass
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get backend capabilities."""
        return self._capabilities
    
    def benchmark_operation(self, operation: str, size: int = 1000000) -> float:
        """
        Benchmark a specific operation.
        
        Args:
            operation: Name of operation to benchmark
            size: Size of test data
            
        Returns:
            Time in seconds
        """
        import time
        
        # Create test data
        a = self.to_device(np.random.randn(size).astype(np.float32))
        b = self.to_device(np.random.randn(size).astype(np.float32))
        
        # Warmup
        for _ in range(10):
            if operation == "add":
                _ = self.add(a, b)
            elif operation == "multiply":
                _ = self.multiply(a, b)
            elif operation == "sin":
                _ = self.sin(a)
            elif operation == "fft":
                _ = self.fft(a)
        
        self.synchronize()
        
        # Benchmark
        start = time.time()
        iterations = 100
        
        for _ in range(iterations):
            if operation == "add":
                _ = self.add(a, b)
            elif operation == "multiply":
                _ = self.multiply(a, b)
            elif operation == "sin":
                _ = self.sin(a)
            elif operation == "fft":
                _ = self.fft(a)
        
        self.synchronize()
        elapsed = time.time() - start
        
        # Cleanup
        self.free(a)
        self.free(b)
        
        return elapsed / iterations 