"""
Metal backend implementation for SynthNN.

This backend uses PyMetal or Metal Performance Shaders for GPU acceleration
on Apple Silicon (M1/M2/M3) and Intel Macs with AMD GPUs.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import platform
import os

from .backend import ComputeBackend


class MetalBackend(ComputeBackend):
    """Metal backend for Apple GPUs."""
    
    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self.device = None
        self.queue = None
        self.mlx_available = False
        self.mps_available = False

    @property
    def backend_type(self):
        from .backend import BackendType
        return BackendType.METAL
        
    def initialize(self) -> None:
        """Initialize Metal backend."""
        # Try MLX first (Apple's new ML framework)
        try:
            import mlx.core as mx
            self.mlx_available = True
            self.mx = mx
            mx.set_default_device(mx.gpu)
        except ImportError:
            self.mlx_available = False
            
        # Try PyTorch MPS backend as fallback
        try:
            import torch
            if torch.backends.mps.is_available():
                self.mps_available = True
                self.torch = torch
                self.device = torch.device("mps")
        except ImportError:
            self.mps_available = False
        
        if not self.mlx_available and not self.mps_available:
            raise RuntimeError("No Metal backend available. Install mlx or torch with MPS support.")
        
        self._capabilities = {
            'backend_type': 'metal',
            'mlx_available': self.mlx_available,
            'mps_available': self.mps_available,
            'unified_memory': True,  # Apple Silicon feature
            'max_memory': self._get_gpu_memory(),
            'preferred_dtype': np.float32
        }
        self.is_initialized = True
    
    def is_available(self) -> bool:
        """Check if Metal is available."""
        if platform.system() != 'Darwin':
            return False
            
        # Check for MLX
        try:
            import mlx.core as mx
            return True
        except ImportError:
            pass
            
        # Check for PyTorch MPS
        try:
            import torch
            return torch.backends.mps.is_available()
        except ImportError:
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get Metal device information."""
        info = {
            'device_type': 'Metal GPU',
            'system': platform.system(),
            'architecture': platform.machine()
        }
        
        # Get Apple Silicon info if available
        if platform.processor() == 'arm':
            info['silicon_type'] = 'Apple Silicon'
            # Try to detect specific chip
            try:
                import subprocess
                chip_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
                info['chip'] = chip_info
            except:
                info['chip'] = 'Unknown Apple Silicon'
        
        return info
    
    def _get_gpu_memory(self) -> int:
        """Estimate available GPU memory."""
        # Apple Silicon has unified memory
        try:
            import subprocess
            # Get total system memory
            mem_bytes = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']).decode().strip())
            # Assume we can use up to 75% for GPU on Apple Silicon
            return int(mem_bytes * 0.75)
        except:
            return 8 * 1024**3  # Default 8GB
    
    # Memory management
    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> Any:
        """Allocate memory on Metal device."""
        if self.mlx_available:
            return self.mx.zeros(shape, dtype=self._numpy_to_mlx_dtype(dtype))
        elif self.mps_available:
            return self.torch.zeros(shape, dtype=self._numpy_to_torch_dtype(dtype), device=self.device)
    
    def free(self, array: Any) -> None:
        """Free memory (handled automatically)."""
        del array
        if self.mlx_available:
            self.mx.synchronize()
        elif self.mps_available:
            self.torch.mps.synchronize()
    
    def to_device(self, host_array: np.ndarray) -> Any:
        """Transfer array to Metal device."""
        if self.mlx_available:
            return self.mx.array(host_array)
        elif self.mps_available:
            return self.torch.from_numpy(host_array).to(self.device)
    
    def to_host(self, device_array: Any) -> np.ndarray:
        """Transfer array from Metal device to host."""
        if self.mlx_available:
            return np.array(device_array)
        elif self.mps_available:
            return device_array.cpu().numpy()
    
    # Basic operations
    def add(self, a: Any, b: Any, out: Optional[Any] = None) -> Any:
        """Element-wise addition."""
        if self.mlx_available:
            result = self.mx.add(a, b)
            if out is not None:
                out[:] = result
            return result
        elif self.mps_available:
            return self.torch.add(a, b, out=out)
    
    def multiply(self, a: Any, b: Any, out: Optional[Any] = None) -> Any:
        """Element-wise multiplication."""
        if self.mlx_available:
            result = self.mx.multiply(a, b)
            if out is not None:
                out[:] = result
            return result
        elif self.mps_available:
            return self.torch.multiply(a, b, out=out)
    
    def sin(self, x: Any, out: Optional[Any] = None) -> Any:
        """Element-wise sine."""
        if self.mlx_available:
            result = self.mx.sin(x)
            if out is not None:
                out[:] = result
            return result
        elif self.mps_available:
            return self.torch.sin(x, out=out)
    
    def cos(self, x: Any, out: Optional[Any] = None) -> Any:
        """Element-wise cosine."""
        if self.mlx_available:
            result = self.mx.cos(x)
            if out is not None:
                out[:] = result
            return result
        elif self.mps_available:
            return self.torch.cos(x, out=out)
    
    def exp(self, x: Any, out: Optional[Any] = None) -> Any:
        """Element-wise exponential."""
        if self.mlx_available:
            result = self.mx.exp(x)
            if out is not None:
                out[:] = result
            return result
        elif self.mps_available:
            return self.torch.exp(x, out=out)
    
    # Reductions
    def sum(self, x: Any, axis: Optional[int] = None) -> Any:
        """Sum reduction."""
        if self.mlx_available:
            return self.mx.sum(x, axis=axis)
        elif self.mps_available:
            return self.torch.sum(x, dim=axis)
    
    def mean(self, x: Any, axis: Optional[int] = None) -> Any:
        """Mean reduction."""
        if self.mlx_available:
            return self.mx.mean(x, axis=axis)
        elif self.mps_available:
            return self.torch.mean(x, dim=axis)
    
    def max(self, x: Any, axis: Optional[int] = None) -> Any:
        """Maximum reduction."""
        if self.mlx_available:
            return self.mx.max(x, axis=axis)
        elif self.mps_available:
            if axis is None:
                return self.torch.max(x)
            else:
                return self.torch.max(x, dim=axis)[0]
    
    # Linear algebra
    def dot(self, a: Any, b: Any, out: Optional[Any] = None) -> Any:
        """Matrix multiplication."""
        if self.mlx_available:
            result = self.mx.matmul(a, b)
            if out is not None:
                out[:] = result
            return result
        elif self.mps_available:
            return self.torch.matmul(a, b, out=out)
    
    def norm(self, x: Any, ord: Optional[int] = None, axis: Optional[int] = None) -> Any:
        """Vector/matrix norm."""
        if self.mlx_available:
            # MLX doesn't have direct norm, compute manually
            if ord is None or ord == 2:
                if axis is None:
                    return self.mx.sqrt(self.mx.sum(x * x))
                else:
                    return self.mx.sqrt(self.mx.sum(x * x, axis=axis))
            else:
                raise NotImplementedError(f"Norm order {ord} not implemented for MLX")
        elif self.mps_available:
            return self.torch.linalg.norm(x, ord=ord, dim=axis)
    
    # FFT operations
    def fft(self, x: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """Fast Fourier Transform."""
        if self.mlx_available:
            return self.mx.fft.fft(x, n=n, axis=axis)
        elif self.mps_available:
            return self.torch.fft.fft(x, n=n, dim=axis)
    
    def ifft(self, x: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """Inverse Fast Fourier Transform."""
        if self.mlx_available:
            return self.mx.fft.ifft(x, n=n, axis=axis)
        elif self.mps_available:
            return self.torch.fft.ifft(x, n=n, dim=axis)
    
    def rfft(self, x: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """Real FFT."""
        if self.mlx_available:
            return self.mx.fft.rfft(x, n=n, axis=axis)
        elif self.mps_available:
            return self.torch.fft.rfft(x, n=n, dim=axis)
    
    # Specialized operations
    def phase_coupling(self, phases: Any, weights: Any, connections: Any) -> Any:
        """Compute phase coupling using Metal acceleration."""
        if self.mlx_available:
            # Vectorized computation using MLX
            phase_diff = phases[:, None] - phases[None, :]
            coupling_matrix = self.mx.sin(phase_diff)
            weighted_coupling = coupling_matrix * weights * connections
            return self.mx.sum(weighted_coupling, axis=1)
        elif self.mps_available:
            # Using PyTorch on MPS
            phase_diff = phases.unsqueeze(1) - phases.unsqueeze(0)
            coupling_matrix = self.torch.sin(phase_diff)
            weighted_coupling = coupling_matrix * weights * connections
            return self.torch.sum(weighted_coupling, dim=1)
    
    def oscillator_bank(self, frequencies: Any, phases: Any, amplitudes: Any, 
                       time_points: Any) -> Any:
        """Compute oscillator bank using Metal."""
        if self.mlx_available:
            # Reshape for broadcasting
            freqs = frequencies[:, None]
            phases_reshaped = phases[:, None]
            amps = amplitudes[:, None]
            times = time_points[None, :]
            
            # Compute all oscillators at once
            phase_matrix = 2 * self.mx.pi * freqs * times + phases_reshaped
            signals = amps * self.mx.sin(phase_matrix)
            return signals
        elif self.mps_available:
            # Similar implementation with PyTorch
            freqs = frequencies.unsqueeze(1)
            phases_reshaped = phases.unsqueeze(1)
            amps = amplitudes.unsqueeze(1)
            times = time_points.unsqueeze(0)
            
            phase_matrix = 2 * self.torch.pi * freqs * times + phases_reshaped
            signals = amps * self.torch.sin(phase_matrix)
            return signals
    
    def batch_hilbert(self, signals: Any) -> Any:
        """Batch Hilbert transform on Metal."""
        # For now, transfer to CPU for Hilbert transform
        # Future: implement custom Metal kernel
        np_signals = self.to_host(signals)
        from scipy.signal import hilbert
        
        if np_signals.ndim == 1:
            result = hilbert(np_signals)
        else:
            result = np.array([hilbert(sig) for sig in np_signals])
        
        return self.to_device(result)
    
    def synchronize(self) -> None:
        """Synchronize Metal operations."""
        if self.mlx_available:
            self.mx.synchronize()
        elif self.mps_available:
            self.torch.mps.synchronize()
    
    # Helper methods
    def _numpy_to_mlx_dtype(self, np_dtype: np.dtype):
        """Convert NumPy dtype to MLX dtype."""
        dtype_map = {
            np.float32: self.mx.float32,
            np.float64: self.mx.float32,  # MLX doesn't support float64
            np.int32: self.mx.int32,
            np.int64: self.mx.int64,
        }
        return dtype_map.get(np_dtype.type, self.mx.float32)
    
    def _numpy_to_torch_dtype(self, np_dtype: np.dtype):
        """Convert NumPy dtype to PyTorch dtype."""
        dtype_map = {
            np.float32: self.torch.float32,
            np.float64: self.torch.float64,
            np.int32: self.torch.int32,
            np.int64: self.torch.int64,
        }
        return dtype_map.get(np_dtype.type, self.torch.float32) 