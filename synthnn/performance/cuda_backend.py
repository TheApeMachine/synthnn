"""
CUDA backend implementation for SynthNN.

This backend uses CuPy or PyTorch CUDA for GPU acceleration on NVIDIA GPUs.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

from .backend import ComputeBackend


class CUDABackend(ComputeBackend):
    """CUDA backend for NVIDIA GPUs."""
    
    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self.cupy_available = False
        self.torch_cuda_available = False

    @property
    def backend_type(self) -> "BackendType":
        from .backend import BackendType
        return BackendType.CUDA
        
    def initialize(self) -> None:
        """Initialize CUDA backend."""
        # Try CuPy first (preferred for array operations)
        try:
            import cupy as cp
            self.cupy_available = True
            self.cp = cp
            self.cp.cuda.Device(self.device_id).use()
        except ImportError:
            self.cupy_available = False
            
        # Try PyTorch CUDA as fallback
        try:
            import torch
            if torch.cuda.is_available():
                self.torch_cuda_available = True
                self.torch = torch
                self.device = torch.device(f"cuda:{self.device_id}")
        except ImportError:
            self.torch_cuda_available = False
        
        if not self.cupy_available and not self.torch_cuda_available:
            raise RuntimeError("No CUDA backend available. Install cupy or torch with CUDA support.")
        
        # Get device properties
        if self.cupy_available:
            device = self.cp.cuda.Device(self.device_id)
            props = device.attributes
            memory_info = device.mem_info
        elif self.torch_cuda_available:
            props = self.torch.cuda.get_device_properties(self.device_id)
            memory_info = (
                self.torch.cuda.mem_get_info(self.device_id)[0],
                self.torch.cuda.mem_get_info(self.device_id)[1]
            )
        
        self._capabilities = {
            'backend_type': 'cuda',
            'cupy_available': self.cupy_available,
            'torch_cuda_available': self.torch_cuda_available,
            'compute_capability': getattr(props, 'major', 0) * 10 + getattr(props, 'minor', 0) if self.torch_cuda_available else None,
            'max_memory': memory_info[1] if self.cupy_available else props.total_memory,
            'free_memory': memory_info[0] if self.cupy_available else self.torch.cuda.mem_get_info(self.device_id)[0],
            'preferred_dtype': np.float32
        }
        self.is_initialized = True
    
    def is_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import cupy as cp
            return cp.cuda.runtime.getDeviceCount() > 0
        except ImportError:
            pass
            
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get CUDA device information."""
        if self.cupy_available:
            device = self.cp.cuda.Device(self.device_id)
            return {
                'device_name': device.name.decode() if hasattr(device, 'name') else 'Unknown',
                'device_type': 'CUDA GPU',
                'compute_capability': device.compute_capability,
                'total_memory': device.mem_info[1],
                'multiprocessor_count': device.attributes.get('MultiProcessorCount', 0)
            }
        elif self.torch_cuda_available:
            props = self.torch.cuda.get_device_properties(self.device_id)
            return {
                'device_name': props.name,
                'device_type': 'CUDA GPU',
                'compute_capability': f"{props.major}.{props.minor}",
                'total_memory': props.total_memory,
                'multiprocessor_count': props.multi_processor_count
            }
    
    # Memory management
    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> Any:
        """Allocate memory on CUDA device."""
        if self.cupy_available:
            return self.cp.zeros(shape, dtype=dtype)
        elif self.torch_cuda_available:
            return self.torch.zeros(shape, dtype=self._numpy_to_torch_dtype(dtype), device=self.device)
    
    def free(self, array: Any) -> None:
        """Free CUDA memory."""
        del array
        if self.cupy_available:
            self.cp.get_default_memory_pool().free_all_blocks()
        elif self.torch_cuda_available:
            self.torch.cuda.empty_cache()
    
    def to_device(self, host_array: np.ndarray) -> Any:
        """Transfer array to CUDA device."""
        if self.cupy_available:
            return self.cp.asarray(host_array)
        elif self.torch_cuda_available:
            return self.torch.from_numpy(host_array).to(self.device)
    
    def to_host(self, device_array: Any) -> np.ndarray:
        """Transfer array from CUDA device to host."""
        if self.cupy_available:
            return self.cp.asnumpy(device_array)
        elif self.torch_cuda_available:
            return device_array.cpu().numpy()
    
    # Basic operations
    def add(self, a: Any, b: Any, out: Optional[Any] = None) -> Any:
        """Element-wise addition."""
        if self.cupy_available:
            return self.cp.add(a, b, out=out)
        elif self.torch_cuda_available:
            return self.torch.add(a, b, out=out)
    
    def multiply(self, a: Any, b: Any, out: Optional[Any] = None) -> Any:
        """Element-wise multiplication."""
        if self.cupy_available:
            return self.cp.multiply(a, b, out=out)
        elif self.torch_cuda_available:
            return self.torch.multiply(a, b, out=out)
    
    def sin(self, x: Any, out: Optional[Any] = None) -> Any:
        """Element-wise sine."""
        if self.cupy_available:
            return self.cp.sin(x, out=out)
        elif self.torch_cuda_available:
            return self.torch.sin(x, out=out)
    
    def cos(self, x: Any, out: Optional[Any] = None) -> Any:
        """Element-wise cosine."""
        if self.cupy_available:
            return self.cp.cos(x, out=out)
        elif self.torch_cuda_available:
            return self.torch.cos(x, out=out)
    
    def exp(self, x: Any, out: Optional[Any] = None) -> Any:
        """Element-wise exponential."""
        if self.cupy_available:
            return self.cp.exp(x, out=out)
        elif self.torch_cuda_available:
            return self.torch.exp(x, out=out)
    
    # Reductions
    def sum(self, x: Any, axis: Optional[int] = None) -> Any:
        """Sum reduction."""
        if self.cupy_available:
            return self.cp.sum(x, axis=axis)
        elif self.torch_cuda_available:
            return self.torch.sum(x, dim=axis)
    
    def mean(self, x: Any, axis: Optional[int] = None) -> Any:
        """Mean reduction."""
        if self.cupy_available:
            return self.cp.mean(x, axis=axis)
        elif self.torch_cuda_available:
            return self.torch.mean(x, dim=axis)
    
    def max(self, x: Any, axis: Optional[int] = None) -> Any:
        """Maximum reduction."""
        if self.cupy_available:
            return self.cp.max(x, axis=axis)
        elif self.torch_cuda_available:
            if axis is None:
                return self.torch.max(x)
            else:
                return self.torch.max(x, dim=axis)[0]
    
    # Linear algebra
    def dot(self, a: Any, b: Any, out: Optional[Any] = None) -> Any:
        """Matrix multiplication."""
        if self.cupy_available:
            return self.cp.dot(a, b, out=out)
        elif self.torch_cuda_available:
            return self.torch.matmul(a, b, out=out)
    
    def norm(self, x: Any, ord: Optional[int] = None, axis: Optional[int] = None) -> Any:
        """Vector/matrix norm."""
        if self.cupy_available:
            return self.cp.linalg.norm(x, ord=ord, axis=axis)
        elif self.torch_cuda_available:
            return self.torch.linalg.norm(x, ord=ord, dim=axis)
    
    # FFT operations
    def fft(self, x: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """Fast Fourier Transform using cuFFT."""
        if self.cupy_available:
            return self.cp.fft.fft(x, n=n, axis=axis)
        elif self.torch_cuda_available:
            return self.torch.fft.fft(x, n=n, dim=axis)
    
    def ifft(self, x: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """Inverse Fast Fourier Transform."""
        if self.cupy_available:
            return self.cp.fft.ifft(x, n=n, axis=axis)
        elif self.torch_cuda_available:
            return self.torch.fft.ifft(x, n=n, dim=axis)
    
    def rfft(self, x: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """Real FFT."""
        if self.cupy_available:
            return self.cp.fft.rfft(x, n=n, axis=axis)
        elif self.torch_cuda_available:
            return self.torch.fft.rfft(x, n=n, dim=axis)
    
    # Specialized operations with custom CUDA kernels
    def phase_coupling(self, phases: Any, weights: Any, connections: Any) -> Any:
        """
        Compute phase coupling using CUDA acceleration.
        
        This could be optimized with custom CUDA kernels for better performance.
        """
        if self.cupy_available:
            # Custom CUDA kernel for phase coupling
            phase_coupling_kernel = self.cp.RawKernel(r'''
            extern "C" __global__
            void phase_coupling_kernel(const float* phases, const float* weights, 
                                     const float* connections, float* output,
                                     int n_nodes) {
                int tid = blockDim.x * blockIdx.x + threadIdx.x;
                if (tid < n_nodes) {
                    float coupling = 0.0f;
                    for (int j = 0; j < n_nodes; j++) {
                        float phase_diff = phases[j] - phases[tid];
                        coupling += weights[j * n_nodes + tid] * 
                                  connections[j * n_nodes + tid] * sinf(phase_diff);
                    }
                    output[tid] = coupling;
                }
            }
            ''', 'phase_coupling_kernel')
            
            n_nodes = phases.shape[0]
            output = self.cp.zeros(n_nodes, dtype=self.cp.float32)
            threads_per_block = 256
            blocks = (n_nodes + threads_per_block - 1) // threads_per_block
            
            phase_coupling_kernel((blocks,), (threads_per_block,), 
                                (phases, weights, connections, output, n_nodes))
            return output
        elif self.torch_cuda_available:
            # PyTorch implementation
            phase_diff = phases.unsqueeze(1) - phases.unsqueeze(0)
            coupling_matrix = self.torch.sin(phase_diff)
            weighted_coupling = coupling_matrix * weights * connections
            return self.torch.sum(weighted_coupling, dim=1)
    
    def oscillator_bank(self, frequencies: Any, phases: Any, amplitudes: Any, 
                       time_points: Any) -> Any:
        """Compute oscillator bank using CUDA."""
        if self.cupy_available:
            # Efficient broadcasting with CuPy
            freqs = frequencies[:, self.cp.newaxis]
            phases_reshaped = phases[:, self.cp.newaxis]
            amps = amplitudes[:, self.cp.newaxis]
            times = time_points[self.cp.newaxis, :]
            
            phase_matrix = 2 * self.cp.pi * freqs * times + phases_reshaped
            signals = amps * self.cp.sin(phase_matrix)
            return signals
        elif self.torch_cuda_available:
            # PyTorch implementation
            freqs = frequencies.unsqueeze(1)
            phases_reshaped = phases.unsqueeze(1)
            amps = amplitudes.unsqueeze(1)
            times = time_points.unsqueeze(0)
            
            phase_matrix = 2 * self.torch.pi * freqs * times + phases_reshaped
            signals = amps * self.torch.sin(phase_matrix)
            return signals
    
    def batch_hilbert(self, signals: Any) -> Any:
        """Batch Hilbert transform using CUDA-accelerated FFT."""
        if self.cupy_available:
            # Implement Hilbert transform using FFT
            n = signals.shape[-1]
            fft_result = self.cp.fft.fft(signals, axis=-1)
            
            # Create Hilbert filter
            h = self.cp.zeros(n)
            if n % 2 == 0:
                h[0] = h[n//2] = 1
                h[1:n//2] = 2
            else:
                h[0] = 1
                h[1:(n+1)//2] = 2
            
            # Apply filter and inverse FFT
            if signals.ndim == 2:
                h = h[self.cp.newaxis, :]
            
            analytic = self.cp.fft.ifft(fft_result * h, axis=-1)
            return analytic
        elif self.torch_cuda_available:
            # Similar implementation with PyTorch
            n = signals.shape[-1]
            fft_result = self.torch.fft.fft(signals, dim=-1)
            
            h = self.torch.zeros(n, device=self.device)
            if n % 2 == 0:
                h[0] = h[n//2] = 1
                h[1:n//2] = 2
            else:
                h[0] = 1
                h[1:(n+1)//2] = 2
            
            if signals.ndim == 2:
                h = h.unsqueeze(0)
            
            analytic = self.torch.fft.ifft(fft_result * h, dim=-1)
            return analytic
    
    def synchronize(self) -> None:
        """Synchronize CUDA operations."""
        if self.cupy_available:
            self.cp.cuda.Stream.null.synchronize()
        elif self.torch_cuda_available:
            self.torch.cuda.synchronize(self.device)
    
    def _numpy_to_torch_dtype(self, np_dtype: np.dtype):
        """Convert NumPy dtype to PyTorch dtype."""
        dtype_map = {
            np.float32: self.torch.float32,
            np.float64: self.torch.float64,
            np.int32: self.torch.int32,
            np.int64: self.torch.int64,
        }
        return dtype_map.get(np_dtype.type, self.torch.float32) 