"""
SynthNN Performance Module

This module provides hardware acceleration for SynthNN computations,
supporting CPU optimization, CUDA GPUs, and Apple Metal.
"""

from .backend import BackendType, ComputeBackend
from .backend_manager import BackendManager, AcceleratedResonantNetwork
from .cpu_backend import CPUBackend

# Optional imports (may not be available on all systems)
try:
    from .cuda_backend import CUDABackend
except ImportError:
    CUDABackend = None

try:
    from .metal_backend import MetalBackend
except ImportError:
    MetalBackend = None

__all__ = [
    'BackendType',
    'ComputeBackend', 
    'BackendManager',
    'AcceleratedResonantNetwork',
    'CPUBackend',
    'CUDABackend',
    'MetalBackend'
] 