"""
Backend manager for SynthNN performance optimization.

This module provides automatic backend selection and management,
choosing the best available compute backend for the current system.
"""

import numpy as np
from typing import Optional, Dict, Any, List
import warnings
import platform

from .backend import ComputeBackend, BackendType
from .cpu_backend import CPUBackend
from .cuda_backend import CUDABackend
from .metal_backend import MetalBackend


class BackendManager:
    """
    Manages compute backends and provides automatic selection.
    
    The manager can automatically detect and select the best available
    backend, or allow manual backend selection.
    """
    
    def __init__(self, preferred_backend: Optional[BackendType] = None,
                 device_id: int = 0):
        """
        Initialize the backend manager.
        
        Args:
            preferred_backend: Preferred backend type, or None for auto-select
            device_id: Device ID for GPU backends
        """
        self.preferred_backend = preferred_backend
        self.device_id = device_id
        self.available_backends = self._detect_backends()
        self.current_backend = None
        self._backend_cache = {}
        
    def _detect_backends(self) -> Dict[BackendType, bool]:
        """Detect which backends are available on the system."""
        backends = {}
        
        # CPU is always available
        backends[BackendType.CPU] = True
        
        # Check CUDA
        try:
            cuda_backend = CUDABackend(self.device_id)
            backends[BackendType.CUDA] = cuda_backend.is_available()
        except:
            backends[BackendType.CUDA] = False
            
        # Check Metal (only on macOS)
        if platform.system() == 'Darwin':
            try:
                metal_backend = MetalBackend(self.device_id)
                backends[BackendType.METAL] = metal_backend.is_available()
            except:
                backends[BackendType.METAL] = False
        else:
            backends[BackendType.METAL] = False
            
        return backends
    
    def get_backend(self, backend_type: Optional[BackendType] = None) -> ComputeBackend:
        """
        Get a compute backend instance.
        
        Args:
            backend_type: Specific backend type, or None to use auto-selection
            
        Returns:
            Initialized compute backend
        """
        if backend_type is None:
            backend_type = self._select_best_backend()
            
        # Check cache
        if backend_type in self._backend_cache:
            return self._backend_cache[backend_type]
            
        # Create new backend
        if backend_type == BackendType.CPU:
            backend = CPUBackend(self.device_id)
        elif backend_type == BackendType.CUDA:
            if not self.available_backends.get(BackendType.CUDA, False):
                raise RuntimeError("CUDA backend requested but not available")
            backend = CUDABackend(self.device_id)
        elif backend_type == BackendType.METAL:
            if not self.available_backends.get(BackendType.METAL, False):
                raise RuntimeError("Metal backend requested but not available")
            backend = MetalBackend(self.device_id)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
            
        # Initialize and cache
        backend.initialize()
        self._backend_cache[backend_type] = backend
        self.current_backend = backend
        
        return backend
    
    def _select_best_backend(self) -> BackendType:
        """Automatically select the best available backend."""
        # If user has preference and it's available, use it
        if (self.preferred_backend and 
            self.available_backends.get(self.preferred_backend, False)):
            return self.preferred_backend
            
        # Priority order: Metal (on Mac), CUDA, CPU
        if platform.system() == 'Darwin' and self.available_backends.get(BackendType.METAL, False):
            return BackendType.METAL
        elif self.available_backends.get(BackendType.CUDA, False):
            return BackendType.CUDA
        else:
            return BackendType.CPU
    
    def list_available_backends(self) -> List[BackendType]:
        """Get list of available backends."""
        return [backend for backend, available in self.available_backends.items() if available]

    @property
    def backend_type(self) -> BackendType:
        """Currently selected backend type."""
        if self.current_backend is not None:
            return self.current_backend.backend_type
        return self._select_best_backend()
    
    def get_device_info(self, backend_type: Optional[BackendType] = None) -> Dict[str, Any]:
        """Get device information for a specific backend."""
        backend = self.get_backend(backend_type)
        return backend.get_device_info()
    
    def benchmark_backends(self, operations: List[str] = None,
                          size: int = 1000000) -> Dict[BackendType, Dict[str, float]]:
        """
        Benchmark available backends on common operations.
        
        Args:
            operations: List of operations to benchmark, or None for default set
            size: Size of test arrays
            
        Returns:
            Dictionary mapping backends to operation timings
        """
        if operations is None:
            operations = ['add', 'multiply', 'sin', 'fft']
            
        results = {}
        
        for backend_type in self.list_available_backends():
            print(f"\nBenchmarking {backend_type.value} backend...")
            backend = self.get_backend(backend_type)
            
            backend_results = {}
            for op in operations:
                try:
                    time = backend.benchmark_operation(op, size)
                    backend_results[op] = time
                    print(f"  {op}: {time*1000:.3f} ms")
                except Exception as e:
                    warnings.warn(f"Failed to benchmark {op} on {backend_type.value}: {e}")
                    backend_results[op] = float('inf')
                    
            results[backend_type] = backend_results
            
        return results
    
    def auto_select_by_benchmark(self, operations: List[str] = None) -> BackendType:
        """
        Select backend based on benchmark results.
        
        Args:
            operations: Operations to consider for selection
            
        Returns:
            Best performing backend type
        """
        results = self.benchmark_backends(operations)
        
        # Calculate average time for each backend
        avg_times = {}
        for backend, timings in results.items():
            valid_times = [t for t in timings.values() if t != float('inf')]
            if valid_times:
                avg_times[backend] = np.mean(valid_times)
            else:
                avg_times[backend] = float('inf')
                
        # Select backend with lowest average time
        best_backend = min(avg_times.items(), key=lambda x: x[1])[0]
        
        print(f"\nBest backend based on benchmarks: {best_backend.value}")
        return best_backend


class AcceleratedResonantNetwork:
    """
    ResonantNetwork with automatic GPU/Metal acceleration.
    
    This is a drop-in replacement for ResonantNetwork that automatically
    uses the best available compute backend.
    """
    
    def __init__(self, name: str = "default", backend: Optional[BackendType] = None):
        """
        Initialize accelerated resonant network.
        
        Args:
            name: Network name
            backend: Preferred backend, or None for auto-selection
        """
        # Import here to avoid circular dependency
        from ..core.resonant_network import ResonantNetwork
        
        # Initialize base network
        self._base_network = ResonantNetwork(name)
        
        # Initialize backend
        self.backend_manager = BackendManager(preferred_backend=backend)
        self.backend = self.backend_manager.get_backend()
        
        # Device arrays for node data
        self._device_phases = None
        self._device_frequencies = None
        self._device_amplitudes = None
        self._device_weights = None
        self._device_connections = None
        
        print(f"AcceleratedResonantNetwork initialized with {type(self.backend).__name__}")
    
    def add_node(self, node):
        """Add node and allocate device memory if needed."""
        self._base_network.add_node(node)
        self._sync_to_device()
    
    def connect(self, source_id: str, target_id: str, weight: float = 1.0, delay: float = 0.0):
        """Create connection and update device arrays."""
        self._base_network.connect(source_id, target_id, weight, delay)
        self._sync_to_device()
    
    def _sync_to_device(self):
        """Synchronize node data to device memory."""
        nodes = self._base_network.nodes
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
                if (src_id, tgt_id) in self._base_network.connections:
                    conn = self._base_network.connections[(src_id, tgt_id)]
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
        
        # Update nodes
        node_ids = list(self._base_network.nodes.keys())
        for i, nid in enumerate(node_ids):
            self._base_network.nodes[nid].phase = float(phases[i])
            self._base_network.nodes[nid].amplitude = float(amplitudes[i])
    
    def step(self, dt: float, external_inputs: Optional[Dict[str, float]] = None):
        """
        Accelerated network step using compute backend.
        
        Args:
            dt: Time step
            external_inputs: External signals for specific nodes
        """
        if len(self._base_network.nodes) == 0:
            return
            
        # Apply external inputs if any
        if external_inputs:
            node_ids = list(self._base_network.nodes.keys())
            for node_id, signal in external_inputs.items():
                if node_id in node_ids:
                    idx = node_ids.index(node_id)
                    # Update amplitude on device
                    amp_host = self.backend.to_host(self._device_amplitudes)
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
            self.backend.to_device(np.array(self._base_network.coupling_strength * dt, dtype=np.float32))
        )
        
        # Natural frequency advance
        freq_advance = self.backend.multiply(
            self._device_frequencies,
            self.backend.to_device(np.array(2 * np.pi * dt, dtype=np.float32))
        )
        
        # Total phase change
        phase_change = self.backend.add(freq_advance, coupling_scaled)
        
        # Update phases with modulo 2π
        self._device_phases = self.backend.add(self._device_phases, phase_change)
        # Note: Modulo operation might need custom implementation for some backends
        
        # Apply damping to amplitudes
        damping_factor = 1 - self._base_network.global_damping * dt
        self._device_amplitudes = self.backend.multiply(
            self._device_amplitudes,
            self.backend.to_device(np.array(damping_factor, dtype=np.float32))
        )
        
        # Sync back to CPU network
        self._sync_from_device()
        
        # Update time
        self._base_network.time += dt
    
    def generate_signals(self, duration: float, sample_rate: float = 44100) -> np.ndarray:
        """
        Generate audio signals using accelerated oscillator bank.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Audio signal array
        """
        if len(self._base_network.nodes) == 0:
            return np.zeros(int(duration * sample_rate))
            
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
    
    # Delegate other methods to base network
    def __getattr__(self, name):
        """Delegate unknown attributes to base network."""
        return getattr(self._base_network, name) 