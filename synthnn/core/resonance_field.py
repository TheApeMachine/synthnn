"""
4D Resonance Fields for SynthNN

Extends resonant networks into spatial dimensions, enabling wave propagation,
interference patterns, and emergent spatial dynamics. The "4D" represents
3D space + time, creating a unified field where information travels as waves.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import scipy.ndimage as ndimage
from scipy.spatial import distance_matrix

from .resonant_node import ResonantNode
from .resonant_network import ResonantNetwork
from .signal_processor import SignalProcessor


class BoundaryCondition(Enum):
    """Types of boundary conditions for the field."""
    ABSORBING = "absorbing"      # Waves are absorbed at boundaries
    REFLECTING = "reflecting"    # Waves reflect off boundaries  
    PERIODIC = "periodic"        # Waves wrap around (toroidal topology)
    RADIATING = "radiating"      # Waves radiate outward (open boundary)


@dataclass
class FieldPoint:
    """Represents a point in the resonance field."""
    position: np.ndarray      # 3D position [x, y, z]
    amplitude: float          # Wave amplitude at this point
    phase: float             # Wave phase at this point
    frequency: float         # Local oscillation frequency
    velocity: np.ndarray     # Wave velocity vector [vx, vy, vz]
    damping: float          # Local damping factor


class SpatialResonantNode(ResonantNode):
    """
    Extended ResonantNode with spatial properties.
    """
    
    def __init__(self, node_id: str, position: np.ndarray,
                 frequency: float = 1.0, phase: float = 0.0,
                 amplitude: float = 1.0, damping: float = 0.1,
                 radiation_pattern: str = "omnidirectional"):
        """
        Initialize a spatial resonant node.
        
        Args:
            node_id: Unique identifier
            position: 3D position [x, y, z]
            frequency: Natural frequency
            phase: Initial phase
            amplitude: Initial amplitude
            damping: Damping factor
            radiation_pattern: How the node radiates ("omnidirectional", "dipole", "quadrupole")
        """
        super().__init__(node_id, frequency, phase, amplitude, damping)
        self.position = np.array(position)
        self.radiation_pattern = radiation_pattern
        self.wave_sources = []  # Track waves emanating from this node
        
    def radiate(self, time: float) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate radiation from this node based on its pattern.
        
        Returns:
            Dictionary with 'amplitude', 'phase', and 'direction' information
        """
        base_signal = self.oscillate(time)
        
        if self.radiation_pattern == "omnidirectional":
            # Radiates equally in all directions
            return {
                'amplitude': base_signal,
                'phase': self.phase,
                'direction': None  # No preferred direction
            }
        elif self.radiation_pattern == "dipole":
            # Radiates along a preferred axis (like a speaker)
            dipole_axis = np.array([0, 0, 1])  # Default z-axis
            return {
                'amplitude': base_signal,
                'phase': self.phase,
                'direction': dipole_axis
            }
        elif self.radiation_pattern == "quadrupole":
            # More complex radiation pattern
            return {
                'amplitude': base_signal,
                'phase': self.phase,
                'direction': 'quadrupole'  # Special handling needed
            }
        

class ResonanceField4D:
    """
    A 4D resonance field where waves propagate through 3D space over time.
    """
    
    def __init__(self, dimensions: Tuple[int, int, int],
                 resolution: float = 1.0,
                 wave_speed: float = 343.0,  # Speed of sound in air (m/s)
                 boundary_condition: BoundaryCondition = BoundaryCondition.ABSORBING,
                 medium_properties: Optional[Dict] = None):
        """
        Initialize a 4D resonance field.
        
        Args:
            dimensions: Field size in grid points (nx, ny, nz)
            resolution: Spatial resolution (meters per grid point)
            wave_speed: Wave propagation speed
            boundary_condition: How waves behave at boundaries
            medium_properties: Optional dict with 'density', 'impedance', etc.
        """
        self.dimensions = dimensions
        self.resolution = resolution
        self.wave_speed = wave_speed
        self.boundary_condition = boundary_condition
        
        # Initialize field arrays
        self.amplitude_field = np.zeros(dimensions)
        self.phase_field = np.zeros(dimensions)
        self.velocity_field = np.zeros(dimensions + (3,))  # 3D velocity at each point
        
        # Previous state for wave equation
        self.amplitude_prev = np.zeros(dimensions)
        self.amplitude_prev2 = np.zeros(dimensions)
        
        # Medium properties
        self.medium = medium_properties or {
            'density': np.ones(dimensions),
            'impedance': np.ones(dimensions),
            'absorption': np.zeros(dimensions)
        }
        
        # Spatial nodes in the field
        self.nodes: Dict[str, SpatialResonantNode] = {}
        
        # Time tracking
        self.time = 0.0
        self.dt = resolution / (wave_speed * np.sqrt(3))  # CFL condition
        
        # Signal processor for analysis
        self.signal_processor = SignalProcessor()
        
    def add_spatial_node(self, node: SpatialResonantNode) -> None:
        """Add a spatial node to the field."""
        self.nodes[node.node_id] = node
        
        # Convert position to grid indices
        grid_pos = self._position_to_grid(node.position)
        
        # Initialize field at node position
        if self._in_bounds(grid_pos):
            self.amplitude_field[grid_pos] = node.amplitude
            self.phase_field[grid_pos] = node.phase
            
    def _position_to_grid(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Convert continuous position to grid indices."""
        grid_pos = (position / self.resolution).astype(int)
        return tuple(grid_pos)
        
    def _grid_to_position(self, grid_pos: Tuple[int, int, int]) -> np.ndarray:
        """Convert grid indices to continuous position."""
        return np.array(grid_pos) * self.resolution
        
    def _in_bounds(self, grid_pos: Tuple[int, int, int]) -> bool:
        """Check if grid position is within bounds."""
        return all(0 <= grid_pos[i] < self.dimensions[i] for i in range(3))
        
    def step(self, dt: Optional[float] = None) -> None:
        """
        Advance the field by one time step using wave equation.
        
        Uses the 3D wave equation:
        ∂²u/∂t² = c²∇²u - γ∂u/∂t
        
        where u is amplitude, c is wave speed, and γ is damping.
        """
        if dt is None:
            dt = self.dt
            
        # Update node states and inject energy into field
        for node in self.nodes.values():
            node.update_phase(dt)
            grid_pos = self._position_to_grid(node.position)
            
            if self._in_bounds(grid_pos):
                # Inject node's signal into field
                radiation = node.radiate(self.time)
                self.amplitude_field[grid_pos] += radiation['amplitude'] * dt
                
        # Compute Laplacian (∇²u) using finite differences
        laplacian = self._compute_laplacian(self.amplitude_field)
        
        # Wave equation with damping
        # u(t+dt) = 2u(t) - u(t-dt) + (c²dt²)∇²u - γdt(u(t) - u(t-dt))
        c2dt2 = (self.wave_speed * dt) ** 2
        damping = self.medium['absorption'] * dt
        
        new_amplitude = (
            2 * self.amplitude_field 
            - self.amplitude_prev2
            + c2dt2 * laplacian
            - damping * (self.amplitude_field - self.amplitude_prev2)
        )
        
        # Apply boundary conditions
        self._apply_boundary_conditions(new_amplitude)
        
        # Update field states
        self.amplitude_prev2 = self.amplitude_prev.copy()
        self.amplitude_prev = self.amplitude_field.copy()
        self.amplitude_field = new_amplitude
        
        # Update phase field based on local frequency gradients
        self._update_phase_field(dt)
        
        # Increment time
        self.time += dt
        
    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute 3D Laplacian using finite differences."""
        if self.boundary_condition == BoundaryCondition.PERIODIC:
            mode = 'wrap'
        elif self.boundary_condition == BoundaryCondition.REFLECTING:
            mode = 'reflect'
        else:
            mode = 'constant'
            
        # Use scipy's laplace function for efficiency
        return ndimage.laplace(field, mode=mode)
        
    def _apply_boundary_conditions(self, field: np.ndarray) -> None:
        """Apply boundary conditions to the field."""
        if self.boundary_condition == BoundaryCondition.ABSORBING:
            # Gradually reduce amplitude near boundaries
            for dim in range(3):
                # Create absorption zones at boundaries
                absorption_width = 5
                for i in range(absorption_width):
                    factor = i / absorption_width
                    if dim == 0:
                        field[i, :, :] *= factor
                        field[-i-1, :, :] *= factor
                    elif dim == 1:
                        field[:, i, :] *= factor
                        field[:, -i-1, :] *= factor
                    else:
                        field[:, :, i] *= factor
                        field[:, :, -i-1] *= factor
                        
        elif self.boundary_condition == BoundaryCondition.RADIATING:
            # Sommerfeld radiation condition
            # Waves should propagate outward without reflection
            # This is a simplified implementation
            boundary_damping = 0.9
            field[0, :, :] *= boundary_damping
            field[-1, :, :] *= boundary_damping
            field[:, 0, :] *= boundary_damping
            field[:, -1, :] *= boundary_damping
            field[:, :, 0] *= boundary_damping
            field[:, :, -1] *= boundary_damping
            
    def _update_phase_field(self, dt: float) -> None:
        """Update phase field based on local wave propagation."""
        # Simplified phase update based on local frequency
        # In reality, this would involve solving the eikonal equation
        freq_field = self._estimate_local_frequency()
        self.phase_field += 2 * np.pi * freq_field * dt
        self.phase_field = self.phase_field % (2 * np.pi)
        
    def _estimate_local_frequency(self) -> np.ndarray:
        """Estimate local frequency from amplitude changes."""
        # Simple estimation based on zero-crossings
        # More sophisticated methods could use Hilbert transform
        time_derivative = (self.amplitude_field - self.amplitude_prev) / self.dt
        
        # Avoid division by zero
        safe_amplitude = np.maximum(np.abs(self.amplitude_field), 1e-10)
        
        # Rough frequency estimate
        freq_estimate = np.abs(time_derivative) / (2 * np.pi * safe_amplitude)
        
        # Smooth the estimate
        freq_estimate = ndimage.gaussian_filter(freq_estimate, sigma=1.0)
        
        return np.clip(freq_estimate, 0, 1000)  # Reasonable frequency range
        
    def create_resonant_cavity(self, center: np.ndarray, 
                             dimensions: np.ndarray,
                             impedance_ratio: float = 10.0) -> None:
        """
        Create a resonant cavity with different medium properties.
        
        Args:
            center: Center position of cavity
            dimensions: Size of cavity [dx, dy, dz]
            impedance_ratio: Impedance inside vs outside cavity
        """
        grid_center = self._position_to_grid(center)
        grid_dims = (dimensions / self.resolution).astype(int)
        
        # Create cavity boundaries with higher impedance
        for i in range(max(0, grid_center[0] - grid_dims[0]//2),
                      min(self.dimensions[0], grid_center[0] + grid_dims[0]//2)):
            for j in range(max(0, grid_center[1] - grid_dims[1]//2),
                          min(self.dimensions[1], grid_center[1] + grid_dims[1]//2)):
                for k in range(max(0, grid_center[2] - grid_dims[2]//2),
                              min(self.dimensions[2], grid_center[2] + grid_dims[2]//2)):
                    # Check if on boundary of cavity
                    on_boundary = (
                        i == grid_center[0] - grid_dims[0]//2 or
                        i == grid_center[0] + grid_dims[0]//2 - 1 or
                        j == grid_center[1] - grid_dims[1]//2 or
                        j == grid_center[1] + grid_dims[1]//2 - 1 or
                        k == grid_center[2] - grid_dims[2]//2 or
                        k == grid_center[2] + grid_dims[2]//2 - 1
                    )
                    
                    if on_boundary:
                        self.medium['impedance'][i, j, k] = impedance_ratio
                        self.medium['absorption'][i, j, k] = 0.1  # Some absorption
                        
    def measure_field_energy(self) -> float:
        """Calculate total energy in the field."""
        # Kinetic energy: (1/2) * density * (du/dt)²
        time_derivative = (self.amplitude_field - self.amplitude_prev) / self.dt
        kinetic = 0.5 * np.sum(self.medium['density'] * time_derivative**2)
        
        # Potential energy from spatial gradients
        grad_x = np.gradient(self.amplitude_field, axis=0)
        grad_y = np.gradient(self.amplitude_field, axis=1)
        grad_z = np.gradient(self.amplitude_field, axis=2)
        
        # Sum of squared gradients
        grad_squared = grad_x**2 + grad_y**2 + grad_z**2
        potential = 0.5 * self.wave_speed**2 * np.sum(grad_squared)
        
        # Return as Python float
        total_energy = kinetic + potential
        return float(total_energy.item() if hasattr(total_energy, 'item') else total_energy)
        
    def find_resonant_modes(self, frequency_range: Tuple[float, float],
                           num_modes: int = 10) -> List[Dict]:
        """
        Find resonant modes of the field using spectral analysis.
        
        Args:
            frequency_range: (min_freq, max_freq) to search
            num_modes: Number of modes to find
            
        Returns:
            List of dicts with 'frequency', 'amplitude_pattern', 'q_factor'
        """
        modes = []
        
        # Test different frequencies
        test_freqs = np.linspace(frequency_range[0], frequency_range[1], 100)
        
        for test_freq in test_freqs:
            # Create test excitation
            test_node = SpatialResonantNode(
                "test",
                position=np.array(self.dimensions) * self.resolution / 2,  # Center
                frequency=test_freq,
                amplitude=1.0
            )
            
            # Add temporarily
            self.add_spatial_node(test_node)
            
            # Let field evolve
            initial_energy = self.measure_field_energy()
            for _ in range(int(10 / test_freq / self.dt)):  # 10 periods
                self.step()
                
            # Measure response
            final_energy = self.measure_field_energy()
            
            if final_energy > initial_energy * 2:  # Resonance amplification
                mode = {
                    'frequency': test_freq,
                    'amplitude_pattern': self.amplitude_field.copy(),
                    'q_factor': final_energy / initial_energy
                }
                modes.append(mode)
                
            # Remove test node
            del self.nodes["test"]
            
            # Reset field
            self.amplitude_field.fill(0)
            self.amplitude_prev.fill(0)
            self.amplitude_prev2.fill(0)
            
        # Sort by Q factor and return top modes
        modes.sort(key=lambda m: m['q_factor'], reverse=True)
        return modes[:num_modes]
        
    def create_standing_wave_pattern(self, wavelength: float,
                                   direction: np.ndarray) -> None:
        """
        Initialize field with a standing wave pattern.
        
        Args:
            wavelength: Wavelength of the standing wave
            direction: Direction vector [dx, dy, dz]
        """
        direction = direction / np.linalg.norm(direction)  # Normalize
        
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                for k in range(self.dimensions[2]):
                    position = self._grid_to_position((i, j, k))
                    
                    # Project position onto direction
                    distance_along = np.dot(position, direction)
                    
                    # Create standing wave
                    self.amplitude_field[i, j, k] = np.sin(
                        2 * np.pi * distance_along / wavelength
                    )
                    
    def extract_slice(self, axis: str = 'z', index: Optional[int] = None) -> np.ndarray:
        """
        Extract a 2D slice of the field for visualization.
        
        Args:
            axis: Which axis to slice along ('x', 'y', or 'z')
            index: Index along the axis (default: middle)
            
        Returns:
            2D array of field values
        """
        if index is None:
            # Use middle slice
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            axis_idx = axis_map[axis]
            index = self.dimensions[axis_idx] // 2
            
        if axis == 'x':
            return self.amplitude_field[index, :, :]
        elif axis == 'y':
            return self.amplitude_field[:, index, :]
        else:  # z
            return self.amplitude_field[:, :, index]
            
    def create_holographic_pattern(self, target_pattern: np.ndarray,
                                  recording_plane_z: int) -> None:
        """
        Create a holographic interference pattern that reconstructs target_pattern.
        
        Args:
            target_pattern: 2D pattern to reconstruct
            recording_plane_z: Z-index of the recording plane
        """
        # This is a simplified holography simulation
        # Real holography would involve more complex calculations
        
        # Reference wave (plane wave)
        ref_wavelength = 10 * self.resolution
        
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                # Object wave from target pattern
                if i < target_pattern.shape[0] and j < target_pattern.shape[1]:
                    object_amplitude = target_pattern[i, j]
                else:
                    object_amplitude = 0
                    
                # Reference wave
                ref_phase = 2 * np.pi * i / ref_wavelength
                ref_amplitude = 1.0
                
                # Interference pattern
                interference = object_amplitude + ref_amplitude * np.cos(ref_phase)
                
                self.amplitude_field[i, j, recording_plane_z] = interference
                
    def get_field_statistics(self) -> Dict[str, float]:
        """Get various statistics about the current field state."""
        return {
            'total_energy': self.measure_field_energy(),
            'max_amplitude': np.max(np.abs(self.amplitude_field)),
            'mean_amplitude': np.mean(np.abs(self.amplitude_field)),
            'coherence': self._measure_spatial_coherence(),
            'entropy': self._calculate_field_entropy()
        }
        
    def _measure_spatial_coherence(self) -> float:
        """Measure spatial coherence of the field."""
        # Use spatial autocorrelation as a measure of coherence
        field_normalized = self.amplitude_field / (np.max(np.abs(self.amplitude_field)) + 1e-10)
        
        # Compute autocorrelation for small shifts
        coherence = 0
        for shift in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            shifted = np.roll(field_normalized, shift, axis=(0, 1, 2))
            coherence += np.mean(field_normalized * shifted)
            
        return coherence / 3
        
    def _calculate_field_entropy(self) -> float:
        """Calculate entropy of the amplitude distribution."""
        # Discretize amplitudes
        hist, _ = np.histogram(self.amplitude_field.flatten(), bins=50)
        
        # Normalize to probability distribution
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]  # Remove zeros
        
        # Shannon entropy
        entropy = -np.sum(prob * np.log2(prob))
        
        return entropy 