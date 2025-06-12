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
    signal: complex           # Complex wave value at this point
    velocity: np.ndarray      # Wave velocity vector [vx, vy, vz]
    damping: float            # Local damping factor
    
    @property
    def amplitude(self) -> float:
        return abs(self.signal)
        
    @property
    def phase(self) -> float:
        return np.angle(self.signal)


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
        super().__init__(node_id, frequency=frequency, phase=phase, amplitude=amplitude, damping=damping)
        self.position = np.array(position)
        self.radiation_pattern = radiation_pattern
        
    def radiate(self) -> complex:
        """
        Calculate the complex radiation from this node.
        
        Returns:
            A complex value representing the wave source.
        """
        # The radiation is simply the node's current signal.
        # Directional patterns would modify this complex value.
        # For now, we assume omnidirectional radiation.
        return self.signal


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
        
        # A single complex field for the wave state (amplitude and phase)
        self.signal_field = np.zeros(dimensions, dtype=np.complex128)
        
        # Previous states for the wave equation solver
        self.signal_prev = np.zeros(dimensions, dtype=np.complex128)
        
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
            self.signal_field[grid_pos] = node.signal
            
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
        Advance the field by one time step using the complex wave equation.
        
        Uses the 3D wave equation on a complex field `u`:
        ∂²u/∂t² = c²∇²u - γ∂u/∂t
        """
        if dt is None:
            dt = self.dt
            
        # Update node states and inject their signals into the field
        for node in self.nodes.values():
            # Nodes evolve on their own (internal frequency + damping)
            node.step(dt) 
            grid_pos = self._position_to_grid(node.position)
            
            if self._in_bounds(grid_pos):
                # Inject node's signal into the field, acting as a source term
                self.signal_field[grid_pos] += node.radiate() * dt
                
        # Compute Laplacian (∇²u) on the complex field
        laplacian = self._compute_laplacian(self.signal_field)
        
        # Store current state before update
        signal_current = self.signal_field.copy()

        # Wave equation with damping for a complex field
        c2dt2 = (self.wave_speed * dt) ** 2
        damping = self.medium['absorption'] * dt
        
        # Verlet integration for the wave equation
        new_signal = (
            (2 - damping) * self.signal_field 
            - (1 - damping) * self.signal_prev
            + c2dt2 * laplacian
        )
        
        # Apply boundary conditions
        self._apply_boundary_conditions(new_signal)
        
        # Update field states for the next iteration
        self.signal_prev = signal_current
        self.signal_field = new_signal
        
        # Increment time
        self.time += dt
        
    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute 3D Laplacian using finite differences on a complex field."""
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
                    elif dim == 2:
                        field[:, :, i] *= factor
                        field[:, :, -i-1] *= factor
                        
        elif self.boundary_condition == BoundaryCondition.RADIATING:
            # Simple radiating boundary (Sommerfeld condition)
            # Not fully implemented here, requires more complex stencil
            pass
        
    def measure_field_energy(self) -> float:
        """
        Calculate the total energy in the field.
        
        Energy is proportional to the squared amplitude of the signal.
        """
        # Energy is integral of |signal|^2 over the volume
        return 0.5 * np.sum(np.abs(self.signal_field)**2) * (self.resolution**3)
        
    def find_resonant_modes(self, frequency_range: Tuple[float, float],
                           num_modes: int = 10) -> List[Dict]:
        """
        Find resonant modes of the field using spectral analysis.
        
        Args:
            frequency_range: (min_freq, max_freq) to search
            num_modes: Number of modes to find
            
        Returns:
            List of dicts with 'frequency', 'amplitude_pattern', 'phase_pattern', 'q_factor'
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
            for _ in range(int(10 / (test_freq + 1e-6) / self.dt)):  # 10 periods
                self.step()
                
            # Measure response
            final_energy = self.measure_field_energy()
            
            if final_energy > initial_energy * 1.5:  # Resonance amplification
                mode = {
                    'frequency': test_freq,
                    'amplitude_pattern': np.abs(self.signal_field.copy()),
                    'phase_pattern': np.angle(self.signal_field.copy()),
                    'q_factor': final_energy / (initial_energy + 1e-9)
                }
                modes.append(mode)
                
            # Remove test node
            del self.nodes["test"]
            
            # Reset field
            self.signal_field.fill(0j)
            self.signal_prev.fill(0j)
            
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
        direction = direction / np.linalg.norm(direction)
        k = 2 * np.pi / wavelength  # Wave number
        
        # Create a grid of positions
        x = np.arange(self.dimensions[0]) * self.resolution
        y = np.arange(self.dimensions[1]) * self.resolution
        z = np.arange(self.dimensions[2]) * self.resolution
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        positions = np.stack([xx, yy, zz], axis=-1)
        
        # Projection of position onto direction vector
        projection = np.dot(positions, direction)
        
        # Create standing wave: A * cos(k*x)
        self.signal_field = np.cos(k * projection)
        
    def extract_slice(self, axis: str = 'z', index: Optional[int] = None) -> np.ndarray:
        """
        Extract a 2D slice of the complex field.
        
        Args:
            axis: 'x', 'y', or 'z'
            index: Slice index (defaults to center)
            
        Returns:
            2D numpy array of the complex field slice
        """
        if index is None:
            if axis == 'x': index = self.dimensions[0] // 2
            if axis == 'y': index = self.dimensions[1] // 2
            if axis == 'z': index = self.dimensions[2] // 2
            
        if axis == 'x':
            return self.signal_field[index, :, :]
        elif axis == 'y':
            return self.signal_field[:, index, :]
        else: # z
            return self.signal_field[:, :, index]
            
    def create_holographic_pattern(self, target_pattern: np.ndarray,
                                  recording_plane_z: int) -> None:
        """
        Create a holographic interference pattern that reconstructs target_pattern.
        
        Args:
            target_pattern: 2D pattern to reconstruct
            recording_plane_z: Z-index of the recording plane
        """
        target_fft = np.fft.fft2(target_pattern)
        
        # Back-propagate this field to the origin (z=0)
        # This is a simplified model (Fraunhofer diffraction)
        # A full implementation would use Rayleigh-Sommerfeld propagation.
        
        # For simplicity, we just set the field to the target pattern at the plane
        if recording_plane_z < self.dimensions[2]:
            self.signal_field[:, :, recording_plane_z] = target_pattern
        
    def get_field_statistics(self) -> Dict[str, float]:
        """Get statistics about the current state of the field."""
        mean_amplitude = np.mean(np.abs(self.signal_field))
        max_amplitude = np.max(np.abs(self.signal_field))
        total_energy = self.measure_field_energy()
        
        return {
            'mean_amplitude': mean_amplitude,
            'max_amplitude': max_amplitude,
            'total_energy': total_energy,
            'spatial_coherence': self._measure_spatial_coherence(),
            'entropy': self._calculate_field_entropy()
        }
        
    def _measure_spatial_coherence(self) -> float:
        """Measure the spatial coherence of the field."""
        # A simple measure: correlation of field with a shifted version
        shifted_field = np.roll(self.signal_field, 1, axis=0)
        
        # Normalized correlation
        corr = np.sum(self.signal_field * np.conj(shifted_field))
        norm = np.sqrt(np.sum(np.abs(self.signal_field)**2) * np.sum(np.abs(shifted_field)**2))
        
        return np.abs(corr / norm) if norm > 0 else 0.0
        
    def _calculate_field_entropy(self) -> float:
        """Calculate the spectral entropy of the field's amplitude distribution."""
        amplitudes = np.abs(self.signal_field).flatten()
        
        if np.sum(amplitudes) == 0:
            return 0.0
            
        # Create a probability distribution from amplitudes
        prob_dist = amplitudes / np.sum(amplitudes)
        prob_dist = prob_dist[prob_dist > 0]  # Avoid log(0)
        
        # Shannon entropy
        return -np.sum(prob_dist * np.log2(prob_dist)) 