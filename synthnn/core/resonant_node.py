"""
Core ResonantNode implementation for the SynthNN framework.

This module provides the fundamental building block for all resonant networks
in the system, supporting various signal types and modulation schemes.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field, InitVar


@dataclass
class ResonantNode:
    """
    A single resonant node that oscillates at a specific frequency with phase and amplitude.
    
    This is the fundamental unit of computation in the SynthNN framework,
    representing a wave-based neuron that can resonate with input signals
    and influence connected nodes through phase coupling.
    
    The node's state is stored as a single complex number (signal) representing
    both amplitude and phase, which simplifies physics-based computations.
    
    Attributes:
        node_id: Unique identifier for the node
        natural_freq: Natural oscillation frequency (Hz)
        signal: Complex number representing the node's state (amplitude and phase)
        damping: Energy dissipation factor (0 = no damping, 1 = critical damping)
        metadata: Optional dictionary for storing additional node properties
    """
    
    node_id: str
    damping: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Internal state is complex, but we offer a legacy-compatible __init__
    signal: complex = field(init=False, repr=False)
    natural_freq: float = field(init=False, repr=False)

    # Legacy __init__ args, converted to `signal` in __post_init__
    frequency: InitVar[float] = 1.0
    phase: InitVar[float] = 0.0
    amplitude: InitVar[float] = 1.0
    
    def __post_init__(self, frequency: float, phase: float, amplitude: float):
        """Validate and normalize initial parameters, creating the complex signal."""
        self.natural_freq = frequency
        self.damping = np.clip(self.damping, 0.0, 1.0)
        
        # Create the complex signal from legacy amplitude and phase
        # This is the single source of truth for the node's state
        self.signal = max(0.0, amplitude) * np.exp(1j * (phase % (2 * np.pi)))

    # --- Convenience views for legacy compatibility ---
    
    @property
    def amplitude(self) -> float:
        """Returns the absolute magnitude of the signal."""
        return abs(self.signal)

    @amplitude.setter
    def amplitude(self, value: float):
        """Sets the signal's magnitude, preserving its phase."""
        value = max(0.0, value)
        if abs(self.signal) > 1e-9:
            self.signal = self.signal / abs(self.signal) * value
        else:
            self.signal = complex(value, 0)

    @property
    def phase(self) -> float:
        """Returns the phase angle of the signal in radians."""
        return np.angle(self.signal)

    @phase.setter
    def phase(self, value: float):
        """Sets the signal's phase, preserving its magnitude."""
        self.signal = abs(self.signal) * np.exp(1j * value)

    @property
    def frequency(self) -> float:
        """Legacy access to natural_freq."""
        return self.natural_freq
        
    @frequency.setter
    def frequency(self, value: float):
        """Legacy access to set natural_freq."""
        self.natural_freq = value
        
    # --- Core wave-based evolution ---

    def step(self, dt: float, coupling: complex = 0j, damping_override: Optional[float] = None) -> None:
        """
        Evolve the node's state over a time step `dt` using complex algebra.

        Args:
            dt: The time step for the evolution.
            coupling: A complex value representing external driving forces.
            damping_override: Optional override for the damping factor.
        """
        # 1. Free evolution (rotation in the complex plane)
        self.signal *= np.exp(1j * 2 * np.pi * self.natural_freq * dt)
        
        # 2. Apply external coupling (vector addition)
        self.signal += coupling
        
        # 3. Apply energy loss (scaling the vector)
        damping_factor = self.damping if damping_override is None else damping_override
        self.signal *= (1.0 - damping_factor * dt)

    def oscillate(self, time: float) -> float:
        """
        Calculate the node's REAL signal component at a given time.
        
        This is a projection of the complex state onto the real axis.
        
        Args:
            time: Time point to evaluate the oscillation.
            
        Returns:
            Real-valued signal at the specified time.
        """
        # This method is now conceptually simpler: it projects the evolved state.
        # For a simple projection, we can evolve the current state forward.
        evolved_signal = self.signal * np.exp(1j * 2 * np.pi * self.natural_freq * time)
        return evolved_signal.real
    
    def update_phase(self, dt: float, phase_coupling: float = 0.0):
        """
        [DEPRECATED] Update the node's state. Use step() for complex evolution.
        
        This method is kept for backward compatibility but delegates to step().
        The `phase_coupling` is treated as a purely real-valued driving force.
        
        Args:
            dt: Time step.
            phase_coupling: External phase influence (interpreted as a real coupling).
        """
        # For compatibility, we can treat phase_coupling as a real-valued force
        self.step(dt, coupling=complex(phase_coupling, 0))
    
    def apply_stimulus(self, stimulus: float, sensitivity: float = 1.0) -> None:
        """
        Apply external stimulus, now interpreted as a complex-plane operation.
        
        Args:
            stimulus: External input signal strength.
            sensitivity: How responsive the node is to external stimuli.
        """
        # A stimulus can be modeled as a complex number that adds to the signal
        # A simple interpretation: a real-valued push.
        self.signal += sensitivity * stimulus
        # We still clip amplitude to prevent runaway oscillations
        if self.amplitude > 10.0:
            self.amplitude = 10.0
    
    def energy(self) -> float:
        """Calculate the instantaneous energy of the node, proportional to A^2."""
        return 0.5 * self.amplitude ** 2
    
    def sync_measure(self, other: 'ResonantNode') -> float:
        """
        Measure synchronization with another node using their signal vectors.
        
        Args:
            other: Another ResonantNode to compare with.
            
        Returns:
            Synchronization measure (0 = anti-phase, 1 = in-phase).
        """
        # The dot product of normalized vectors gives the cosine of the phase difference
        if self.amplitude == 0 or other.amplitude == 0:
            return 0.0
        
        dot_product = np.dot(self.signal.view(np.float64), other.signal.view(np.float64))
        norm_product = self.amplitude * other.amplitude
        
        cosine_similarity = dot_product / norm_product
        return 0.5 * (1 + cosine_similarity)

    def retune(self, target_frequency: float, rate: float = 0.1) -> None:
        """
        Gradually retune the node to a target frequency.
        
        Args:
            target_frequency: Desired frequency to tune towards.
            rate: Learning rate for frequency adjustment (0-1).
        """
        rate = np.clip(rate, 0.0, 1.0)
        self.natural_freq += rate * (target_frequency - self.natural_freq)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node state to dictionary, preserving legacy format."""
        return {
            'node_id': self.node_id,
            'frequency': self.frequency,
            'phase': self.phase,
            'amplitude': self.amplitude,
            'damping': self.damping,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResonantNode':
        """Create node from dictionary representation."""
        # The __init__ is already compatible with the old format.
        return cls(**data)
    
    def __repr__(self) -> str:
        return (f"ResonantNode(id={self.node_id}, freq={self.natural_freq:.2f}Hz, "
                f"amp={self.amplitude:.2f}, phase={self.phase:.2f}rad)") 