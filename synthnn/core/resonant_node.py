"""
Core ResonantNode implementation for the SynthNN framework.

This module provides the fundamental building block for all resonant networks
in the system, supporting various signal types and modulation schemes.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ResonantNode:
    """
    A single resonant node that oscillates at a specific frequency with phase and amplitude.
    
    This is the fundamental unit of computation in the SynthNN framework,
    representing a wave-based neuron that can resonate with input signals
    and influence connected nodes through phase coupling.
    
    Attributes:
        node_id: Unique identifier for the node
        frequency: Natural oscillation frequency (Hz)
        phase: Current phase in radians (0 to 2π)
        amplitude: Oscillation strength/magnitude
        damping: Energy dissipation factor (0 = no damping, 1 = critical damping)
        metadata: Optional dictionary for storing additional node properties
    """
    
    node_id: str
    frequency: float = 1.0
    phase: float = 0.0
    amplitude: float = 1.0
    damping: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize initial parameters."""
        self.phase = self.phase % (2 * np.pi)
        self.amplitude = max(0.0, self.amplitude)
        self.damping = np.clip(self.damping, 0.0, 1.0)
    
    def oscillate(self, time: float) -> float:
        """
        Calculate the node's signal at a given time.
        
        Args:
            time: Time point to evaluate the oscillation
            
        Returns:
            Signal value at the specified time
        """
        # Apply damping to amplitude over time
        damped_amplitude = self.amplitude * np.exp(-self.damping * time)
        return damped_amplitude * np.sin(2 * np.pi * self.frequency * time + self.phase)
    
    def update_phase(self, dt: float, phase_coupling: float = 0.0) -> None:
        """
        Update the node's phase based on natural frequency and external coupling.
        
        Args:
            dt: Time step
            phase_coupling: External phase influence from connected nodes
        """
        # Natural frequency contribution
        natural_advance = 2 * np.pi * self.frequency * dt
        
        # Total phase change including coupling
        phase_change = natural_advance + phase_coupling
        
        # Update phase with wrapping
        self.phase = (self.phase + phase_change) % (2 * np.pi)
    
    def apply_stimulus(self, stimulus: float, sensitivity: float = 1.0) -> None:
        """
        Apply external stimulus to the node, affecting its amplitude and phase.
        
        Args:
            stimulus: External input signal strength
            sensitivity: How responsive the node is to external stimuli
        """
        # Amplitude modulation based on stimulus
        self.amplitude *= (1 + sensitivity * np.tanh(stimulus))
        self.amplitude = np.clip(self.amplitude, 0.0, 10.0)  # Prevent runaway amplitudes
        
        # Phase perturbation proportional to stimulus
        phase_shift = sensitivity * stimulus * 0.1
        self.phase = (self.phase + phase_shift) % (2 * np.pi)
    
    def energy(self) -> float:
        """Calculate the instantaneous energy of the node."""
        return 0.5 * self.amplitude ** 2
    
    def sync_measure(self, other: 'ResonantNode') -> float:
        """
        Measure synchronization with another node using phase coherence.
        
        Args:
            other: Another ResonantNode to compare with
            
        Returns:
            Synchronization measure between 0 (no sync) and 1 (perfect sync)
        """
        phase_diff = self.phase - other.phase
        return 0.5 * (1 + np.cos(phase_diff))
    
    def retune(self, target_frequency: float, rate: float = 0.1) -> None:
        """
        Gradually retune the node to a target frequency.
        
        Args:
            target_frequency: Desired frequency to tune towards
            rate: Learning rate for frequency adjustment (0-1)
        """
        rate = np.clip(rate, 0.0, 1.0)
        self.frequency += rate * (target_frequency - self.frequency)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node state to dictionary."""
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
        return cls(**data)
    
    def __repr__(self) -> str:
        return (f"ResonantNode(id={self.node_id}, freq={self.frequency:.2f}Hz, "
                f"phase={self.phase:.2f}rad, amp={self.amplitude:.2f})") 