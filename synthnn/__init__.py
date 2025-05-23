"""
SynthNN - Synthetic Resonant Neural Networks

A framework for building neural networks based on resonance, wave physics,
and harmonic principles. Supports audio synthesis, pattern recognition,
and cross-modal learning.
"""

from synthnn.core import (
    ResonantNode,
    ResonantNetwork,
    SignalProcessor,
    UniversalPatternCodec
)

# Import performance module if available
try:
    from synthnn.performance import (
        BackendManager,
        AcceleratedResonantNetwork,
        BackendType
    )
    _performance_available = True
except ImportError:
    _performance_available = False
    BackendManager = None
    AcceleratedResonantNetwork = None
    BackendType = None

__all__ = [
    'ResonantNode',
    'ResonantNetwork', 
    'SignalProcessor',
    'UniversalPatternCodec'
]

# Add performance exports if available
if _performance_available:
    __all__.extend([
        'BackendManager',
        'AcceleratedResonantNetwork',
        'BackendType'
    ])

__version__ = '0.1.0' 