"""
SynthNN Core Module

This module provides the fundamental building blocks for synthetic resonant neural networks,
including nodes, networks, signal processing, and pattern encoding/decoding.
"""

from .resonant_node import ResonantNode
from .resonant_network import ResonantNetwork
from .signal_processor import SignalProcessor
from .pattern_codec import (
    PatternEncoder,
    PatternDecoder,
    AudioPatternEncoder,
    AudioPatternDecoder,
    TextPatternEncoder,
    TextPatternDecoder,
    ImagePatternEncoder,
    ImagePatternDecoder,
    UniversalPatternCodec
)
from .musical_extensions import MusicalResonantNetwork
from .accelerated_musical_network import AcceleratedMusicalNetwork
from .microtonal_extensions import (
    MicrotonalScale,
    MicrotonalScaleLibrary,
    MicrotonalResonantNetwork,
    AdaptiveMicrotonalSystem
)
from .emotional_resonance import EmotionalResonanceEngine, EmotionCategory
from .resonance_field import ResonanceField4D, SpatialResonantNode, BoundaryCondition
from .collective_intelligence import CollectiveIntelligence, CommunicationMode, ConsensusMethod, NetworkRole
from .evolutionary_resonance import EvolutionaryResonance, FitnessMetric, Genome, Species

__all__ = [
    'ResonantNode',
    'ResonantNetwork',
    'SignalProcessor',
    'PatternEncoder',
    'PatternDecoder',
    'AudioPatternEncoder',
    'AudioPatternDecoder',
    'TextPatternEncoder',
    'TextPatternDecoder',
    'ImagePatternEncoder',
    'ImagePatternDecoder',
    'UniversalPatternCodec',
    'MusicalResonantNetwork',
    'AcceleratedMusicalNetwork',
    'MicrotonalScale',
    'MicrotonalScaleLibrary',
    'MicrotonalResonantNetwork',
    'AdaptiveMicrotonalSystem',
    'EmotionalResonanceEngine',
    'EmotionCategory',
    'ResonanceField4D',
    'SpatialResonantNode',
    'BoundaryCondition',
    'CollectiveIntelligence',
    'CommunicationMode',
    'ConsensusMethod',
    'NetworkRole',
    'EvolutionaryResonance',
    'FitnessMetric',
    'Genome',
    'Species'
]

__version__ = '0.1.0' 