"""
SynthNN Core Module

This module provides the fundamental building blocks for the SynthNN framework:
- ResonantNode: Basic oscillating unit
- ResonantNetwork: Network of interconnected nodes
- SignalProcessor: Signal analysis and transformation tools
- Pattern codecs: Encoders and decoders for various data types
"""

from .resonant_node import ResonantNode
from .resonant_network import ResonantNetwork, Connection
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

__all__ = [
    'ResonantNode',
    'ResonantNetwork',
    'Connection',
    'SignalProcessor',
    'PatternEncoder',
    'PatternDecoder',
    'AudioPatternEncoder',
    'AudioPatternDecoder',
    'TextPatternEncoder',
    'TextPatternDecoder',
    'ImagePatternEncoder',
    'ImagePatternDecoder',
    'UniversalPatternCodec'
]

__version__ = '0.1.0' 