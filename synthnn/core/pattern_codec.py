"""
Pattern encoding and decoding for the SynthNN framework.

This module provides tools for converting between various data types
(audio, text, images) and resonant network patterns.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod

from .resonant_node import ResonantNode
from .resonant_network import ResonantNetwork


class PatternEncoder(ABC):
    """Abstract base class for encoding data into resonant patterns."""
    
    @abstractmethod
    def encode(self, data: Any) -> Dict[str, Dict[str, float]]:
        """
        Encode data into node parameters.
        
        Returns:
            Dictionary mapping node IDs to parameter updates
            (frequency, phase, amplitude)
        """
        pass


class PatternDecoder(ABC):
    """Abstract base class for decoding resonant patterns into data."""
    
    @abstractmethod
    def decode(self, network: ResonantNetwork) -> Any:
        """
        Decode network state into output data.
        
        Args:
            network: ResonantNetwork to decode from
            
        Returns:
            Decoded data in appropriate format
        """
        pass


class AudioPatternEncoder(PatternEncoder):
    """Encode audio signals into resonant network patterns."""
    
    def __init__(self, num_nodes: int = 32, sample_rate: float = 44100):
        self.num_nodes = num_nodes
        self.sample_rate = sample_rate
        self.freq_bands = self._create_frequency_bands()
    
    def _create_frequency_bands(self) -> np.ndarray:
        """Create logarithmically spaced frequency bands."""
        # Cover range from 20Hz to 20kHz
        min_freq = 20
        max_freq = min(20000, self.sample_rate / 2)
        return np.logspace(np.log10(min_freq), np.log10(max_freq), self.num_nodes)
    
    def encode(self, audio_data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Encode audio data into resonant patterns using spectral decomposition.
        
        Args:
            audio_data: Audio signal array
            
        Returns:
            Node parameter updates based on spectral content
        """
        # Compute spectrum
        fft_data = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
        magnitudes = np.abs(fft_data[:len(fft_data)//2])
        freqs = freqs[:len(freqs)//2]
        
        # Map to frequency bands
        node_params = {}
        
        for i in range(self.num_nodes):
            node_id = f"audio_{i}"
            
            # Find band boundaries
            if i == 0:
                low_freq = 0
            else:
                low_freq = (self.freq_bands[i-1] + self.freq_bands[i]) / 2
            
            if i == self.num_nodes - 1:
                high_freq = self.sample_rate / 2
            else:
                high_freq = (self.freq_bands[i] + self.freq_bands[i+1]) / 2
            
            # Extract band energy
            band_mask = (freqs >= low_freq) & (freqs < high_freq)
            band_energy = np.sum(magnitudes[band_mask])
            
            # Normalize and convert to amplitude
            amplitude = np.tanh(band_energy / (np.max(magnitudes) + 1e-8))
            
            # Set node parameters
            node_params[node_id] = {
                'frequency': self.freq_bands[i] / 1000,  # Scale to reasonable range
                'amplitude': amplitude,
                'phase': np.random.uniform(0, 2*np.pi)  # Random initial phase
            }
        
        return node_params


class AudioPatternDecoder(PatternDecoder):
    """Decode resonant patterns into audio signals."""
    
    def __init__(self, sample_rate: float = 44100, duration: float = 1.0):
        self.sample_rate = sample_rate
        self.duration = duration
    
    def decode(self, network: ResonantNetwork) -> np.ndarray:
        """
        Decode network state into audio signal.
        
        Uses additive synthesis based on node oscillations.
        """
        num_samples = int(self.sample_rate * self.duration)
        audio = np.zeros(num_samples)
        time_points = np.linspace(0, self.duration, num_samples)
        
        # Sum contributions from all nodes
        for node_id, node in network.nodes.items():
            if node_id.startswith('audio_'):
                # Generate oscillation
                signal = node.amplitude * np.sin(
                    2 * np.pi * node.frequency * 1000 * time_points + node.phase
                )
                audio += signal
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio


class TextPatternEncoder(PatternEncoder):
    """Encode text/symbols into resonant patterns."""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.char_to_freq = self._create_character_mapping()
    
    def _create_character_mapping(self) -> Dict[str, float]:
        """Map characters to frequencies."""
        # Map ASCII characters to frequencies between 1-10 Hz
        mapping = {}
        for i in range(self.vocab_size):
            char = chr(i) if i < 128 else f"char_{i}"
            mapping[char] = 1.0 + (i / self.vocab_size) * 9.0
        return mapping
    
    def encode(self, text: str) -> Dict[str, Dict[str, float]]:
        """
        Encode text into temporal resonant patterns.
        
        Each character creates a node with specific frequency.
        """
        node_params = {}
        
        for i, char in enumerate(text):
            node_id = f"text_{i}"
            
            # Get frequency for character
            freq = self.char_to_freq.get(char, 5.0)  # Default frequency
            
            # Phase encodes position
            phase = (i / len(text)) * 2 * np.pi
            
            # Amplitude based on character importance (simple heuristic)
            if char.isalpha():
                amplitude = 1.0
            elif char.isdigit():
                amplitude = 0.8
            else:
                amplitude = 0.5
            
            node_params[node_id] = {
                'frequency': freq,
                'phase': phase,
                'amplitude': amplitude
            }
        
        return node_params


class TextPatternDecoder(PatternDecoder):
    """Decode resonant patterns into text."""
    
    def __init__(self, vocab_size: int = 256, threshold: float = 0.5):
        self.vocab_size = vocab_size
        self.threshold = threshold
        self.freq_to_char = self._create_frequency_mapping()
    
    def _create_frequency_mapping(self) -> Dict[int, str]:
        """Map frequency ranges to characters."""
        mapping = {}
        for i in range(self.vocab_size):
            freq_center = 1.0 + (i / self.vocab_size) * 9.0
            mapping[int(freq_center * 10)] = chr(i) if i < 128 else f"char_{i}"
        return mapping
    
    def decode(self, network: ResonantNetwork) -> str:
        """
        Decode network state into text.
        
        Extracts characters based on node frequencies and phases.
        """
        # Get text nodes sorted by phase (position)
        text_nodes = [
            (node_id, node) for node_id, node in network.nodes.items()
            if node_id.startswith('text_') and node.amplitude > self.threshold
        ]
        
        # Sort by phase to get correct order
        text_nodes.sort(key=lambda x: x[1].phase)
        
        # Decode characters
        result = []
        for node_id, node in text_nodes:
            # Find closest character frequency
            freq_key = int(node.frequency * 10)
            char = self.freq_to_char.get(freq_key, '?')
            result.append(char)
        
        return ''.join(result)


class ImagePatternEncoder(PatternEncoder):
    """Encode images into spatial resonant patterns."""
    
    def __init__(self, grid_size: Tuple[int, int] = (8, 8)):
        self.grid_size = grid_size
    
    def encode(self, image: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Encode image into 2D grid of resonant nodes.
        
        Args:
            image: 2D or 3D image array
            
        Returns:
            Node parameters encoding spatial frequencies
        """
        if len(image.shape) == 3:
            # Convert to grayscale
            image = np.mean(image, axis=2)
        
        # Resize to grid
        from scipy.ndimage import zoom
        scale_y = self.grid_size[0] / image.shape[0]
        scale_x = self.grid_size[1] / image.shape[1]
        resized = zoom(image, (scale_y, scale_x))
        
        # Normalize
        resized = (resized - np.min(resized)) / (np.max(resized) - np.min(resized) + 1e-8)
        
        node_params = {}
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                node_id = f"img_{i}_{j}"
                
                # Intensity maps to amplitude
                amplitude = resized[i, j]
                
                # Position encodes spatial frequency
                freq_y = 1.0 + i / self.grid_size[0] * 4.0
                freq_x = 1.0 + j / self.grid_size[1] * 4.0
                frequency = np.sqrt(freq_y**2 + freq_x**2)
                
                # Phase encodes spatial position
                phase = np.arctan2(i - self.grid_size[0]/2, j - self.grid_size[1]/2)
                
                node_params[node_id] = {
                    'frequency': frequency,
                    'phase': phase % (2 * np.pi),
                    'amplitude': amplitude
                }
        
        return node_params


class ImagePatternDecoder(PatternDecoder):
    """Decode spatial resonant patterns into images."""
    
    def __init__(self, output_size: Tuple[int, int] = (64, 64)):
        self.output_size = output_size
    
    def decode(self, network: ResonantNetwork) -> np.ndarray:
        """
        Decode network state into image.
        
        Reconstructs image from spatial frequency components.
        """
        image = np.zeros(self.output_size)
        
        # Extract image nodes
        img_nodes = [
            (node_id, node) for node_id, node in network.nodes.items()
            if node_id.startswith('img_')
        ]
        
        if not img_nodes:
            return image
        
        # Reconstruct using inverse spatial transform
        for y in range(self.output_size[0]):
            for x in range(self.output_size[1]):
                value = 0.0
                
                for node_id, node in img_nodes:
                    # Extract grid position from node_id
                    parts = node_id.split('_')
                    if len(parts) == 3:
                        i, j = int(parts[1]), int(parts[2])
                        
                        # Spatial basis function
                        basis = np.cos(2 * np.pi * (i * y / self.output_size[0] + 
                                                   j * x / self.output_size[1]) + node.phase)
                        
                        value += node.amplitude * basis
                
                image[y, x] = value
        
        # Normalize to [0, 1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        
        return image


class UniversalPatternCodec:
    """
    Universal codec that can handle multiple data types.
    
    Automatically selects appropriate encoder/decoder based on data type.
    """
    
    def __init__(self):
        self.audio_encoder = AudioPatternEncoder()
        self.audio_decoder = AudioPatternDecoder()
        self.text_encoder = TextPatternEncoder()
        self.text_decoder = TextPatternDecoder()
        self.image_encoder = ImagePatternEncoder()
        self.image_decoder = ImagePatternDecoder()
    
    def encode(self, data: Any, data_type: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Encode data of any supported type.
        
        Args:
            data: Input data
            data_type: Optional type hint ('audio', 'text', 'image')
            
        Returns:
            Encoded pattern parameters
        """
        if data_type == 'audio' or isinstance(data, np.ndarray) and len(data.shape) == 1:
            return self.audio_encoder.encode(data)
        elif data_type == 'text' or isinstance(data, str):
            return self.text_encoder.encode(data)
        elif data_type == 'image' or isinstance(data, np.ndarray) and len(data.shape) >= 2:
            return self.image_encoder.encode(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def decode(self, network: ResonantNetwork, output_type: str) -> Any:
        """
        Decode network state to specified output type.
        
        Args:
            network: ResonantNetwork to decode
            output_type: Desired output type ('audio', 'text', 'image')
            
        Returns:
            Decoded data
        """
        if output_type == 'audio':
            return self.audio_decoder.decode(network)
        elif output_type == 'text':
            return self.text_decoder.decode(network)
        elif output_type == 'image':
            return self.image_decoder.decode(network)
        else:
            raise ValueError(f"Unsupported output type: {output_type}") 