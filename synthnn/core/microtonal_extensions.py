"""
Microtonal extensions for the SynthNN framework.

This module extends the musical capabilities to support microtonal scales,
non-Western tuning systems, and continuous pitch spaces - areas where
the resonance-based approach naturally excels.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from .musical_extensions import MusicalResonantNetwork
from .resonant_node import ResonantNode


class MicrotonalScale:
    """
    Represents a microtonal scale with arbitrary pitch relationships.
    """
    
    def __init__(self, name: str, intervals: List[float], description: str = ""):
        """
        Initialize a microtonal scale.
        
        Args:
            name: Scale name
            intervals: List of frequency ratios relative to tonic
            description: Optional description of the scale
        """
        self.name = name
        self.intervals = intervals
        self.description = description
        
    def get_frequency(self, base_freq: float, degree: int) -> float:
        """Get frequency for a scale degree, with octave handling."""
        octave = degree // len(self.intervals)
        degree_in_octave = degree % len(self.intervals)
        
        ratio = self.intervals[degree_in_octave] * (2 ** octave)
        return base_freq * ratio
    
    def to_cents(self) -> List[float]:
        """Convert intervals to cents (1200 cents = 1 octave)."""
        return [1200 * np.log2(ratio) for ratio in self.intervals]


class MicrotonalScaleLibrary:
    """
    Library of microtonal scales from various traditions and theories.
    """
    
    @staticmethod
    def get_scales() -> Dict[str, MicrotonalScale]:
        """Get collection of predefined microtonal scales."""
        scales = {}
        
        # Just Intonation scales
        scales['just_major'] = MicrotonalScale(
            'Just Major',
            [1.0, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2.0],
            'Pure harmonic ratios based on the overtone series'
        )
        
        scales['just_minor'] = MicrotonalScale(
            'Just Minor',
            [1.0, 9/8, 6/5, 4/3, 3/2, 8/5, 9/5, 2.0],
            'Just intonation minor scale'
        )
        
        # Pythagorean tuning
        scales['pythagorean'] = MicrotonalScale(
            'Pythagorean',
            [1.0, 9/8, 81/64, 4/3, 3/2, 27/16, 243/128, 2.0],
            'Based on pure fifths (3:2 ratio)'
        )
        
        # Arabic Maqam scales
        scales['maqam_rast'] = MicrotonalScale(
            'Maqam Rast',
            [1.0, 9/8, 27/22, 4/3, 3/2, 27/16, 15/8, 2.0],
            'Arabic Maqam Rast with quarter-tones'
        )
        
        scales['maqam_bayati'] = MicrotonalScale(
            'Maqam Bayati',
            [1.0, 12/11, 32/27, 4/3, 3/2, 128/81, 16/9, 2.0],
            'Arabic Maqam Bayati with neutral seconds'
        )
        
        # Indian Classical scales
        scales['shruti_22'] = MicrotonalScale(
            '22 Shruti System',
            [1.0, 256/243, 16/15, 10/9, 9/8, 32/27, 6/5, 5/4, 81/64,
             4/3, 27/20, 45/32, 729/512, 3/2, 128/81, 8/5, 5/3, 27/16,
             16/9, 9/5, 15/8, 243/128, 2.0],
            'Indian classical 22-shruti system'
        )
        
        # Turkish Makam scales
        scales['makam_hicaz'] = MicrotonalScale(
            'Makam Hicaz',
            [1.0, 16/15, 5/4, 4/3, 3/2, 8/5, 15/8, 2.0],
            'Turkish Makam Hicaz'
        )
        
        # Gamelan scales
        scales['slendro'] = MicrotonalScale(
            'Slendro',
            [1.0, 1.134, 1.287, 1.528, 1.734, 2.0],
            'Javanese 5-tone equal temperament approximation'
        )
        
        scales['pelog_bem'] = MicrotonalScale(
            'Pelog Bem',
            [1.0, 1.076, 1.252, 1.357, 1.559, 1.686, 1.872, 2.0],
            'Javanese Pelog scale, Bem pathet'
        )
        
        # Contemporary/Experimental scales
        scales['bohlen_pierce'] = MicrotonalScale(
            'Bohlen-Pierce',
            [1.0, 27/25, 25/21, 9/7, 7/5, 75/49, 5/3, 9/5, 49/25, 
             15/7, 7/3, 63/25, 25/9, 3.0],
            '13-step scale based on 3:1 ratio instead of 2:1'
        )
        
        scales['wendy_carlos_alpha'] = MicrotonalScale(
            'Wendy Carlos Alpha',
            [1.0, 1.0595, 1.1225, 1.1892, 1.2599, 1.3348, 1.4142,
             1.4983, 1.5874, 1.6818, 1.7818, 1.8877, 2.0],
            'Wendy Carlos Alpha scale - 15.385 cents per step'
        )
        
        scales['harmonic_series_12'] = MicrotonalScale(
            'Harmonic Series 12',
            [1.0, 9/8, 10/8, 11/8, 12/8, 13/8, 14/8, 15/8, 
             16/8, 17/8, 18/8, 19/8, 20/8],
            'First 12 partials of the harmonic series'
        )
        
        # Continuous pitch space generators
        scales['golden_ratio'] = MicrotonalScale(
            'Golden Ratio Scale',
            [1.0, 1.618, 1.618**2, 1.618**3],
            'Scale based on golden ratio (1.618...)'
        )
        
        return scales


class MicrotonalResonantNetwork(MusicalResonantNetwork):
    """
    Extended musical network with microtonal capabilities.
    """
    
    def __init__(self, name: str = "microtonal_network", 
                 base_freq: float = 440.0,
                 scale: Optional[MicrotonalScale] = None,
                 mode_detector: Optional[Any] = None):
        """
        Initialize a microtonal resonant network.
        
        Args:
            name: Network name
            base_freq: Base frequency
            scale: MicrotonalScale instance
            mode_detector: Optional mode detector
        """
        super().__init__(name, base_freq, "custom", mode_detector)
        
        self.scale = scale or MicrotonalScaleLibrary.get_scales()['just_major']
        self.pitch_bend_resolution = 0.01  # cents
        self.glissando_rate = 50.0  # cents per second
        
        # Continuous pitch tracking
        self.pitch_trajectories = {}
        self.target_pitches = {}
        
    def create_scale_nodes(self, num_octaves: int = 1, 
                          amplitude: float = 1.0) -> None:
        """
        Create nodes for all scale degrees across specified octaves.
        
        Args:
            num_octaves: Number of octaves to generate
            amplitude: Initial amplitude for nodes
        """
        self.nodes.clear()
        self.connections.clear()
        
        total_degrees = len(self.scale.intervals) * num_octaves
        
        for degree in range(total_degrees):
            freq = self.scale.get_frequency(self.base_freq, degree)
            node = ResonantNode(
                node_id=f"degree_{degree}",
                frequency=freq,
                phase=0.0,
                amplitude=amplitude / np.sqrt(total_degrees)
            )
            self.add_node(node)
            
            # Track initial pitch
            self.pitch_trajectories[node.node_id] = [freq]
            self.target_pitches[node.node_id] = freq
            
    def create_continuous_pitch_field(self, freq_range: Tuple[float, float],
                                     num_nodes: int = 20,
                                     distribution: str = 'logarithmic') -> None:
        """
        Create a field of nodes distributed across a continuous frequency range.
        
        Args:
            freq_range: (min_freq, max_freq) tuple
            num_nodes: Number of nodes to create
            distribution: 'linear', 'logarithmic', or 'golden'
        """
        self.nodes.clear()
        self.connections.clear()
        
        min_freq, max_freq = freq_range
        
        if distribution == 'linear':
            frequencies = np.linspace(min_freq, max_freq, num_nodes)
        elif distribution == 'logarithmic':
            frequencies = np.logspace(np.log10(min_freq), 
                                    np.log10(max_freq), num_nodes)
        elif distribution == 'golden':
            # Golden ratio spacing
            phi = (1 + np.sqrt(5)) / 2
            frequencies = [min_freq * (phi ** (i/4)) for i in range(num_nodes)]
            frequencies = [f for f in frequencies if f <= max_freq]
        
        for i, freq in enumerate(frequencies):
            node = ResonantNode(
                node_id=f"pitch_{i}",
                frequency=freq,
                phase=np.random.uniform(0, 2*np.pi),
                amplitude=1.0 / np.sqrt(len(frequencies))
            )
            self.add_node(node)
            
            self.pitch_trajectories[node.node_id] = [freq]
            self.target_pitches[node.node_id] = freq
            
    def glissando_to_pitch(self, node_id: str, target_freq: float) -> None:
        """
        Set a target pitch for smooth glissando.
        
        Args:
            node_id: Node to glissando
            target_freq: Target frequency
        """
        if node_id in self.nodes:
            self.target_pitches[node_id] = target_freq
            
    def step_with_glissando(self, dt: float) -> None:
        """
        Step network with smooth pitch glides.
        """
        # Update pitches with glissando
        for node_id, node in self.nodes.items():
            if node_id in self.target_pitches:
                current_freq = node.frequency
                target_freq = self.target_pitches[node_id]
                
                if abs(current_freq - target_freq) > 0.1:  # Hz threshold
                    # Calculate glissando step
                    cents_diff = 1200 * np.log2(target_freq / current_freq)
                    max_cents_change = self.glissando_rate * dt
                    
                    if abs(cents_diff) > max_cents_change:
                        # Partial step towards target
                        cents_change = np.sign(cents_diff) * max_cents_change
                        new_freq = current_freq * (2 ** (cents_change / 1200))
                    else:
                        # Reach target
                        new_freq = target_freq
                    
                    node.retune(new_freq)
                    self.pitch_trajectories[node_id].append(new_freq)
        
        # Regular network step
        super().step(dt)
        
    def apply_comma_pump(self, comma_type: str = 'syntonic') -> None:
        """
        Apply comma pump modulation for exploring microtonal spaces.
        
        Args:
            comma_type: Type of comma ('syntonic', 'pythagorean', 'diaschisma')
        """
        commas = {
            'syntonic': 81/80,  # Syntonic comma
            'pythagorean': 3**12 / 2**19,  # Pythagorean comma
            'diaschisma': 2048/2025,  # Diaschisma
            'schisma': 32805/32768  # Schisma
        }
        
        ratio = commas.get(comma_type, 1.0)
        
        # Apply comma shift to all nodes
        for node in self.nodes.values():
            new_freq = node.frequency * ratio
            self.glissando_to_pitch(node.node_id, new_freq)
            
    def create_spectral_connections(self, harmonicity_threshold: float = 0.1) -> None:
        """
        Create connections based on spectral relationships.
        
        Args:
            harmonicity_threshold: Threshold for harmonic relationship detection
        """
        node_list = list(self.nodes.items())
        
        for i, (id1, node1) in enumerate(node_list):
            for j, (id2, node2) in enumerate(node_list[i+1:], i+1):
                # Calculate frequency ratio
                ratio = node2.frequency / node1.frequency
                
                # Check for harmonic relationship
                closest_harmonic = round(ratio)
                if closest_harmonic > 0:
                    harmonicity = 1.0 - abs(ratio - closest_harmonic) / closest_harmonic
                    
                    if harmonicity > harmonicity_threshold:
                        # Weight based on harmonicity and harmonic number
                        weight = harmonicity / closest_harmonic
                        self.connect(id1, id2, weight)
                        self.connect(id2, id1, weight)

    def generate_signals(self, duration: float, sample_rate: float = 44100) -> np.ndarray:
        """
        Generate an audio signal from the current network state for a given duration.

        Args:
            duration: Duration of the signal in seconds.
            sample_rate: The sample rate for the generated signal.

        Returns:
            A numpy array representing the audio signal.
        """
        num_samples = int(duration * sample_rate)
        output_signal = np.zeros(num_samples)
        dt = 1.0 / sample_rate

        for i in range(num_samples):
            # Use the step method from the parent class (ResonantNetwork)
            # to advance the network state. This is suitable for generating
            # segments of audio for distinct notes without continuous glissando.
            super().step(dt) 
            
            current_sum = 0.0
            # get_signals() is inherited from ResonantNetwork
            active_signals = self.get_signals() 
            for signal_val in active_signals.values():
                current_sum += signal_val
            output_signal[i] = current_sum
        
        # Scale the output to prevent clipping, similar to generate_microtonal_texture.
        # A common scaling factor.
        return output_signal * 0.5
                        
    def generate_microtonal_texture(self, duration: float, 
                                   density: float = 0.5,
                                   evolution_rate: float = 0.1,
                                   sample_rate: float = 44100) -> np.ndarray:
        """
        Generate evolving microtonal texture.
        
        Args:
            duration: Duration in seconds
            density: Node activation density (0-1)
            evolution_rate: Rate of timbral evolution
            sample_rate: Output sample rate
            
        Returns:
            Audio signal
        """
        num_samples = int(duration * sample_rate)
        output = np.zeros(num_samples)
        
        # Initialize random activations based on density
        for node in self.nodes.values():
            if np.random.random() < density:
                node.amplitude = np.random.uniform(0.3, 1.0)
            else:
                node.amplitude = 0.0
                
        # Generate texture with evolution
        for i in range(num_samples):
            # Occasional activation changes
            if np.random.random() < evolution_rate / sample_rate:
                node_id = np.random.choice(list(self.nodes.keys()))
                node = self.nodes[node_id]
                
                if node.amplitude > 0.1:
                    # Fade out
                    self.target_pitches[node_id] = node.frequency * np.random.uniform(0.98, 1.02)
                    node.amplitude *= 0.9
                else:
                    # Fade in with slight detuning
                    node.amplitude = np.random.uniform(0.3, 0.7)
                    
            # Step with glissando
            self.step_with_glissando(1.0 / sample_rate)
            
            # Collect output
            signals = self.get_signals()
            output[i] = sum(signals.values()) * 0.5  # Scale for headroom
            
        return output


class AdaptiveMicrotonalSystem:
    """
    System that can learn and adapt to microtonal contexts.
    """
    
    def __init__(self, base_network: MicrotonalResonantNetwork):
        """
        Initialize adaptive microtonal system.
        
        Args:
            base_network: Base microtonal network
        """
        self.network = base_network
        self.learned_scales = {}
        self.interval_memory = []
        self.consonance_map = {}
        
    def learn_scale_from_performance(self, audio: np.ndarray, 
                                    sample_rate: float = 44100,
                                    name: str = "learned_scale") -> MicrotonalScale:
        """
        Extract and learn a scale from a musical performance.
        
        Args:
            audio: Audio signal to analyze
            sample_rate: Sample rate
            name: Name for the learned scale
            
        Returns:
            Learned MicrotonalScale
        """
        # Extract pitch contour
        processor = self.network.signal_processor
        processor.sample_rate = sample_rate
        
        # Get fundamental frequency over time
        window_size = int(0.1 * sample_rate)  # 100ms windows
        hop_size = window_size // 2
        
        pitches = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i+window_size]
            fundamental = processor.extract_fundamental(window, freq_range=(50, 2000))
            if fundamental > 0:
                pitches.append(fundamental)
                
        # Cluster pitches to find scale degrees
        if not pitches:
            return MicrotonalScale(name, [1.0], "No pitches detected")
            
        # Convert to cents relative to median pitch
        median_pitch = np.median(pitches)
        cents = [1200 * np.log2(p / median_pitch) for p in pitches]
        
        # Simple clustering - find modes of the distribution
        hist, bins = np.histogram(cents, bins=50)
        peaks = []
        
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peak_cents = (bins[i] + bins[i+1]) / 2
                peaks.append(peak_cents)
                
        # Convert back to ratios
        intervals = [2 ** (c / 1200) for c in sorted(peaks)]
        
        # Normalize to start at 1.0
        if intervals and intervals[0] > 0:
            intervals = [i / intervals[0] for i in intervals]
            
        scale = MicrotonalScale(name, intervals, f"Learned from performance")
        self.learned_scales[name] = scale
        
        return scale
        
    def find_consonant_intervals(self, freq_range: Tuple[float, float],
                                resolution_cents: float = 5.0) -> List[float]:
        """
        Find consonant intervals in a frequency range using network resonance.
        
        Args:
            freq_range: (min_ratio, max_ratio) to test
            resolution_cents: Resolution in cents
            
        Returns:
            List of consonant frequency ratios
        """
        min_ratio, max_ratio = freq_range
        
        # Test intervals
        test_ratios = []
        current_ratio = min_ratio
        
        while current_ratio <= max_ratio:
            test_ratios.append(current_ratio)
            current_ratio *= 2 ** (resolution_cents / 1200)
            
        consonant_ratios = []
        
        # Create test network with two nodes
        test_network = MicrotonalResonantNetwork("consonance_test")
        test_network.add_node(ResonantNode("root", 440.0, 0, 1.0))
        test_network.add_node(ResonantNode("interval", 440.0, 0, 1.0))
        test_network.connect("root", "interval", 0.5)
        test_network.connect("interval", "root", 0.5)
        
        for ratio in test_ratios:
            # Set interval
            test_network.nodes["interval"].retune(440.0 * ratio)
            
            # Measure phase coherence after settling
            for _ in range(100):
                test_network.step(0.001)
                
            # Check phase alignment
            phase_diff = abs(test_network.nodes["root"].phase - 
                           test_network.nodes["interval"].phase)
            phase_diff = phase_diff % (2 * np.pi)
            
            # Consonance based on phase locking
            consonance = np.cos(phase_diff) ** 2
            
            if consonance > 0.8:  # Threshold for consonance
                consonant_ratios.append(ratio)
                self.consonance_map[ratio] = consonance
                
        return consonant_ratios 