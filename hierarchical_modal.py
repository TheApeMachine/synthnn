import numpy as np
from adaptive import AdaptiveModalNetwork
from detector import ModeDetector
from abstract import ResonantNetwork

class HierarchicalModalProcessor:
    """
    Processes signals at multiple time scales using modal networks.
    Each level operates at different temporal resolutions, capturing both
    fast local changes and slow global patterns.
    """
    
    def __init__(self, time_scales=[0.1, 0.5, 2.0], num_nodes=5):
        """
        Initialize hierarchical processor with multiple time scales.
        
        Args:
            time_scales: List of time scale factors (smaller = faster/local, larger = slower/global)
            num_nodes: Number of nodes per network
        """
        self.time_scales = time_scales
        self.num_levels = len(time_scales)
        
        # Create adaptive networks for each time scale
        self.scale_networks = []
        for scale in time_scales:
            network = AdaptiveModalNetwork(
                num_nodes_per_network=num_nodes,
                initial_base_freq=1.0 / scale  # Higher freq for faster scales
            )
            self.scale_networks.append({
                'network': network,
                'scale': scale,
                'buffer': [],
                'buffer_size': int(100 * scale)
            })
        
        # Cross-scale interaction weights
        self.interaction_matrix = self._initialize_interactions()
        
        # Feature fusion network
        self.fusion_network = ModalFeatureFusion(num_levels=self.num_levels)
        
    def _initialize_interactions(self):
        """Initialize cross-scale interaction weights"""
        # Create interaction matrix where nearby scales influence each other more
        matrix = np.zeros((self.num_levels, self.num_levels))
        for i in range(self.num_levels):
            for j in range(self.num_levels):
                distance = abs(i - j)
                if distance == 0:
                    matrix[i, j] = 1.0
                elif distance == 1:
                    matrix[i, j] = 0.5
                elif distance == 2:
                    matrix[i, j] = 0.2
        return matrix
    
    def process_signal(self, signal_stream, chunk_size=100):
        """
        Process incoming signal stream hierarchically.
        
        Args:
            signal_stream: Incoming signal data
            chunk_size: Size of processing chunks
            
        Returns:
            dict: Multi-scale analysis results
        """
        results = {
            'scale_outputs': [],
            'fused_output': None,
            'dominant_modes': {},
            'temporal_coherence': {}
        }
        
        # Process at each time scale
        for level_idx, scale_info in enumerate(self.scale_networks):
            network = scale_info['network']
            scale = scale_info['scale']
            buffer = scale_info['buffer']
            
            # Add to buffer
            buffer.extend(signal_stream)
            
            # Process when buffer is full
            if len(buffer) >= scale_info['buffer_size']:
                # Downsample for larger time scales
                if scale > 1.0:
                    downsampled = self._downsample(buffer, int(scale))
                    scale_output = network.process(np.array(downsampled))
                else:
                    scale_output = network.process(np.array(buffer))
                
                # Get mode information
                mode_probs = network.mode_detector.get_mode_probabilities(
                    np.array(buffer), 
                    sample_rate=1.0/scale
                )
                
                # Store results
                results['scale_outputs'].append({
                    'level': level_idx,
                    'scale': scale,
                    'output': scale_output,
                    'mode_probabilities': mode_probs,
                    'dominant_mode': max(mode_probs.items(), key=lambda x: x[1])[0]
                })
                
                # Clear buffer
                scale_info['buffer'] = buffer[-scale_info['buffer_size']//2:]
        
        # Cross-scale interaction
        if len(results['scale_outputs']) == self.num_levels:
            results['fused_output'] = self._fuse_scales(results['scale_outputs'])
            results['temporal_coherence'] = self._analyze_coherence(results['scale_outputs'])
            
        return results
    
    def _downsample(self, signal, factor):
        """Downsample signal by averaging chunks"""
        downsampled = []
        for i in range(0, len(signal) - factor + 1, factor):
            downsampled.append(np.mean(signal[i:i+factor]))
        return downsampled
    
    def _fuse_scales(self, scale_outputs):
        """Fuse outputs from different scales using interaction matrix"""
        # Extract outputs
        outputs = np.array([s['output'] for s in scale_outputs])
        
        # Apply interaction weights
        weighted_outputs = np.dot(self.interaction_matrix, outputs)
        
        # Feature fusion
        fused = self.fusion_network.fuse(scale_outputs)
        
        return {
            'weighted_sum': np.sum(weighted_outputs),
            'fusion_output': fused,
            'scale_contributions': weighted_outputs
        }
    
    def _analyze_coherence(self, scale_outputs):
        """Analyze temporal coherence across scales"""
        coherence = {}
        
        # Check if modes align across scales
        modes = [s['dominant_mode'] for s in scale_outputs]
        mode_agreement = len(set(modes)) == 1
        
        coherence['mode_alignment'] = mode_agreement
        coherence['mode_distribution'] = modes
        
        # Calculate cross-scale correlation
        if len(scale_outputs) > 1:
            correlations = []
            for i in range(len(scale_outputs) - 1):
                corr = np.corrcoef(
                    [scale_outputs[i]['output']], 
                    [scale_outputs[i+1]['output']]
                )[0, 1]
                correlations.append(corr)
            coherence['scale_correlations'] = correlations
        
        return coherence


class ModalFeatureFusion:
    """Fuses features from multiple modal networks using attention"""
    
    def __init__(self, num_levels):
        self.num_levels = num_levels
        self.attention_weights = np.ones(num_levels) / num_levels
        
    def fuse(self, scale_outputs):
        """Fuse multi-scale modal features"""
        # Extract features
        features = []
        for output in scale_outputs:
            mode_probs = output['mode_probabilities']
            feature_vec = [
                mode_probs.get(mode, 0) 
                for mode in ['Ionian', 'Dorian', 'Phrygian', 'Lydian', 
                            'Mixolydian', 'Aeolian', 'Locrian']
            ]
            features.append(feature_vec)
        
        features = np.array(features)
        
        # Apply attention
        attended_features = features * self.attention_weights[:, np.newaxis]
        
        # Compute final fusion
        fused = np.sum(attended_features, axis=0)
        
        # Update attention weights based on mode stability
        self._update_attention(scale_outputs)
        
        return {
            'fused_features': fused,
            'attention_weights': self.attention_weights.copy(),
            'feature_matrix': features
        }
    
    def _update_attention(self, scale_outputs):
        """Update attention weights based on mode stability"""
        # Scales with more stable modes get higher attention
        stabilities = []
        for output in scale_outputs:
            mode_probs = output['mode_probabilities']
            # Entropy as inverse of stability
            entropy = -np.sum([p * np.log(p + 1e-10) for p in mode_probs.values()])
            stability = 1.0 / (1.0 + entropy)
            stabilities.append(stability)
        
        # Normalize to get attention weights
        stabilities = np.array(stabilities)
        self.attention_weights = stabilities / np.sum(stabilities)


def demonstrate_hierarchical_processing():
    """Demonstrate hierarchical modal processing"""
    # Create processor
    processor = HierarchicalModalProcessor(
        time_scales=[0.5, 1.0, 2.0],  # Fast, medium, slow
        num_nodes=5
    )
    
    # Generate test signal with multiple frequency components
    t = np.linspace(0, 10, 1000)
    
    # Fast oscillation
    fast_component = np.sin(2 * np.pi * 5 * t)
    
    # Medium oscillation  
    medium_component = 0.5 * np.sin(2 * np.pi * 1 * t + np.pi/4)
    
    # Slow modulation
    slow_component = 0.3 * np.sin(2 * np.pi * 0.2 * t)
    
    # Combine
    signal = fast_component + medium_component + slow_component
    
    # Add some mode-specific harmonics
    signal += 0.2 * np.sin(2 * np.pi * 5 * 9/8 * t)  # Dorian second
    signal += 0.1 * np.sin(2 * np.pi * 5 * 5/4 * t)  # Major third
    
    # Process in chunks
    chunk_size = 100
    all_results = []
    
    for i in range(0, len(signal) - chunk_size, chunk_size):
        chunk = signal[i:i+chunk_size]
        results = processor.process_signal(chunk, chunk_size)
        all_results.append(results)
    
    # Analyze results
    print("Hierarchical Modal Processing Results:")
    print("=" * 50)
    
    for i, results in enumerate(all_results[-5:]):  # Last 5 chunks
        print(f"\nChunk {i}:")
        if 'scale_outputs' in results:
            for scale_out in results['scale_outputs']:
                print(f"  Scale {scale_out['scale']}: {scale_out['dominant_mode']}")
        
        if 'temporal_coherence' in results:
            print(f"  Mode alignment: {results['temporal_coherence'].get('mode_alignment', 'N/A')}")
        
        if 'fused_output' in results and results['fused_output']:
            print(f"  Fused output: {results['fused_output']['weighted_sum']:.4f}")


if __name__ == "__main__":
    demonstrate_hierarchical_processing() 