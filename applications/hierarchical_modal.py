import numpy as np
from .adaptive import AdaptiveModalNetwork
from .detector import ModeDetector

class HierarchicalModalProcessor:
    """
    Processes signals at multiple time scales using modal networks.
    Each level operates at different temporal resolutions, capturing both
    fast local changes and slow global patterns.
    """
    
    def __init__(self, time_scales=[0.1, 0.5, 2.0], num_nodes=5, base_process_sample_rate=44100.0):
        """
        Initialize hierarchical processor with multiple time scales.
        
        Args:
            time_scales: List of time scale factors (e.g., duration in seconds for buffers)
            num_nodes: Number of nodes per adaptive network
            base_process_sample_rate: The sample rate at which this processor expects to receive signal chunks.
        """
        self.time_scales = time_scales
        self.num_levels = len(time_scales)
        self.base_process_sample_rate = base_process_sample_rate
        
        # Create adaptive networks for each time scale
        self.scale_networks = []
        for i, scale_duration in enumerate(time_scales):
            # initial_base_freq for AdaptiveModalNetwork is somewhat arbitrary here,
            # as it gets retuned by its process() method based on input signal fundamental.
            # We can set it based on the scale, e.g., faster scales might have higher typical fundamentals.
            # The buffer_size determines how much audio is processed at this scale.
            buffer_size_samples = int(scale_duration * self.base_process_sample_rate)
            if buffer_size_samples == 0: 
                raise ValueError(f"Time scale {scale_duration}s is too short for sample rate {self.base_process_sample_rate}Hz, results in 0 sample buffer.")

            # The effective frequency content seen by this network will depend on its buffer_size / processing window.
            # We can use a nominal initial_base_freq, perhaps related to 1/scale_duration.
            # The num_nodes_per_network in AdaptiveModalNetwork is different from num_nodes here.
            # Let's use num_nodes for num_nodes_per_network in AdaptiveModalNetwork.
            network = AdaptiveModalNetwork(
                num_nodes_per_network=num_nodes,
                initial_base_freq= 1.0 / scale_duration if scale_duration > 0 else 1.0 
            )
            self.scale_networks.append({
                'network': network,
                'scale_duration': scale_duration, # Store the intended duration for this scale
                'buffer': [],
                'buffer_size_samples': buffer_size_samples
            })
        
        self.interaction_matrix = self._initialize_interactions()
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
    
    def process_signal_chunk(self, signal_chunk, chunk_sample_rate):
        """
        Process an incoming signal chunk hierarchically.
        Note: Each scale network operates on its own buffer and its effective sample rate
        might change if downsampling is applied internally to its buffer content.
        
        Args:
            signal_chunk: Current chunk of signal data
            chunk_sample_rate: Sample rate of the incoming signal_chunk
            
        Returns:
            dict: Multi-scale analysis results for this chunk processing cycle
        """
        results = {
            'scale_outputs': [],
            'fused_output': None,
            'dominant_modes': {},
            'temporal_coherence': {}
        }
        
        all_scale_outputs_this_cycle = []

        for level_idx, scale_info in enumerate(self.scale_networks):
            network = scale_info['network']
            scale_duration = scale_info['scale_duration'] # Time duration this scale focuses on
            buffer = scale_info['buffer']
            buffer_size_samples = scale_info['buffer_size_samples']
            
            # Add current chunk to this scale's buffer
            buffer.extend(signal_chunk)
            
            # Process if buffer has enough data for this scale's intended duration
            if len(buffer) >= buffer_size_samples:
                # Take the most recent `buffer_size_samples` for processing
                current_processing_segment = np.array(buffer[-buffer_size_samples:])
                effective_sample_rate_for_segment = chunk_sample_rate # Assuming input chunk_sample_rate is constant

                # No explicit downsampling here anymore, AdaptiveModalNetwork handles segment
                # The `scale_duration` concept is now managed by `buffer_size_samples`.
                # AdaptiveModalNetwork will process this segment at `effective_sample_rate_for_segment`.

                scale_output_scalar = network.process(current_processing_segment, effective_sample_rate_for_segment)
                
                mode_probs = network.mode_detector.get_mode_probabilities(
                    current_processing_segment, 
                    sample_rate=effective_sample_rate_for_segment
                )
                
                current_scale_result = {
                    'level': level_idx,
                    'scale_duration': scale_duration,
                    'output': scale_output_scalar,
                    'mode_probabilities': mode_probs,
                    'dominant_mode': max(mode_probs.items(), key=lambda x: x[1])[0] if mode_probs else "N/A"
                }
                all_scale_outputs_this_cycle.append(current_scale_result)
                
                # Manage buffer: keep some overlap, discard the processed part that led to full buffer_size_samples
                # This is a simple FIFO overlap; more sophisticated windowing could be used.
                overlap = buffer_size_samples // 2 
                scale_info['buffer'] = list(buffer[len(buffer) - buffer_size_samples + overlap :])
            else:
                # Not enough data in buffer for this scale to process yet
                all_scale_outputs_this_cycle.append(None) # Placeholder or previous state?

        # Only proceed with fusion if all scales produced an output this cycle
        # Or handle partial updates depending on desired behavior.
        # For now, let's assume we wait for all scales that were supposed to produce output.
        valid_scale_outputs = [s_out for s_out in all_scale_outputs_this_cycle if s_out is not None]
        results['scale_outputs'] = valid_scale_outputs

        if len(valid_scale_outputs) == self.num_levels: # Or some other condition like len(valid_scale_outputs) > 0
            results['fused_output'] = self._fuse_scales(valid_scale_outputs)
            results['temporal_coherence'] = self._analyze_coherence(valid_scale_outputs)
            
        return results
    
    def _downsample(self, signal_array, factor):
        """Downsample signal by averaging chunks"""
        if factor <= 1:
            return signal_array
        # Ensure signal_array is a numpy array for efficient slicing
        signal_array = np.asarray(signal_array)
        num_chunks = len(signal_array) // factor
        if num_chunks == 0:
            return np.array([]) # Return empty if not enough data to form one chunk
        trimmed_length = num_chunks * factor
        return np.mean(signal_array[:trimmed_length].reshape(-1, factor), axis=1)
    
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
        modes = [s['dominant_mode'] for s in scale_outputs if s and 'dominant_mode' in s]
        if len(modes) == self.num_levels and self.num_levels > 0: # Ensure all levels reported a mode
            mode_agreement = len(set(modes)) == 1
        else:
            mode_agreement = "N/A" # Not all scales produced output or dominant mode
        
        coherence['mode_alignment'] = mode_agreement
        coherence['mode_distribution'] = modes
        
        # Calculate cross-scale correlation - only meaningful if outputs are time series
        # Current 'output' from AdaptiveModalNetwork.process is a scalar per chunk.
        # To do meaningful correlation, we'd need to store a history of these scalars
        # for each scale and then correlate those histories.
        # For now, this part will likely produce NaNs or be skipped.
        coherence['scale_correlations'] = [] # Initialize as empty
        if len(scale_outputs) > 1:
            # Example: If we were to correlate the *current* scalar outputs (not very meaningful)
            # This part is illustrative and likely won't yield deep insights with single scalars.
            # For a proper implementation, accumulate time series of outputs per scale.
            # For now, let's leave it empty to avoid warnings with scalar inputs.
            pass
            # output_values = [s['output'] for s in scale_outputs if s and 'output' in s]
            # if len(output_values) > 1:
            #     try:
            #         # This will still warn if not enough variance, e.g. all outputs are same
            #         # Pad to at least 2 elements if only one for a pair to avoid error,
            #         # but corrcoef of [x,x] and [y,y] is nan.
            #         # We simply skip if not enough data for meaningful correlation.
            #         pass # Skipping for now to avoid warnings with scalar outputs.
            #     except Exception as e:
            #         print(f"Warning: Could not compute correlation: {e}")
            #         coherence['scale_correlations'] = [np.nan] * (len(scale_outputs) - 1)
        
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
    global_sample_rate = 1000 # Hz, for the generated test signal
    processor = HierarchicalModalProcessor(
        time_scales=[0.5, 1.0, 2.0],  # Time windows in seconds
        num_nodes=5,
        base_process_sample_rate=global_sample_rate
    )
    
    # Generate test signal with multiple frequency components
    t = np.linspace(0, 10, 10 * global_sample_rate, endpoint=False)
    
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
    
    chunk_size = int(0.1 * global_sample_rate) # Process 100ms chunks
    all_results = []
    
    for i in range(0, len(signal) - chunk_size + 1, chunk_size):
        chunk = signal[i:i+chunk_size]
        results = processor.process_signal_chunk(chunk, global_sample_rate)
        if results['scale_outputs']: # Only append if there was processing done
            all_results.append(results)
    
    # Analyze results
    print("Hierarchical Modal Processing Results:")
    print("=" * 50)
    
    for i, results in enumerate(all_results[-5:]):  # Last 5 chunks
        print(f"\nChunk {i}:")
        if 'scale_outputs' in results:
            for scale_out in results['scale_outputs']:
                print(f"  Scale {scale_out['scale_duration']:.2f}s: {scale_out['dominant_mode']}")
        
        if 'temporal_coherence' in results:
            print(f"  Mode alignment: {results['temporal_coherence'].get('mode_alignment', 'N/A')}")
        
        if 'fused_output' in results and results['fused_output']:
            print(f"  Fused output: {results['fused_output']['weighted_sum']:.4f}")


if __name__ == "__main__":
    demonstrate_hierarchical_processing() 