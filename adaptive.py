from abstract import ResonantNetwork as AbstractResNet, ResonantNode # For type hinting / instantiation
from detector import ModeDetector # Assuming detector.py contains ModeDetector
import numpy as np

class AdaptiveModalNetwork:
    def __init__(self, num_nodes_per_network=5, initial_base_freq=1.0, target_phase=0.0, target_amplitude=1.0):
        self.mode_detector = ModeDetector()
        self.mode_networks = {}
        self.time_steps = np.linspace(0, 1, 100) # Default time_steps for internal processing
        
        # Add temporal memory
        self.mode_history = []  # Track mode probabilities over time
        self.fundamental_history = []  # Track fundamental frequencies
        self.output_history = []  # Track network outputs
        self.transition_detector = ModeTransitionDetector()  # New component
        self.memory_window = 50  # Number of timesteps to remember

        # Initialize networks for each mode
        for mode_name, intervals in self.mode_detector.mode_intervals.items():
            current_ratios = intervals[:num_nodes_per_network]
            if len(current_ratios) < num_nodes_per_network:
                current_ratios.extend([current_ratios[-1]] * (num_nodes_per_network - len(current_ratios)))
            
            # Initialize the ResonantNetwork with its designated mode and intervals
            # The 'ResonantNetwork' needs to be modified to accept 'mode_name' and use it.
            # Or, it just takes harmonic_ratios and doesn't care about the name itself.
            self.mode_networks[mode_name] = AbstractResNet(
                num_nodes=num_nodes_per_network,
                base_freq=initial_base_freq, # This is an initial base frequency
                harmonic_ratios=current_ratios, # These are fixed for this mode network
                target_phase=target_phase,
                target_amplitude=target_amplitude
            )
            # Add a 'process_input' method to ResonantNetwork (as sketched above)
            # For this to work, ResonantNetwork must be designed to call self._apply_retuning correctly.
            # Crucially, `ResonantNetwork.mode` should be set to `mode_name` for the sketch above to work.
            self.mode_networks[mode_name].mode = mode_name # Set the mode for this network instance

            print(f"Initialized ResonantNetwork for {mode_name}")

    def process(self, signal_segment, sample_rate=1.0):
        if not signal_segment.any(): # Handle empty signal
            return 0

        mode_probabilities = self.mode_detector.get_mode_probabilities(signal_segment, sample_rate)
        
        # Store in temporal memory
        self.mode_history.append(mode_probabilities)
        if len(self.mode_history) > self.memory_window:
            self.mode_history.pop(0)
        
        # Detect mode transitions and adjust accordingly
        if len(self.mode_history) > 2:
            transition_info = self.transition_detector.analyze_transition(
                self.mode_history[-3:], 
                self.fundamental_history[-3:] if len(self.fundamental_history) >= 3 else None
            )
            
            # Smooth transitions between modes
            if transition_info['transitioning']:
                mode_probabilities = self._smooth_mode_transition(
                    mode_probabilities, 
                    transition_info
                )

        # For adaptive retuning of each mode network, extract fundamental from the current signal segment
        features = self.mode_detector._extract_features(signal_segment, sample_rate)
        signal_fundamental = features.get('fundamental', 1.0)
        if signal_fundamental == 0: signal_fundamental = 1.0 # Avoid zero frequency
        
        total_weighted_output = 0.0

        for mode_name, network_instance in self.mode_networks.items():
            probability = mode_probabilities.get(mode_name, 0.0)
            if probability > 0.001: # Small threshold to avoid processing for near-zero probabilities
                # Temporarily set the base_freq of the network for this processing run
                # This requires ResonantNetwork to be able to adapt its node frequencies
                # based on a new base_freq while keeping its modal harmonic_ratios.
                original_base_freq = network_instance.base_freq
                # network_instance.base_freq = signal_fundamental # This itself isn't enough; nodes need retuning
                
                # Let's refine ResonantNetwork: add a method to retune based on a *new base frequency*
                # while preserving its characteristic modal ratios.
                # `retune_to_new_base(self, new_base_freq, current_time_steps)`
                # Inside ResonantNetwork:
                # def retune_to_new_base(self, new_base_freq, current_time_steps, mode_detector_ref):
                #     self.base_freq = new_base_freq
                #     modal_ratios = mode_detector_ref.mode_intervals[self.mode]
                #     # Trim/pad modal_ratios to self.num_nodes
                #     final_ratios = modal_ratios[:self.num_nodes] 
                #     if len(final_ratios) < self.num_nodes:
                #         final_ratios.extend(...) 
                #     new_frequencies = [new_base_freq * ratio for ratio in final_ratios]
                #     self._apply_retuning(new_frequencies, current_time_steps) # Call existing helper
                
                # In AdaptiveModalNetwork.process:
                network_instance.retune_to_new_base(signal_fundamental, self.time_steps, self.mode_detector) # Assume this method exists and retunes nodes
                
                # After retuning to the signal's fundamental, compute harmonic state.
                network_instance.harmonic_outputs = [] # Clear
                network_instance.compute_harmonic_state(self.time_steps) # Uses the newly retuned freqs

                # Define network_output based on the harmonic state
                network_output = np.sum(network_instance.harmonic_outputs[-1]) if network_instance.harmonic_outputs else 0.0
                
                total_weighted_output += probability * network_output
                
                # Optional: restore original_base_freq if necessary, though for continuous processing,
                # the base_freq should adapt to the input.
                # network_instance.base_freq = original_base_freq (and retune nodes back)

        # Store output for temporal coherence
        self.output_history.append(total_weighted_output)
        if len(self.output_history) > self.memory_window:
            self.output_history.pop(0)
            
        # Apply temporal smoothing if needed
        if len(self.output_history) > 3:
            total_weighted_output = self._apply_temporal_smoothing(
                total_weighted_output,
                self.output_history[-3:]
            )

        return total_weighted_output
    
    def _smooth_mode_transition(self, current_probs, transition_info):
        """Smooth transitions between modes to avoid jarring changes"""
        if transition_info['from_mode'] and transition_info['to_mode']:
            # Create interpolation weights
            progress = transition_info['progress']  # 0 to 1
            
            # Boost target mode probability gradually
            smoothed_probs = current_probs.copy()
            smoothed_probs[transition_info['from_mode']] *= (1 - progress * 0.5)
            smoothed_probs[transition_info['to_mode']] *= (1 + progress * 0.5)
            
            # Renormalize
            total = sum(smoothed_probs.values())
            return {k: v/total for k, v in smoothed_probs.items()}
        
        return current_probs
    
    def _apply_temporal_smoothing(self, current_output, recent_outputs):
        """Apply temporal smoothing to reduce abrupt changes"""
        # Combine recent outputs with current
        all_outputs = recent_outputs + [current_output]
        
        # Create weights based on actual number of outputs
        num_outputs = len(all_outputs)
        if num_outputs == 1:
            return current_output
        
        # Exponential weights - more weight on recent
        weights = np.exp(np.linspace(-2, 0, num_outputs))
        weights /= weights.sum()  # Normalize
        
        smoothed = np.average(all_outputs, weights=weights)
        return smoothed


class ModeTransitionDetector:
    """Detects and analyzes transitions between musical modes"""
    
    def analyze_transition(self, mode_history, fundamental_history=None):
        """Analyze recent mode history to detect transitions"""
        if len(mode_history) < 2:
            return {'transitioning': False}
        
        # Calculate mode stability
        dominant_modes = []
        for probs in mode_history:
            dominant_modes.append(max(probs.items(), key=lambda x: x[1])[0])
        
        # Check if mode is changing
        if len(set(dominant_modes)) > 1:
            from_mode = dominant_modes[0]
            to_mode = dominant_modes[-1]
            
            # Calculate transition progress
            progress = 0.0
            for i, probs in enumerate(mode_history):
                progress += probs.get(to_mode, 0) * (i + 1) / len(mode_history)
            progress /= sum(range(1, len(mode_history) + 1)) / len(mode_history)
            
            return {
                'transitioning': True,
                'from_mode': from_mode,
                'to_mode': to_mode,
                'progress': progress,
                'stable': False
            }
        
        return {
            'transitioning': False,
            'stable': True,
            'dominant_mode': dominant_modes[0]
        }