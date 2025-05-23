from synthnn.core import AcceleratedMusicalNetwork, ResonantNode # For type hinting / instantiation
from detector import ModeDetector # Assuming detector.py contains ModeDetector
import numpy as np
import time

class AdaptiveModalNetwork:
    def __init__(self, num_nodes_per_network=5, initial_base_freq=1.0, target_phase=0.0, target_amplitude=1.0):
        self.mode_detector = ModeDetector()
        self.mode_networks: dict[str, AcceleratedMusicalNetwork] = {}
        # self.time_steps = np.linspace(0, 1, 100) # Default time_steps for internal processing - may not be needed if generating audio
        
        # Add temporal memory
        self.mode_history = []  # Track mode probabilities over time
        self.fundamental_history = []  # Track fundamental frequencies
        self.output_history = []  # Track network outputs
        self.transition_detector = ModeTransitionDetector()  # New component
        self.memory_window = 50  # Number of timesteps to remember

        # Initialize networks for each mode
        for mode_name, intervals in self.mode_detector.mode_intervals.items():
            network_instance = AcceleratedMusicalNetwork(
                name=f"adaptive_{mode_name}_{int(time.time())}",
                base_freq=initial_base_freq, 
                mode=mode_name, 
                mode_detector=self.mode_detector
            )
            
            # Populate with nodes based on its own mode's intervals
            current_ratios = intervals[:num_nodes_per_network]
            if len(current_ratios) < num_nodes_per_network:
                current_ratios.extend([current_ratios[-1]] * (num_nodes_per_network - len(current_ratios)))
            
            # Create nodes using the network's method, which uses its base_freq
            # The amplitude and phase are set per node by create_harmonic_nodes if not specified per node
            # Here we are creating nodes one by one to match the old logic more closely initially for parameters.
            for i, ratio in enumerate(current_ratios):
                node = ResonantNode(
                    node_id=f"{mode_name}_node_{i}", 
                    frequency=network_instance.base_freq * ratio, # Uses the network's current base_freq
                    amplitude=target_amplitude / np.sqrt(num_nodes_per_network), # Distribute amplitude
                    phase=target_phase
                )
                network_instance.add_node(node)
            
            if num_nodes_per_network > 1:
                network_instance.create_modal_connections(connection_pattern="nearest_neighbor", weight_scale=0.1)
                network_instance.coupling_strength = 0.05 # Low coupling for more independent modal character

            self.mode_networks[mode_name] = network_instance
            print(f"Initialized AcceleratedMusicalNetwork for {mode_name}")

    def process(self, signal_segment, sample_rate=1.0):
        if not signal_segment.any(): # Handle empty signal
            return 0.0 # Return float

        mode_probabilities = self.mode_detector.get_mode_probabilities(signal_segment, sample_rate)
        
        self.mode_history.append(mode_probabilities)
        if len(self.mode_history) > self.memory_window:
            self.mode_history.pop(0)
        
        if len(self.mode_history) > 2:
            # Fundamental history might need to be populated if transition_detector uses it
            # For now, passing None or an empty list if not available.
            fund_history_slice = self.fundamental_history[-3:] if len(self.fundamental_history) >= 3 else None
            transition_info = self.transition_detector.analyze_transition(
                self.mode_history[-3:], 
                fund_history_slice
            )
            
            if transition_info['transitioning']:
                mode_probabilities = self._smooth_mode_transition(
                    mode_probabilities, 
                    transition_info
                )

        # Extract fundamental for retuning
        # Use the SignalProcessor from an arbitrary network instance (they share one if not passed)
        # Or instantiate one here.
        temp_signal_processor = self.mode_networks[list(self.mode_networks.keys())[0]].signal_processor
        temp_signal_processor.sample_rate = sample_rate # Ensure correct sample rate
        
        # features = self.mode_detector._extract_features(signal_segment, sample_rate)
        # Instead of relying on mode_detector internal method, use SignalProcessor directly
        _, spec_mags = temp_signal_processor.analyze_spectrum(signal_segment)
        signal_fundamental = temp_signal_processor.extract_fundamental(signal_segment, freq_range=(30,2000))

        if signal_fundamental == 0: signal_fundamental = 1.0 # Avoid zero frequency
        self.fundamental_history.append(signal_fundamental)
        if len(self.fundamental_history) > self.memory_window:
             self.fundamental_history.pop(0)
        
        total_weighted_output = 0.0
        segment_duration = len(signal_segment) / sample_rate
        if segment_duration == 0: return 0.0

        for mode_name, network_instance in self.mode_networks.items():
            probability = mode_probabilities.get(mode_name, 0.0)
            if probability > 0.001: 
                # Retune this mode-specific network to the current signal's fundamental
                network_instance._retune_to_mode(
                    signal_fundamental, 
                    self.mode_detector.mode_intervals[mode_name]
                )
                
                # Generate a short audio output from this retuned mode network
                network_output_signal = network_instance.generate_audio_accelerated(
                    segment_duration, 
                    sample_rate
                )
                
                # Represent network's response as a scalar (e.g., mean absolute amplitude)
                # This scalar value is then weighted by the mode probability.
                # Could also be energy, synchronization, or a specific feature.
                network_scalar_response = np.mean(np.abs(network_output_signal)) if network_output_signal.any() else 0.0
                
                total_weighted_output += probability * network_scalar_response

        self.output_history.append(total_weighted_output)
        if len(self.output_history) > self.memory_window:
            self.output_history.pop(0)
            
        if len(self.output_history) > 3:
            total_weighted_output = self._apply_temporal_smoothing(
                total_weighted_output,
                self.output_history[-3:] # Pass only the relevant slice
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
        # Ensure recent_outputs is a list of numbers
        valid_recent_outputs = [out for out in recent_outputs if isinstance(out, (int, float))]
        
        all_outputs = valid_recent_outputs + [current_output]
        
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