import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.model_selection import KFold
import json

# Load Morse code data at module level
with open("morse_data.json") as f:
    data = json.load(f)
    morse_code = data['letters']

class WaveNeuron:
    """
    Represents a single wave neuron in the network.

    Attributes:
        id (str or int): Unique identifier for the neuron.
        natural_freq (float): Intrinsic oscillation frequency.
        phase (float): Current phase of the oscillation (in radians, 0 to 2π).
        amplitude (float): Strength of the oscillation.
        connections (list): List of IDs of neurons this neuron connects to.
    """
    def __init__(self, id, natural_freq=1.0, phase=0, amplitude=1.0):
        """
        Initializes a WaveNeuron.

        Args:
            id (str or int): Unique identifier for the neuron.
            natural_freq (float, optional): Natural frequency of the neuron. Defaults to 1.0.
            phase (float, optional): Initial phase of the neuron (in radians). Defaults to 0.
            amplitude (float, optional): Oscillation amplitude. Defaults to 1.0.
        """
        self.id = id
        self.natural_freq = natural_freq  # Intrinsic frequency preference
        self.phase = phase                # Current phase (0 to 2π)
        self.amplitude = amplitude        # Oscillation strength
        self.connections = []             # Connected neurons

    def signal(self, t):
        """
        Calculates the current wave output of the neuron at a given time.

        Args:
            t (float): Time.

        Returns:
            float: The signal value at time t.
        """
        return self.amplitude * np.sin(2 * np.pi * self.natural_freq * t + self.phase)

    def update_phase(self, dt, influences=0):
        """
        Updates the neuron's phase based on its natural frequency and external influences.

        Args:
            dt (float): Time step size.
            influences (float or np.ndarray, optional): External phase influences. Defaults to 0.
        """
        # Natural frequency contribution
        phase_change = dt * 2 * np.pi * self.natural_freq

        # Add external influences
        phase_change += influences

        # Update phase (keeping within 0-2π range)
        self.phase = (self.phase + phase_change) % (2 * np.pi)


class WaveNeuronNetwork:
    """
    Represents a network of interconnected wave neurons.

    Attributes:
        neurons (dict): Dictionary to store neurons, keyed by their IDs.
        connection_weights (dict): Dictionary to store connection weights, keyed by (source_id, target_id) tuples.
        connection_delays (dict): Dictionary to store connection delays, keyed by (source_id, target_id) tuples.
        time (float): Current simulation time.
        signal_history (dict): Dictionary to store historical signals.
    """
    def __init__(self):
        """
        Initializes a WaveNeuronNetwork.
        """
        self.neurons = {}              # Dictionary of neurons by ID
        self.connection_weights = {}   # Dictionary of connection strengths
        self.connection_delays = {}    # Add delay storage
        self.time = 0.0                # Current simulation time
        self.signal_history = {}  # Track historical signals

    def add_neuron(self, id, freq=1.0, phase=None):
        """
        Adds a neuron to the network.

        Args:
            id (str or int): Unique identifier for the neuron.
            freq (float, optional): Natural frequency of the neuron. Defaults to 1.0.
            phase (float, optional): Initial phase of the neuron (in radians). If None, a random phase is assigned.
        """
        if phase is None:
            phase = np.random.uniform(0, 2*np.pi)  # Random initial phase
        self.neurons[id] = WaveNeuron(id, freq, phase)

    def connect(self, source_id, target_id, weight=0.1, delay=0):
        """
        Connects two neurons with a given weight and delay.

        Args:
            source_id (str or int): ID of the source neuron.
            target_id (str or int): ID of the target neuron.
            weight (float, optional): Connection weight. Defaults to 0.1.
            delay (float, optional): Connection delay. Defaults to 0.
        """
        if source_id in self.neurons and target_id in self.neurons:
            self.connection_weights[(source_id, target_id)] = weight
            self.connection_delays[(source_id, target_id)] = delay  # Store delay
            self.neurons[source_id].connections.append(target_id)

    def update(self, dt=0.01, external_inputs=None):
        """
        Updates all neurons for one time step.

        Args:
            dt (float, optional): Time step size. Defaults to 0.01.
            external_inputs (dict, optional): Dictionary of external inputs, keyed by neuron IDs.
                                            Input values are added to the neuron's influence.
        """
        # Store current signals in history
        current_signals = {id: neuron.signal(self.time) 
                         for id, neuron in self.neurons.items()}
        self.signal_history[self.time] = current_signals
        
        # Calculate influences for all neurons
        influences = {id: 0 for id in self.neurons}

        # Add external inputs if provided
        if external_inputs:
            for neuron_id, input_value in external_inputs.items():
                influences[neuron_id] += input_value

        # Calculate internal influences from connected neurons
        for source_id, neuron in self.neurons.items():
            for target_id in neuron.connections:
                # Phase difference influence
                phase_diff = np.sin(self.neurons[source_id].phase -
                                    self.neurons[target_id].phase)
                weight = self.connection_weights[(source_id, target_id)]
                influences[target_id] += weight * phase_diff

        # Calculate delayed influences
        for (src, tgt), delay in self.connection_delays.items():
            if delay > 0:
                history_time = self.time - delay
                if history_time in self.signal_history:
                    delayed_signal = self.signal_history[history_time][src]
                    influences[tgt] += self.connection_weights[(src, tgt)] * delayed_signal

        # Update all neurons
        for neuron_id, neuron in self.neurons.items():
            neuron.update_phase(dt, influences[neuron_id])

        # Increment time
        self.time += dt

    def update_connections(self, learning_rate=0.01):
        """
        Updates connection weights based on phase alignment using a Hebbian-like rule.

        Args:
            learning_rate (float, optional): Learning rate for weight updates. Defaults to 0.01.
        """
        for (source_id, target_id), weight in self.connection_weights.items():
            source = self.neurons[source_id]
            target = self.neurons[target_id]

            # Hebbian-like learning: strengthen connections between
            # neurons that are in phase with each other
            phase_alignment = np.cos(source.phase - target.phase)

            # Update weight based on phase alignment
            new_weight = weight + learning_rate * phase_alignment

            # Keep weights in reasonable range
            self.connection_weights[(source_id, target_id)] = np.clip(new_weight, 0, 1)

    def encode_input(self, data, input_neuron_ids):
        """
        Encodes input data into phase influences for specified input neurons.

        Args:
            data (int, float, list, or np.ndarray): Input data to encode.
            input_neuron_ids (list): List of neuron IDs to receive input influence.

        Returns:
            dict: Dictionary of influences, keyed by neuron IDs.
        """
        influences = {}

        # For simple numerical inputs:
        if isinstance(data, (int, float)):
            for id in input_neuron_ids:
                influences[id] = data * 0.1

        # For array data (e.g., audio samples, frequency bands):
        elif isinstance(data, (list, np.ndarray)):
            for i, value in enumerate(data):
                if i < len(input_neuron_ids):
                    influences[input_neuron_ids[i]] = value * 0.1

        return influences

    def read_output(self, output_neuron_ids):
        """
        Reads the current output from specified output neurons.

        Args:
            output_neuron_ids (list): List of neuron IDs to read output from.

        Returns:
            list: List of output values (current signals of output neurons).
        """
        outputs = []

        # Simple approach: read current signal values
        for id in output_neuron_ids:
            outputs.append(
                self.neurons[id].signal(self.time)
            )

        # More advanced: measure synchronization between neuron groups
        # group_sync = self.measure_synchronization(output_neuron_ids)

        return outputs

def morse_to_signal(morse_pattern, dot_duration=0.1, amplitude=1.0):
    """Convert a Morse code pattern to a time-domain signal."""
    # Define durations
    dash_duration = 3 * dot_duration
    symbol_gap = dot_duration
    
    # Calculate total duration
    total_duration = 0
    for symbol in morse_pattern:
        if symbol == '.':
            total_duration += dot_duration
        elif symbol == '-':
            total_duration += dash_duration
        total_duration += symbol_gap  # Gap after each symbol
    
    # Remove the last gap
    total_duration -= symbol_gap
    
    # Create time and signal arrays
    dt = dot_duration / 10  # Sample at 10x the dot duration for smoothness
    times = np.arange(0, total_duration, dt)
    signal = np.zeros_like(times)
    
    # Create the signal
    current_time = 0
    for symbol in morse_pattern:
        if symbol == '.':
            # Find indices for this dot
            indices = np.where((times >= current_time) & (times < current_time + dot_duration))
            signal[indices] = amplitude
            current_time += dot_duration
        elif symbol == '-':
            # Find indices for this dash
            indices = np.where((times >= current_time) & (times < current_time + dash_duration))
            signal[indices] = amplitude
            current_time += dash_duration
        
        # Add gap after each symbol (except the last one)
        if current_time < total_duration:
            current_time += symbol_gap
    
    return times, signal

class MorseCodeRecognizer:
    def __init__(self, num_input_neurons=5, num_hidden_neurons=8):
        self.morse_code = morse_code  # Use the global variable
        self.network = WaveNeuronNetwork()
        self.input_neurons = [f"input_{i}" for i in range(num_input_neurons)]
        self.hidden_neurons = [f"hidden_{i}" for i in range(num_hidden_neurons)]
        self.output_neurons = [f"output_{char}" for char in self.morse_code.keys()]
        
        # Add input neurons with specialized temporal encoding
        for i, id in enumerate(self.input_neurons):
            # Create three types of input neurons:
            # 1. Direct temporal neurons (lower frequencies)
            # 2. Pattern detection neurons (mid frequencies)
            # 3. Integration neurons (higher frequencies)
            if i < num_input_neurons // 3:
                # Direct temporal neurons - lower frequencies for precise timing
                freq = 0.8 + 0.1 * i
            elif i < 2 * (num_input_neurons // 3):
                # Pattern detection neurons - mid frequencies
                freq = 1.5 + 0.2 * (i - num_input_neurons // 3)
            else:
                # Integration neurons - higher frequencies
                freq = 2.5 + 0.3 * (i - 2 * (num_input_neurons // 3))
            self.network.add_neuron(id, freq=freq)
            
        # Add hidden neurons in two groups: pattern and timing
        hidden_per_group = num_hidden_neurons // 2
        for i, id in enumerate(self.hidden_neurons):
            if i < hidden_per_group:
                # Pattern detection group (higher frequencies)
                freq = 2.0 + 0.15 * i
            else:
                # Timing group (lower frequencies)
                freq = 1.5 + 0.1 * (i - hidden_per_group)
            self.network.add_neuron(id, freq=freq)
            
        # Add output neurons with distinct frequencies
        for i, id in enumerate(self.output_neurons):
            freq = 3.0 + 0.3 * i  # Higher frequencies for output
            self.network.add_neuron(id, freq=freq)
            
        # Enhanced connection strategy
        self._setup_enhanced_connections(hidden_per_group)
    
    def _setup_enhanced_connections(self, hidden_per_group):
        """Setup enhanced connections between layers with specialized weights."""
        num_input_groups = 3  # Direct temporal, Pattern detection, Integration
        inputs_per_group = len(self.input_neurons) // num_input_groups
        
        # Connect inputs to hidden layer with specialized weights
        for i, input_id in enumerate(self.input_neurons):
            input_group = i // inputs_per_group  # 0: temporal, 1: pattern, 2: integration
            
            for j, hidden_id in enumerate(self.hidden_neurons):
                if j < hidden_per_group:  # Pattern detection hidden neurons
                    if input_group == 1:  # Pattern detection inputs
                        # Strong connections for pattern matching
                        weight = np.random.uniform(0.15, 0.25)
                    else:
                        # Weaker connections from other input types
                        weight = np.random.uniform(0.05, 0.15)
                else:  # Timing hidden neurons
                    if input_group == 0:  # Direct temporal inputs
                        # Strong connections for timing
                        weight = np.random.uniform(0.15, 0.25)
                    else:
                        # Weaker connections from other input types
                        weight = np.random.uniform(0.05, 0.15)
                self.network.connect(input_id, hidden_id, weight=weight)
        
        # Connect hidden to output layer with enhanced specificity
        for i, hidden_id in enumerate(self.hidden_neurons):
            for j, output_id in enumerate(self.output_neurons):
                morse_pattern = self.morse_code[output_id.split('_')[1]]
                pattern_complexity = len(morse_pattern)
                
                if i < hidden_per_group:  # Pattern neurons
                    # Scale weight based on pattern complexity
                    base_weight = 0.2 + 0.05 * pattern_complexity
                    weight = np.random.uniform(base_weight - 0.05, base_weight + 0.05)
                else:  # Timing neurons
                    # Inverse scale - simpler patterns need more timing precision
                    base_weight = 0.15 + 0.05 * (1/pattern_complexity)
                    weight = np.random.uniform(base_weight - 0.05, base_weight + 0.05)
                self.network.connect(hidden_id, output_id, weight=weight)
        
        # Add lateral inhibition among output neurons with pattern-based weights
        for i, output_id1 in enumerate(self.output_neurons):
            pattern1 = self.morse_code[output_id1.split('_')[1]]
            for j, output_id2 in enumerate(self.output_neurons):
                if i != j:
                    pattern2 = self.morse_code[output_id2.split('_')[1]]
                    # Stronger inhibition between similar patterns
                    similarity = self._pattern_similarity(pattern1, pattern2)
                    inhibition_weight = -0.2 - 0.1 * similarity
                    self.network.connect(output_id1, output_id2, weight=inhibition_weight)
        
        # Add recurrent connections to hidden layer for temporal context
        for i, hid1 in enumerate(self.hidden_neurons):
            for j, hid2 in enumerate(self.hidden_neurons):
                if i != j and np.random.rand() < 0.3:
                    weight = np.random.uniform(-0.1, 0.2)
                    self.network.connect(hid1, hid2, weight=weight)
        
        # Add delayed connections from input to hidden
        for input_id in self.input_neurons:
            for hid in self.hidden_neurons:
                if np.random.rand() < 0.4:
                    self.network.connect(input_id, hid, 
                                       weight=np.random.uniform(0.05, 0.15),
                                       delay=3)  # Add delay parameter
    
    def _pattern_similarity(self, pattern1, pattern2):
        # Enhanced similarity measure with temporal alignment
        max_len = max(len(pattern1), len(pattern2))
        
        # Temporal alignment score
        alignment = sum(
            1 - abs(i/len(pattern1) - j/len(pattern2))
            for i,s1 in enumerate(pattern1)
            for j,s2 in enumerate(pattern2)
            if s1 == s2
        ) / (max_len * 0.5)
        
        # Length difference penalty
        len_penalty = abs(len(pattern1) - len(pattern2)) * 0.2
        
        return np.clip(alignment - len_penalty, 0, 1)

    def train(self, letter, num_epochs=23, learning_rate=0.025, l2_reg=0.1):
        """Train the network to recognize a specific Morse code letter.
        
        Args:
            letter (str): The letter to train on.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Base learning rate.
            l2_reg (float): L2 regularization strength.
        """
        if letter not in self.morse_code:
            raise ValueError(f"Letter {letter} not in defined Morse code patterns")
            
        morse_pattern = self.morse_code[letter]
        times, signal = morse_to_signal(morse_pattern)
        dt = times[1] - times[0]
        
        # Training loop with adaptive learning and regularization
        for epoch in range(num_epochs):
            self.network.time = 0.0
            
            # Adaptive learning rate schedule
            if epoch < num_epochs * 0.2:  # Warmup phase
                current_lr = learning_rate * (epoch / (num_epochs * 0.2))
            else:  # Cosine decay with restarts
                cycle_length = num_epochs * 0.4
                cycle_position = (epoch - num_epochs * 0.2) % cycle_length
                cosine_decay = 0.5 * (1 + np.cos(np.pi * cycle_position / cycle_length))
                current_lr = learning_rate * (0.1 + 0.9 * cosine_decay)
            
            # Process the signal in small batches for stability
            batch_size = 10
            for i in range(0, len(times), batch_size):
                batch_times = times[i:i + batch_size]
                batch_signals = signal[i:i + batch_size]
                
                # Track activations for intrinsic plasticity
                hidden_activations = {hid: [] for hid in self.hidden_neurons}
                output_activations = {out: [] for out in self.output_neurons}
                
                for t, signal_value in zip(batch_times, batch_signals):
                    # Enhanced temporal encoding using all three input neuron types
                    for j, input_id in enumerate(self.input_neurons):
                        input_group = j // (len(self.input_neurons) // 3)
                        
                        if input_group == 0:  # Direct temporal neurons
                            # Direct mapping of signal
                            shifted_signal = signal_value
                        elif input_group == 1:  # Pattern detection neurons
                            # Use cosine basis functions
                            phase_shift = j * 0.2 * np.pi
                            shifted_signal = signal_value * np.cos(phase_shift)
                        else:  # Integration neurons
                            # Use exponential decay for temporal integration
                            decay_factor = np.exp(-0.5 * j)
                            shifted_signal = signal_value * decay_factor
                        
                        external_inputs = {input_id: shifted_signal * 0.8}
                        self.network.update(dt, external_inputs)
                    
                    # Record activations for intrinsic plasticity
                    for hid in self.hidden_neurons:
                        hidden_activations[hid].append(
                            self.network.neurons[hid].signal(self.network.time)
                        )
                    for out in self.output_neurons:
                        output_activations[out].append(
                            self.network.neurons[out].signal(self.network.time)
                        )
                
                # Update weights with adaptive STDP-like rule
                for (source_id, target_id), weight in self.network.connection_weights.items():
                    # Calculate spike timing difference
                    phase_diff = (self.network.neurons[source_id].phase - 
                                self.network.neurons[target_id].phase)
                    
                    # STDP-inspired weight update
                    if phase_diff > 0:  # Source leads target
                        weight_update = 0.1 * np.exp(-phase_diff/0.5)
                    else:  # Source follows target
                        weight_update = -0.1 * np.exp(phase_diff/0.5)
                    
                    # Apply update with homeostasis
                    new_weight = weight + current_lr * weight_update
                    new_weight = np.clip(new_weight, -1.0, 1.0)
                    self.network.connection_weights[source_id, target_id] = new_weight
            
            # Apply intrinsic plasticity adjustments at epoch end
            self._adjust_intrinsic_properties(hidden_activations, output_activations)
    
    def _adjust_intrinsic_properties(self, hidden_activations, output_activations):
        """Adjust intrinsic properties of neurons based on their activation history."""
        target_mean_activity = 0.5  # Desired mean activity level
        
        # Adjust hidden neuron properties
        for hid in self.hidden_neurons:
            mean_activity = np.mean(np.abs(hidden_activations[hid]))
            if mean_activity > 0:  # Avoid division by zero
                # Adjust amplitude to normalize activity
                self.network.neurons[hid].amplitude *= target_mean_activity / mean_activity
                
                # Slightly adjust frequency based on activity
                activity_ratio = mean_activity / target_mean_activity
                freq_adjustment = np.clip(activity_ratio - 1, -0.1, 0.1)
                self.network.neurons[hid].natural_freq *= (1 + 0.01 * freq_adjustment)
        
        # Adjust output neuron properties
        for out in self.output_neurons:
            mean_activity = np.mean(np.abs(output_activations[out]))
            if mean_activity > 0:
                # More conservative adjustments for output layer
                self.network.neurons[out].amplitude *= np.sqrt(target_mean_activity / mean_activity)

    def recognize(self, morse_pattern):
        """Test the network's ability to recognize a Morse code pattern using enhanced decoding."""
        times, signal = morse_to_signal(morse_pattern)
        dt = times[1] - times[0]
        
        self.network.time = 0.0
        output_history = {letter: [] for letter in self.morse_code.keys()}
        
        # Track additional metrics for enhanced decoding
        phase_coherence = {letter: [] for letter in self.morse_code.keys()}
        temporal_patterns = {letter: [] for letter in self.morse_code.keys()}
        hidden_states = []
        
        # Process signal with multi-scale temporal integration
        window_sizes = [3, 5, 7]  # Multiple window sizes for different temporal scales
        for window_size in window_sizes:
            self.network.time = 0.0  # Reset time for each scale
            
            for i in range(0, len(times) - window_size + 1):
                window_signals = signal[i:i + window_size]
                window_mean = np.mean(window_signals)
                
                # Apply signal to input neurons with enhanced encoding
                for j, input_id in enumerate(self.input_neurons):
                    input_group = j // (len(self.input_neurons) // 3)
                    
                    if input_group == 0:  # Direct temporal neurons
                        shifted_signal = window_mean
                    elif input_group == 1:  # Pattern detection neurons
                        phase_shift = j * 0.2 * np.pi
                        shifted_signal = window_mean * np.cos(phase_shift)
                    else:  # Integration neurons
                        decay_factor = np.exp(-0.5 * j)
                        shifted_signal = window_mean * decay_factor
                    
                    external_inputs = {input_id: shifted_signal * 0.8}
                    self.network.update(dt, external_inputs)
                
                # Record hidden layer state
                hidden_state = [
                    self.network.neurons[hid].signal(self.network.time)
                    for hid in self.hidden_neurons
                ]
                hidden_states.append(hidden_state)
                
                # Record output activities and phase relationships
                for letter in self.morse_code.keys():
                    output_id = f"output_{letter}"
                    output_neuron = self.network.neurons[output_id]
                    
                    # Record amplitude
                    activity = abs(output_neuron.signal(self.network.time))
                    output_history[letter].append(activity)
                    
                    # Calculate phase coherence with hidden layer
                    phase_diffs = [
                        np.cos(output_neuron.phase - self.network.neurons[hid].phase)
                        for hid in self.hidden_neurons
                    ]
                    phase_coherence[letter].append(np.mean(phase_diffs))
                    
                    # Record temporal pattern match
                    pattern = self.morse_code[letter]
                    expected_duration = len(pattern) * dt * 10  # Approximate expected duration
                    temporal_match = np.exp(-abs(i*dt - expected_duration) / expected_duration)
                    temporal_patterns[letter].append(temporal_match)
        
        # Compute enhanced recognition scores
        recognition_scores = {}
        for letter in self.morse_code.keys():
            # 1. Activity-based score (weighted by temporal position)
            activities = output_history[letter]
            temporal_weights = np.linspace(0.5, 1.0, len(activities))
            activity_score = np.average(activities, weights=temporal_weights)
            
            # 2. Phase coherence score
            coherence_score = np.mean(phase_coherence[letter])
            
            # 3. Temporal pattern match score
            pattern_score = np.mean(temporal_patterns[letter])
            
            # 4. Hidden layer state analysis
            hidden_states_array = np.array(hidden_states)
            output_id = f"output_{letter}"
            
            # Calculate correlation between hidden states and output activity
            output_activities = np.array(output_history[letter])
            correlations = []
            for h in range(len(self.hidden_neurons)):
                hidden_activities = hidden_states_array[:, h]
                if len(hidden_activities) == len(output_activities):
                    corr = np.corrcoef(hidden_activities, output_activities)[0, 1]
                    correlations.append(0 if np.isnan(corr) else corr)
            
            hidden_correlation_score = np.mean(correlations) if correlations else 0
            
            # Combine scores with learned weights
            # These weights could be optimized through meta-learning
            recognition_scores[letter] = (
                0.4 * activity_score +
                0.3 * coherence_score +
                0.2 * pattern_score +
                0.1 * hidden_correlation_score
            )
        
        return recognition_scores

    def visualize_signal(self, morse_pattern):
        """Visualize a Morse code signal."""
        # Validate pattern exists in dataset
        valid_patterns = set(self.morse_code.values())
        if morse_pattern not in valid_patterns:
            raise ValueError(f"Pattern {morse_pattern} not in loaded dataset")
        
        times, signal = morse_to_signal(morse_pattern)
        
        plt.figure(figsize=(10, 3))
        plt.plot(times, signal)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Morse Code Signal: {morse_pattern}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def visualize_network_response(self, morse_pattern):
        """Visualize how the network responds to a Morse code pattern."""
        times, signal = morse_to_signal(morse_pattern)
        dt = times[1] - times[0]
        
        # Reset network time
        self.network.time = 0.0
        
        # Storage for neuron activities
        input_activity = []
        output_activity = {letter: [] for letter in self.morse_code.keys()}
        
        # Run the simulation
        for t, signal_value in zip(times, signal):
            # Create inputs
            external_inputs = {id: signal_value * 0.5 for id in self.input_neurons}
            
            # Update network
            self.network.update(dt, external_inputs)
            
            # Record activities
            input_activity.append(signal_value)
            for letter in self.morse_code.keys():
                output_id = f"output_{letter}"
                activity = self.network.neurons[output_id].signal(self.network.time)
                output_activity[letter].append(activity)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        # Plot input signal
        plt.subplot(2, 1, 1)
        plt.plot(times, input_activity)
        plt.title('Input Signal')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # Plot output neuron responses
        plt.subplot(2, 1, 2)
        for letter, activities in output_activity.items():
            plt.plot(times, activities, label=f"Letter {letter}")
        
        plt.title('Output Neuron Responses')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def evaluate_network(recognizer, test_letters):
    """
    Evaluate network performance across multiple letters.
    Returns accuracy score.
    """
    correct_predictions = 0
    total_predictions = 0
    
    for test_letter in test_letters:
        test_pattern = recognizer.morse_code[test_letter]
        results = recognizer.recognize(test_pattern)
        
        predicted_letter = max(results.items(), key=lambda x: x[1])[0]
        
        if predicted_letter == test_letter:
            correct_predictions += 1
        total_predictions += 1
    
    return correct_predictions / total_predictions

def evaluate_configuration(params, letters, n_splits=3, verbose=False):
    """
    Evaluate a configuration using k-fold cross-validation.
    """
    num_input_neurons, num_hidden_neurons, epochs, learning_rate = params
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(letters)):
        train_letters = [letters[i] for i in train_idx]
        test_letters = [letters[i] for i in test_idx]
        
        # Create and train network
        recognizer = MorseCodeRecognizer(
            num_input_neurons=int(num_input_neurons),
            num_hidden_neurons=int(num_hidden_neurons)
        )
        
        # Train on training letters
        for letter in train_letters:
            recognizer.train(letter, num_epochs=int(epochs), learning_rate=learning_rate)
        
        # Evaluate on test letters
        accuracy = evaluate_network(recognizer, test_letters)
        accuracies.append(accuracy)
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    if verbose:
        print(f"Config: i={int(num_input_neurons)}, h={int(num_hidden_neurons)}, "
              f"e={int(epochs)}, lr={learning_rate:.4f} → "
              f"acc={mean_accuracy:.4f}±{std_accuracy:.4f}")
    
    return -mean_accuracy  # Return negative accuracy for minimization

def find_optimal_parameters(n_calls=15, n_jobs=-1):
    """
    Find optimal hyperparameters using Bayesian optimization with cross-validation.
    """
    # Define the search space
    space = [
        Integer(4, 8, name='num_input_neurons'),
        Integer(8, 16, name='num_hidden_neurons'),
        Integer(10, 25, name='epochs'),
        Real(0.01, 0.05, name='learning_rate')  # Narrower learning rate range
    ]
    
    # Create partial function for evaluation
    objective = partial(evaluate_configuration, 
                       letters=list(morse_code.keys()),
                       verbose=True)
    
    print(f"Starting Bayesian optimization with {n_calls} iterations...")
    print("Using 3-fold cross-validation for more reliable estimates...")
    
    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_jobs=n_jobs,
        noise=0.05,  # Reduced noise assumption
        verbose=False
    )
    
    # Extract best parameters
    best_params = {
        'num_input_neurons': int(result.x[0]),
        'num_hidden_neurons': int(result.x[1]),
        'epochs': int(result.x[2]),
        'learning_rate': float(result.x[3])
    }
    
    print("\nOptimal parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best cross-validated accuracy: {-result.fun:.4f}")
    
    return best_params

def run_morse_code_experiment():
    # Find optimal parameters using Bayesian optimization
    optimal_params = find_optimal_parameters(n_calls=20)
    
    # Create recognizer with optimal parameters
    recognizer = MorseCodeRecognizer(
        num_input_neurons=optimal_params['num_input_neurons'],
        num_hidden_neurons=optimal_params['num_hidden_neurons']
    )
    
    # Train on all letters with optimal parameters
    print("\nTraining final model with optimal parameters...")
    for letter in recognizer.morse_code.keys():
        recognizer.train(
            letter,
            num_epochs=optimal_params['epochs'],
            learning_rate=optimal_params['learning_rate']
        )
    
    # Test recognition on all letters
    total_correct = 0
    total_tests = 0
    
    print("\nTesting all letters...")
    for test_letter in recognizer.morse_code.keys():
        test_pattern = recognizer.morse_code[test_letter]
        results = recognizer.recognize(test_pattern)
        predicted_letter = max(results.items(), key=lambda x: x[1])[0]
        
        correct = predicted_letter == test_letter
        total_correct += int(correct)
        total_tests += 1
        
        print(f"{test_letter} ({test_pattern}) → {predicted_letter}" + 
              (" ✓" if correct else " ✗"))
    
    print(f"\nFinal accuracy: {total_correct/total_tests:.4f}")

if __name__ == "__main__":
    run_morse_code_experiment()
