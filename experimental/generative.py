"""
NOTE: This file contains code from an earlier experimental phase of the project,
focused on simple token-based code generation. It is not currently part of 
the main music generation pipeline but is retained for reference.
"""
import numpy as np
import re
from sklearn.model_selection import train_test_split
from itertools import product
from collections import Counter
from scipy.spatial.distance import cosine, euclidean, cityblock
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Enhanced tokenization function
def tokenize_code(code_snippet):
    """
    Tokenizes a code snippet into individual tokens, handling operators, delimiters, and quoted strings.
    """
    pattern = r"\'[^\']*\'|\"[^\"]*\"|[\w']+|[()+\-*/:,=<>!{}[\]]"
    tokens = re.findall(pattern, code_snippet)
    return tokens

# Example code snippets (simple Python examples)
code_snippets = [
    "def add(a, b): return a + b",
    "def subtract(a, b): return a - b",
    "for i in range(10): print(i)",
    "if x > 0: print('positive') else: print('non-positive')",
    "def multiply(a, b): return a * b",
    "def divide(a, b): return a / b if b != 0 else None",
    "def power(a, b): return a ** b",
    "def modulus(a, b): return a % b",
    "def floor(a, b): return a // b",
    "def exponent(a, b): return a ** b",
    "def logarithm(a, b): return math.log(a, b)",
    "def square_root(a): return a ** 0.5",
    "def cube_root(a): return a ** (1/3)",
    "def sine(a): return math.sin(a)",
    "def cosine(a): return math.cos(a)",
    "def tangent(a): return math.tan(a)",
    "def cotangent(a): return 1 / math.tan(a)",
    "def secant(a): return 1 / math.cos(a)",
    "def cosecant(a): return 1 / math.sin(a)",
    "def arcsine(a): return math.asin(a)",
    "def arccosine(a): return math.acos(a)",
    "def arctangent(a): return math.atan(a)",
    "def arccotangent(a): return math.atan(1 / a)",
    "def hyperbolic_sine(a): return (math.exp(a) - math.exp(-a)) / 2",
    "def hyperbolic_cosine(a): return (math.exp(a) + math.exp(-a)) / 2",
    "def hyperbolic_tangent(a): return (math.exp(a) - math.exp(-a)) / (math.exp(a) + math.exp(-a))",
    "def hyperbolic_cotangent(a): return 1 / math.tanh(a)",
    "def hyperbolic_secant(a): return 2 / (math.exp(a) + math.exp(-a))",
    "def hyperbolic_cosecant(a): return 2 / (math.exp(a) - math.exp(-a))",
    "def exponential(a): return math.exp(a)",
    "def natural_logarithm(a): return math.log(a)",
]

# Tokenize all code snippets
tokenized_snippets = [tokenize_code(snippet) for snippet in code_snippets]

# Count token frequencies
token_counts = Counter(token for snippet in tokenized_snippets for token in snippet)

# Define frequency threshold
frequency_threshold = 2

# Replace rare tokens with '<UNK>'
tokenized_snippets_replaced = []
for snippet in tokenized_snippets:
    new_snippet = [token if token_counts[token] >= frequency_threshold else '<UNK>' for token in snippet]
    tokenized_snippets_replaced.append(new_snippet)

# Rebuild vocabulary
vocab = set(token for snippet in tokenized_snippets_replaced for token in snippet)
vocab = sorted(vocab)
vocab.append('<PAD>')  # Padding token
# '<UNK>' is already included from replacements
token_to_idx = {token: idx for idx, token in enumerate(vocab)}
idx_to_token = {idx: token for token, idx in token_to_idx.items()}

print("Updated Vocabulary Size:", len(vocab))
print("Sample Vocabulary:", vocab[:20])

# Parameters
window_size = 3  # Number of previous tokens to consider
stride = 1       # Step size for the sliding window

X_sequences = []
y_targets = []

for snippet in tokenized_snippets_replaced:
    # Add padding at the beginning
    padded_snippet = ['<PAD>'] * window_size + snippet
    for i in range(len(snippet)):
        input_seq = padded_snippet[i:i + window_size]
        target_token = snippet[i]
        X_sequences.append(input_seq)
        y_targets.append(target_token)

print("Number of sequences:", len(X_sequences))
print("First sequence:", X_sequences[0], "->", y_targets[0])

# Convert tokens to indices
def encode_sequence(sequence, token_to_idx):
    return [token_to_idx.get(token, token_to_idx['<UNK>']) for token in sequence]

X_encoded = [encode_sequence(seq, token_to_idx) for seq in X_sequences]
y_encoded = [token_to_idx.get(token, token_to_idx['<UNK>']) for token in y_targets]

print("Encoded first sequence:", X_encoded[0], "->", y_encoded[0])

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Define syntax rules and token types for syntax-aware generation
syntax_rules = {
    'def': ['identifier'],
    '(': ['identifier', ')'],
    'identifier': ['(', ')', '=', '+', '-', '*', '/', '**', '%', '//', ':', ','],
    ':': ['indent', 'return', 'pass', 'if', 'for', 'while', 'print', 'expression'],
    'return': ['expression'],
    'if': ['expression'],
    'for': ['expression'],
    'while': ['expression'],
    'print': ['expression'],
    'expression': ['identifier', 'literal', 'operator', '(', ')'],
    # Add more rules as needed
}

# Map tokens to types
token_types = {
    'def': 'def',
    'return': 'return',
    'if': 'if',
    'for': 'for',
    'while': 'while',
    'print': 'print',
    '(': '(',
    ')': ')',
    '=': '=',
    '+': 'operator',
    '-': 'operator',
    '*': 'operator',
    '/': 'operator',
    '**': 'operator',
    '%': 'operator',
    '//': 'operator',
    ':': ':',
    ',': ',',
    '<UNK>': 'identifier',
    '<PAD>': '<PAD>',
    # All other tokens are considered 'identifier' or 'literal'
}

def get_token_type(token):
    return token_types.get(token, 'identifier')

class ResonantNode:
    """
    Represents a single resonant node with specific frequency, phase, and amplitude.
    """
    def __init__(self, freq=1.0, phase=0.0, amplitude=1.0, node_id=0):
        self.freq = freq
        self.phase = phase
        self.amplitude = amplitude
        self.node_id = node_id

    def compute(self, time_steps):
        """
        Computes the signal output of the node over the given time steps.
        """
        return self.amplitude * np.sin(2 * np.pi * self.freq * time_steps + self.phase)

    def retune(self, new_freq, new_phase):
        """
        Retunes the node's frequency and phase.
        """
        self.freq = new_freq
        self.phase = new_phase

class ResonantNetwork:
    """
    Represents a network of resonant nodes organized per class with specific harmonic ratios.
    """
    def __init__(self, num_nodes, base_freq, class_harmonic_ratios, initial_phase=0.0):
        """
        Initializes the resonant network.

        Parameters:
        - num_nodes: Number of nodes per class.
        - base_freq: Base frequency for the network.
        - class_harmonic_ratios: Dictionary mapping class labels to their harmonic ratios.
        - initial_phase: Initial phase for all nodes.
        """
        self.num_nodes = num_nodes
        self.base_freq = base_freq
        self.class_harmonic_ratios = class_harmonic_ratios
        self.initial_phase = initial_phase
        self.class_outputs = {}

        # Initialize nodes with harmonic intervals and initial phase
        for class_label, ratios in class_harmonic_ratios.items():
            if len(ratios) != num_nodes:
                raise ValueError(f"Number of harmonic ratios for class {class_label} does not match num_nodes.")
            class_nodes = [
                ResonantNode(freq=base_freq * ratio, phase=self.initial_phase, amplitude=1.0, node_id=i+1)
                for i, ratio in enumerate(ratios)
            ]
            self.class_outputs[class_label] = class_nodes

    def compute_class_signals(self, class_label, time_steps):
        """
        Computes the signals for all nodes of a given class over the specified time steps.

        Returns:
        - A NumPy array of shape (len(time_steps), num_nodes)
        """
        if class_label not in self.class_outputs:
            raise ValueError(f"Class label {class_label} not found in the network.")

        class_nodes = self.class_outputs[class_label]
        # Vectorized computation for efficiency
        signals = np.array([node.compute(time_steps) for node in class_nodes]).T  # Shape: (len(time_steps), num_nodes)
        return signals

    def classify_input(self, input_data, time_steps, similarity_metric='cosine'):
        """
        Classifies the input data by computing dissonance scores against each class.

        Parameters:
        - input_data: NumPy array of shape (num_nodes, len(time_steps))
        - time_steps: NumPy array of time steps.
        - similarity_metric: String indicating the similarity metric ('cosine', 'euclidean', 'manhattan').

        Returns:
        - Predicted class label with the minimum dissonance.
        """
        dissonance_scores = {}
        similarity_functions = {
            'cosine': cosine,
            'euclidean': euclidean,
            'manhattan': cityblock
        }

        if similarity_metric not in similarity_functions:
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

        similarity_func = similarity_functions[similarity_metric]

        # Flatten input data: shape (num_nodes * len(time_steps),)
        input_flat = input_data.flatten()

        for class_label, nodes in self.class_outputs.items():
            # Compute class signals
            class_signals = self.compute_class_signals(class_label, time_steps)
            class_flat = class_signals.flatten()

            # Compute dissonance using the selected similarity metric
            if similarity_metric == 'cosine':
                # Handle cases where vectors might be zero vectors
                norm_input = np.linalg.norm(input_flat)
                norm_class = np.linalg.norm(class_flat)
                if norm_input == 0 or norm_class == 0:
                    dissonance = 1.0  # Maximum dissonance for zero vectors
                else:
                    dissonance = cosine(input_flat, class_flat)
            else:
                dissonance = similarity_func(input_flat, class_flat)

            dissonance_scores[class_label] = dissonance

        # Select the class with the minimum dissonance
        predicted_class = min(dissonance_scores, key=dissonance_scores.get)
        return predicted_class

    def update_harmonic_ratios(self, new_data, learning_rate=0.1):
        """
        Updates harmonic ratios based on new data.

        Parameters:
        - new_data: Tuple of (input_sequence, target_token)
        - learning_rate: Rate at which to adjust harmonic ratios.
        """
        input_seq, target_token = new_data
        time_steps = np.linspace(0, 1, 100)
        input_signals = convert_features_to_signals_dynamic(input_seq, time_steps, max_nodes=self.num_nodes)
        input_signals_classify = input_signals.T

        # Calculate dissonance with current harmonic ratios
        predicted_class = self.classify_input(input_signals_classify, time_steps, similarity_metric='cosine')
        if predicted_class != target_token:
            # Adjust harmonic ratios for the target class to reduce dissonance
            current_ratios = self.class_harmonic_ratios.get(target_token, [1.0] * self.num_nodes)
            # Simple adjustment: Increase frequencies slightly
            adjusted_ratios = [ratio + learning_rate for ratio in current_ratios]
            self.class_harmonic_ratios[target_token] = adjusted_ratios
            # Update the nodes for the target class
            class_nodes = [
                ResonantNode(freq=self.base_freq * ratio, phase=self.initial_phase, amplitude=1.0, node_id=i+1)
                for i, ratio in enumerate(adjusted_ratios)
            ]
            self.class_outputs[target_token] = class_nodes

class HierarchicalResonantNetwork:
    """
    Represents a hierarchical resonant network with multiple levels to capture local and global code structures.
    """
    def __init__(self, num_levels, num_nodes_per_level, base_freq, initial_phase=0.0):
        """
        Initializes the hierarchical resonant network.

        Parameters:
        - num_levels: Number of hierarchical levels.
        - num_nodes_per_level: List specifying the number of nodes at each level.
        - base_freq: Base frequency for the network.
        - initial_phase: Initial phase for all nodes.
        """
        self.num_levels = num_levels
        self.network_levels = []

        for level in range(num_levels):
            num_nodes = num_nodes_per_level[level]
            # Initialize a ResonantNetwork for each level
            network = ResonantNetwork(
                num_nodes=num_nodes,
                base_freq=base_freq * (level + 1),  # Different base freq for each level
                class_harmonic_ratios={},  # To be set during training
                initial_phase=initial_phase
            )
            self.network_levels.append(network)

def optimize_harmonic_ratios(X_train, y_train, num_nodes, base_freq, initial_phase=0.0, grid_size=3):
    """
    Optimizes harmonic ratios based on the training data using grid search.

    Parameters:
    - X_train: Training input sequences (list of lists of token indices).
    - y_train: Training target tokens (list of token indices).
    - num_nodes: Number of resonant nodes.
    - base_freq: Base frequency for the network.
    - initial_phase: Initial phase for all nodes.
    - grid_size: Number of different ratios to try per harmonic ratio.

    Returns:
    - best_class_harmonic_ratios: Dictionary mapping class labels to optimized harmonic ratios.
    """
    # Define a range of harmonic ratios to explore
    ratio_options = np.linspace(1.0, 3.0, grid_size)
    
    # Get unique class labels
    class_labels = np.unique(y_train)
    best_harmonic_ratios = {cls: [1.0]*num_nodes for cls in class_labels}
    best_total_accuracy = 0

    # Generate all possible combinations for each class
    for cls in class_labels:
        print(f"Optimizing harmonic ratios for token '{idx_to_token[cls]}' (Class {cls})...")
        best_cls_accuracy = 0
        best_cls_ratios = None
        cls_ratio_combinations = list(product(ratio_options, repeat=num_nodes))
        for ratios in cls_ratio_combinations:
            # Set harmonic ratios for all classes, only changing current class
            temp_harmonic_ratios = {}
            for other_cls in class_labels:
                if other_cls == cls:
                    temp_harmonic_ratios[other_cls] = ratios
                else:
                    # Use existing best ratios for other classes
                    temp_harmonic_ratios[other_cls] = best_harmonic_ratios[other_cls]
            # Initialize network
            network = ResonantNetwork(
                num_nodes=num_nodes,
                base_freq=base_freq,
                class_harmonic_ratios=temp_harmonic_ratios,
                initial_phase=initial_phase
            )
            # Evaluate on training data
            correct = 0
            for input_seq, label in zip(X_train, y_train):
                # Convert input sequence to signals
                time_steps = np.linspace(0, 1, 100)
                input_signals = convert_features_to_signals_dynamic(input_seq, time_steps, max_nodes=num_nodes)
                # Transpose to shape: (num_nodes, len(time_steps))
                input_signals_classify = input_signals.T
                predicted = network.classify_input(input_signals_classify, time_steps, similarity_metric='cosine')
                if predicted == label:
                    correct += 1
            accuracy = correct / len(y_train)
            if accuracy > best_cls_accuracy:
                best_cls_accuracy = accuracy
                best_cls_ratios = ratios
        best_harmonic_ratios[cls] = list(best_cls_ratios)
        best_total_accuracy += best_cls_accuracy
        print(f"Best ratios for token '{idx_to_token[cls]}': {best_cls_ratios} with training accuracy: {best_cls_accuracy * 100:.2f}%")

    average_accuracy = best_total_accuracy / len(class_labels)
    print(f"\nAverage Training Accuracy across tokens: {average_accuracy * 100:.2f}%")
    return best_harmonic_ratios

def convert_features_to_signals_dynamic(features, time_steps, phase_shift=0.1, max_nodes=10):
    """
    Converts feature vectors into signal representations with dynamic node allocation.

    Parameters:
    - features: List of feature values (token indices).
    - time_steps: NumPy array of time steps.
    - phase_shift: Phase shift multiplier for signal differentiation.
    - max_nodes: Maximum number of nodes to allocate.

    Returns:
    - A NumPy array of shape (len(time_steps), num_nodes)
    """
    num_features = len(features)
    num_nodes = min(num_features, max_nodes)
    signals = []

    for i in range(num_nodes):
        feature = features[i]
        # Generate signal
        signal_sin = np.sin(2 * np.pi * feature * time_steps + (i * phase_shift))
        # Normalize signal
        signal_sin = (signal_sin - np.mean(signal_sin)) / (np.std(signal_sin) + 1e-8)
        signals.append(signal_sin)

    # Pad signals if fewer than max_nodes
    while len(signals) < max_nodes:
        signals.append(np.zeros_like(time_steps))

    return np.array(signals).T

def train_resonant_network(X_train, y_train, num_nodes, base_freq, initial_phase=0.0, grid_size=3):
    """
    Trains the resonant network by optimizing harmonic ratios.

    Parameters:
    - X_train: Training input sequences (list of lists of token indices).
    - y_train: Training target tokens (list of token indices).
    - num_nodes: Number of resonant nodes.
    - base_freq: Base frequency for the network.
    - initial_phase: Initial phase for all nodes.
    - grid_size: Number of different ratios to try per harmonic ratio.

    Returns:
    - network: Trained ResonantNetwork instance.
    - optimized_ratios: Dictionary of optimized harmonic ratios.
    """
    # Optimize harmonic ratios
    optimized_ratios = optimize_harmonic_ratios(
        X_train, y_train, num_nodes, base_freq, initial_phase, grid_size
    )

    # Initialize the resonant network with optimized ratios
    network = ResonantNetwork(
        num_nodes=num_nodes,
        base_freq=base_freq,
        class_harmonic_ratios=optimized_ratios,
        initial_phase=initial_phase
    )

    return network, optimized_ratios

def train_hierarchical_network(X_train, y_train, num_levels, num_nodes_per_level, base_freq, initial_phase=0.0, grid_size=3):
    """
    Trains the hierarchical resonant network by optimizing harmonic ratios at each level.

    Returns:
    - hierarchical_network: Trained HierarchicalResonantNetwork instance.
    """
    hierarchical_network = HierarchicalResonantNetwork(
        num_levels=num_levels,
        num_nodes_per_level=num_nodes_per_level,
        base_freq=base_freq,
        initial_phase=initial_phase
    )

    for level in range(num_levels):
        print(f"\n--- Training Level {level + 1} ---")
        num_nodes = num_nodes_per_level[level]
        # Optimize harmonic ratios for this level
        network, optimized_ratios = train_resonant_network(
            X_train, y_train, num_nodes, base_freq * (level + 1), initial_phase, grid_size
        )
        # Update the class harmonic ratios
        hierarchical_network.network_levels[level] = network

    return hierarchical_network

def evaluate_resonant_network(network, X_test, y_test, similarity_metric='cosine'):
    """
    Evaluates the resonant network on the test set.

    Parameters:
    - network: Trained ResonantNetwork instance.
    - X_test: Testing input sequences (list of lists of token indices).
    - y_test: Testing target tokens (list of token indices).
    - similarity_metric: Similarity metric to use for classification.

    Returns:
    - accuracy: Classification accuracy.
    - conf_matrix: Confusion matrix.
    - class_report: Classification report.
    """
    y_pred = []
    for input_seq in X_test:
        time_steps = np.linspace(0, 1, 100)
        input_signals = convert_features_to_signals_dynamic(input_seq, time_steps, max_nodes=network.num_nodes)
        input_signals_classify = input_signals.T
        predicted = network.classify_input(input_signals_classify, time_steps, similarity_metric=similarity_metric)
        y_pred.append(predicted)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Handle target_names to include only tokens present in y_test and y_pred
    present_tokens = sorted(set(y_test) | set(y_pred))
    target_names = [idx_to_token[i] for i in present_tokens]
    class_report = classification_report(y_test, y_pred, target_names=target_names)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    return accuracy, conf_matrix, class_report

def compute_attention_weights(input_seq):
    """
    Computes attention weights for the input sequence.

    For simplicity, we'll use a basic positional attention where more recent tokens have higher weights.
    """
    seq_length = len(input_seq)
    weights = np.linspace(1.0, 0.5, seq_length)  # Decreasing weights for earlier tokens
    # Normalize weights
    weights /= np.sum(weights)
    return weights

def convert_features_to_signals_attention(features, time_steps, attention_weights, phase_shift=0.1, max_nodes=10):
    """
    Converts feature vectors into signal representations with attention weights.

    Parameters:
    - features: List of feature values (token indices).
    - time_steps: NumPy array of time steps.
    - attention_weights: List or array of attention weights corresponding to each feature.
    - phase_shift: Phase shift multiplier for signal differentiation.
    - max_nodes: Maximum number of nodes to allocate.

    Returns:
    - A NumPy array of shape (len(time_steps), num_nodes)
    """
    num_features = len(features)
    num_nodes = min(num_features, max_nodes)
    signals = []

    for i in range(num_nodes):
        feature = features[i]
        weight = attention_weights[i]
        # Generate signal with amplitude adjusted by attention weight
        signal_sin = weight * np.sin(2 * np.pi * feature * time_steps + (i * phase_shift))
        # Normalize signal
        signal_sin = (signal_sin - np.mean(signal_sin)) / (np.std(signal_sin) + 1e-8)
        signals.append(signal_sin)

    # Pad signals if fewer than max_nodes
    while len(signals) < max_nodes:
        signals.append(np.zeros_like(time_steps))

    return np.array(signals).T

def generate_code_syntax_aware(network, seed_tokens, num_generate=5):
    """
    Generates code based on the given seed tokens using syntax-aware generation.

    Parameters:
    - network (ResonantNetwork): The trained resonant network.
    - seed_tokens (List[str]): The initial tokens to start generation.
    - num_generate (int): Number of tokens to generate.

    Returns:
    - None: Prints the generated code.
    """
    current_tokens = seed_tokens.copy()
    print("\n--- Generating Code (Syntax-Aware) ---")
    print("Seed:", ' '.join(current_tokens), end=' ')

    for _ in range(num_generate):
        # Encode the current context
        encoded_tokens = encode_sequence(current_tokens[-window_size:], token_to_idx)
        if len(encoded_tokens) < window_size:
            encoded_tokens = [token_to_idx['<PAD>']] * (window_size - len(encoded_tokens)) + encoded_tokens

        # Convert to signals
        time_steps = np.linspace(0, 1, 100)
        input_signals = convert_features_to_signals_dynamic(encoded_tokens, time_steps, max_nodes=network.num_nodes)
        input_signals_classify = input_signals.T

        # Get possible next tokens based on syntax
        last_token = current_tokens[-1]
        last_token_type = get_token_type(last_token)
        possible_next_types = syntax_rules.get(last_token_type, ['identifier'])
        possible_next_tokens = [token for token in vocab if get_token_type(token) in possible_next_types]

        # Map possible tokens to their indices
        possible_indices = [token_to_idx[token] for token in possible_next_tokens if token in token_to_idx]

        # Predict the next token among possible options
        dissonance_scores = {}
        for idx in possible_indices:
            class_signals = network.compute_class_signals(idx, time_steps)
            class_flat = class_signals.flatten()

            # Compute dissonance
            input_flat = input_signals_classify.flatten()
            norm_input = np.linalg.norm(input_flat)
            norm_class = np.linalg.norm(class_flat)
            if norm_input == 0 or norm_class == 0:
                dissonance = 1.0
            else:
                dissonance = cosine(input_flat, class_flat)

            dissonance_scores[idx] = dissonance

        if dissonance_scores:
            # Select the token with minimum dissonance
            predicted_idx = min(dissonance_scores, key=dissonance_scores.get)
            predicted_token = idx_to_token.get(predicted_idx, '<UNK>')
        else:
            predicted_token = '<UNK>'

        print(predicted_token, end=' ')
        current_tokens.append(predicted_token)
    print()

def generate_code_with_attention(network, seed_tokens, num_generate=5):
    """
    Generates code based on the given seed tokens using attention mechanisms.

    Parameters:
    - network (ResonantNetwork): The trained resonant network.
    - seed_tokens (List[str]): The initial tokens to start generation.
    - num_generate (int): Number of tokens to generate.

    Returns:
    - None: Prints the generated code.
    """
    current_tokens = seed_tokens.copy()
    print("\n--- Generating Code (With Attention) ---")
    print("Seed:", ' '.join(current_tokens), end=' ')

    for _ in range(num_generate):
        # Encode the current context
        encoded_tokens = encode_sequence(current_tokens[-window_size:], token_to_idx)
        if len(encoded_tokens) < window_size:
            encoded_tokens = [token_to_idx['<PAD>']] * (window_size - len(encoded_tokens)) + encoded_tokens

        # Compute attention weights
        attention_weights = compute_attention_weights(encoded_tokens)

        # Convert to signals with attention
        time_steps = np.linspace(0, 1, 100)
        input_signals = convert_features_to_signals_attention(encoded_tokens, time_steps, attention_weights, max_nodes=network.num_nodes)
        input_signals_classify = input_signals.T

        # Predict the next token
        predicted_idx = network.classify_input(input_signals_classify, time_steps, similarity_metric='cosine')
        predicted_token = idx_to_token.get(predicted_idx, '<UNK>')

        print(predicted_token, end=' ')
        current_tokens.append(predicted_token)
    print()

def main():
    # Parameters
    num_levels = 2
    num_nodes_per_level = [6, 6]  # Adjust nodes per level
    base_freq = 1.0
    initial_phase = 0.0
    grid_size = 3  # Adjust based on computational resources

    # Train the hierarchical resonant network
    print("\n--- Training Hierarchical Resonant Network ---")
    hierarchical_network = train_hierarchical_network(
        X_train, y_train, num_levels, num_nodes_per_level, base_freq, initial_phase, grid_size
    )

    # Evaluate on the test set using the top level
    print("\n--- Evaluating Hierarchical Resonant Network ---")
    top_level_network = hierarchical_network.network_levels[-1]
    accuracy, conf_matrix, class_report = evaluate_resonant_network(
        top_level_network, X_test, y_test, similarity_metric='cosine'
    )

    # Continuous Learning with New Data
    print("\n--- Continuous Learning ---")
    new_data_batch = list(zip(X_test, y_test))
    for data in new_data_batch:
        top_level_network.update_harmonic_ratios(data)

    # Evaluate after continuous learning
    print("\n--- Evaluating After Continuous Learning ---")
    accuracy, conf_matrix, class_report = evaluate_resonant_network(
        top_level_network, X_test, y_test, similarity_metric='cosine'
    )

    # Example Code Generation with Syntax Awareness and Attention
    seed = ['def', 'add', '(']
    generate_code_syntax_aware(top_level_network, seed_tokens=seed, num_generate=10)
    generate_code_with_attention(top_level_network, seed_tokens=seed, num_generate=10)

    # Optional: Save the trained hierarchical network
    with open('hierarchical_resonant_network_codegen.pkl', 'wb') as f:
        pickle.dump({
            'num_levels': hierarchical_network.num_levels,
            'num_nodes_per_level': hierarchical_network.network_levels,
            'base_freq': base_freq,
            'initial_phase': initial_phase,
            # Add more attributes if needed
        }, f)
    print("\nTrained hierarchical resonant network saved to 'hierarchical_resonant_network_codegen.pkl'.")

if __name__ == "__main__":
    main()
