import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.spatial.distance import cosine, euclidean, cityblock
from itertools import product

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
                ResonantNode(freq=base_freq * ratio, phase=self.initial_phase, node_id=i+1)
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

def convert_features_to_signals(features, time_steps, phase_shift=0.1):
    """
    Converts feature vectors into signal representations using sine and cosine functions.

    Parameters:
    - features: NumPy array of feature values.
    - time_steps: NumPy array of time steps.
    - phase_shift: Phase shift multiplier for signal differentiation.

    Returns:
    - A NumPy array of shape (len(time_steps), num_nodes)
    """
    signals = []
    num_features = len(features)
    num_nodes = 6  # Must match the number of nodes in the network

    # Define how many signals to generate per feature
    # For 4 features and 6 nodes, we can generate:
    # - First two features: sine and cosine
    # - Last two features: sine only
    for i, feature in enumerate(features):
        if i < 2:
            # Generate sine and cosine signals
            signal_sin = np.sin(2 * np.pi * feature * time_steps + (i * phase_shift))
            signal_cos = np.cos(2 * np.pi * feature * time_steps + (i * phase_shift))
            # Normalize signals
            signal_sin = (signal_sin - np.mean(signal_sin)) / (np.std(signal_sin) + 1e-8)
            signal_cos = (signal_cos - np.mean(signal_cos)) / (np.std(signal_cos) + 1e-8)
            signals.append(signal_sin)
            signals.append(signal_cos)
        else:
            # Generate sine signal only
            signal_sin = np.sin(2 * np.pi * feature * time_steps + (i * phase_shift))
            # Normalize signal
            signal_sin = (signal_sin - np.mean(signal_sin)) / (np.std(signal_sin) + 1e-8)
            signals.append(signal_sin)

    # Ensure that the number of signals matches num_nodes
    assert len(signals) == num_nodes, f"Number of signals ({len(signals)}) does not match number of nodes ({num_nodes})"

    return np.array(signals).T  # Shape: (len(time_steps), num_nodes)

def plot_overview_single_window(time_steps, initial_harmonic, input_signals, retuned_harmonic, dissonance_values, true_class, predicted_class, accuracy, conf_matrix, class_labels):
    """
    Plots the overview with all required subplots in a single figure.

    Parameters:
    - time_steps: NumPy array of time steps.
    - initial_harmonic: NumPy array of initial harmonic signals (len(time_steps), num_nodes)
    - input_signals: NumPy array of input signals (len(time_steps), num_nodes)
    - retuned_harmonic: NumPy array of retuned harmonic signals (len(time_steps), num_nodes)
    - dissonance_values: List of dissonance values over time.
    - true_class: True class label.
    - predicted_class: Predicted class label.
    - accuracy: Classification accuracy.
    - conf_matrix: Confusion matrix.
    - class_labels: List or array of unique class labels.
    """
    fig, axs = plt.subplots(5, 1, figsize=(18, 25))

    # Subplot 1: Initial Harmonic State
    ax = axs[0]
    for i in range(initial_harmonic.shape[1]):
        ax.plot(time_steps, initial_harmonic[:, i], label=f'Node {i+1}')
    ax.set_title('Initial Harmonic State (Class 0)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)

    # Subplot 2: Input Signals (Dissonant State)
    ax = axs[1]
    for i in range(input_signals.shape[1]):
        ax.plot(time_steps, input_signals[:, i], label=f'Node {i+1}')
    ax.set_title('Input Signals (Dissonant State)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)

    # Subplot 3: Retuned Harmonic State (Predicted Class)
    ax = axs[2]
    for i in range(retuned_harmonic.shape[1]):
        ax.plot(time_steps, retuned_harmonic[:, i], label=f'Node {i+1}')
    ax.set_title(f'Retuned Harmonic State (Predicted Class: {predicted_class})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)

    # Subplot 4: Dissonance Over Time
    ax = axs[3]
    ax.plot(time_steps, dissonance_values, color='red')
    ax.set_title('Dissonance Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Dissonance')
    ax.grid(True)

    # Subplot 5: Classification Metrics
    ax = axs[4]
    ax.axis('off')  # Hide the axis
    ax.text(0.1, 0.95, f'True Class: {true_class}', fontsize=14)
    ax.text(0.1, 0.90, f'Predicted Class: {predicted_class}', fontsize=14)
    ax.text(0.1, 0.85, f'Classification Accuracy: {accuracy * 100:.2f}%', fontsize=14)
    ax.text(0.1, 0.80, 'Confusion Matrix:', fontsize=14)

    # Plot Confusion Matrix
    cax = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_labels)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # Add text annotations to confusion matrix
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

def plot_accuracy_metrics(y_test, y_pred):
    """
    Computes and returns confusion matrix and classification report.

    Parameters:
    - y_test: True labels.
    - y_pred: Predicted labels.

    Returns:
    - conf_matrix: Confusion matrix.
    - class_report: Classification report.
    """
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    return conf_matrix, class_report

def optimize_harmonic_ratios(X_train, y_train, num_nodes, base_freq, initial_phase=0.0, grid_size=3):
    """
    Optimizes harmonic ratios based on the training data using grid search.

    Parameters:
    - X_train: Training features.
    - y_train: Training labels.
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
        print(f"Optimizing harmonic ratios for class {cls}...")
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
                    # Use default ratios for other classes
                    temp_harmonic_ratios[other_cls] = [1.0] * num_nodes
            # Initialize network
            network = ResonantNetwork(
                num_nodes=num_nodes,
                base_freq=base_freq,
                class_harmonic_ratios=temp_harmonic_ratios,
                initial_phase=initial_phase
            )
            # Evaluate on training data
            correct = 0
            for features, label in zip(X_train, y_train):
                input_signals = convert_features_to_signals(features, time_steps := np.linspace(0, 1, 100))
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
        print(f"Best ratios for class {cls}: {best_cls_ratios} with training accuracy: {best_cls_accuracy * 100:.2f}%")

    average_accuracy = best_total_accuracy / len(class_labels)
    print(f"\nAverage Training Accuracy across classes: {average_accuracy * 100:.2f}%")
    return best_harmonic_ratios

def optimize_harmonic_ratios_random(X_train, y_train, num_nodes, base_freq, initial_phase=0.0, num_samples=1000):
    """
    Optimizes harmonic ratios based on the training data using random sampling.

    Parameters:
    - X_train: Training features.
    - y_train: Training labels.
    - num_nodes: Number of resonant nodes.
    - base_freq: Base frequency for the network.
    - initial_phase: Initial phase for all nodes.
    - num_samples: Number of random samples to try.

    Returns:
    - best_class_harmonic_ratios: Dictionary mapping class labels to optimized harmonic ratios.
    """
    class_labels = np.unique(y_train)
    best_harmonic_ratios = {cls: [1.0]*num_nodes for cls in class_labels}
    best_total_accuracy = 0

    for cls in class_labels:
        print(f"Optimizing harmonic ratios for class {cls}...")
        best_cls_accuracy = 0
        best_cls_ratios = None
        for _ in range(num_samples):
            # Random harmonic ratios between 1.0 and 3.0
            ratios = np.random.uniform(1.0, 3.0, num_nodes)
            temp_harmonic_ratios = {}
            for other_cls in class_labels:
                if other_cls == cls:
                    temp_harmonic_ratios[other_cls] = ratios
                else:
                    # Use default ratios for other classes
                    temp_harmonic_ratios[other_cls] = [1.0] * num_nodes
            # Initialize network
            network = ResonantNetwork(
                num_nodes=num_nodes,
                base_freq=base_freq,
                class_harmonic_ratios=temp_harmonic_ratios,
                initial_phase=initial_phase
            )
            # Evaluate on training data
            correct = 0
            for features, label in zip(X_train, y_train):
                input_signals = convert_features_to_signals(features, time_steps := np.linspace(0, 1, 100))
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
        print(f"Best ratios for class {cls}: {best_cls_ratios} with training accuracy: {best_cls_accuracy * 100:.2f}%")

    average_accuracy = best_total_accuracy / len(class_labels)
    print(f"\nAverage Training Accuracy across classes: {average_accuracy * 100:.2f}%")
    return best_harmonic_ratios


def main():
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define similarity metrics to evaluate
    similarity_metrics = ['cosine', 'euclidean', 'manhattan']
    metric_accuracies = {metric: [] for metric in similarity_metrics}

    # Define number of nodes
    num_nodes = 6  # Expanded from 4 to 6

    # Define base frequency and initial phase
    base_freq = 1.0
    initial_phase = 0.0

    # Perform cross-validation
    for fold, (train_index, test_index) in enumerate(skf.split(X_scaled, y)):
        print(f"\n--- Fold {fold + 1} ---")
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Optimize harmonic ratios on the training data
        optimized_ratios = optimize_harmonic_ratios(X_train, y_train, num_nodes, base_freq, initial_phase, grid_size=5)
        # optimized_ratios = optimize_harmonic_ratios_random(X_train, y_train, num_nodes, base_freq, initial_phase, num_samples=1000)

        
        # Initialize the resonant network with optimized ratios
        network = ResonantNetwork(
            num_nodes=num_nodes,
            base_freq=base_freq,
            class_harmonic_ratios=optimized_ratios,
            initial_phase=initial_phase
        )

        # Evaluate each similarity metric
        for metric in similarity_metrics:
            correct = 0
            for features, label in zip(X_test, y_test):
                input_signals = convert_features_to_signals(features, time_steps := np.linspace(0, 1, 100))
                # Transpose to shape: (num_nodes, len(time_steps))
                input_signals_classify = input_signals.T
                predicted_class = network.classify_input(input_signals_classify, time_steps, similarity_metric=metric)
                if predicted_class == label:
                    correct += 1
            accuracy = correct / len(y_test)
            metric_accuracies[metric].append(accuracy)
            print(f"Similarity Metric: {metric}, Fold Accuracy: {accuracy * 100:.2f}%")

    # Calculate average accuracy for each similarity metric
    avg_accuracies = {metric: np.mean(accs) for metric, accs in metric_accuracies.items()}
    best_metric = max(avg_accuracies, key=avg_accuracies.get)
    print("\n--- Cross-Validation Results ---")
    for metric, acc in avg_accuracies.items():
        print(f"Average Accuracy with {metric}: {acc * 100:.2f}%")
    print(f"\nBest Similarity Metric: {best_metric} with Average Accuracy: {avg_accuracies[best_metric] * 100:.2f}%")

    # Final Train-Test Split for Visualization
    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # Optimize harmonic ratios on the full training data
    print("\n--- Final Optimization on Full Training Data ---")
    optimized_ratios_final = optimize_harmonic_ratios(X_train_full, y_train_full, num_nodes, base_freq, initial_phase, grid_size=3)

    # Initialize the resonant network with optimized ratios
    final_network = ResonantNetwork(
        num_nodes=num_nodes,
        base_freq=base_freq,
        class_harmonic_ratios=optimized_ratios_final,
        initial_phase=initial_phase
    )

    # Perform classification on the final test set using the best similarity metric
    y_pred_final = []
    for features in X_test_final:
        input_signals = convert_features_to_signals(features, time_steps := np.linspace(0, 1, 100))
        input_signals_classify = input_signals.T
        predicted_class = final_network.classify_input(input_signals_classify, time_steps, similarity_metric=best_metric)
        y_pred_final.append(predicted_class)

    # Evaluate final accuracy
    final_accuracy = accuracy_score(y_test_final, y_pred_final)
    print(f"\nFinal Classification Accuracy: {final_accuracy * 100:.2f}%")

    # Compute confusion matrix and classification report
    conf_matrix_final, class_report_final = plot_accuracy_metrics(y_test_final, y_pred_final)

    # Select one example for visualization
    example_idx = 0
    features = X_test_final[example_idx]
    true_class = y_test_final[example_idx]
    predicted_class = y_pred_final[example_idx]

    # Convert input features to signals
    input_signals = convert_features_to_signals(features, time_steps)  # Shape: (len(time_steps), num_nodes)

    # Compute class signals for true and predicted classes
    true_class_signals = final_network.compute_class_signals(true_class, time_steps)  # Shape: (len(time_steps), num_nodes)
    predicted_class_signals = final_network.compute_class_signals(predicted_class, time_steps)  # Shape: (len(time_steps), num_nodes)

    # Compute dissonance over time using the best similarity metric
    dissonance_values = []
    for t in range(len(time_steps)):
        # Compute dissonance between input signals at time t and predicted class signals at time t
        input_vector = input_signals[t]
        predicted_vector = predicted_class_signals[t]
        if best_metric == 'cosine':
            norm_input = np.linalg.norm(input_vector)
            norm_pred = np.linalg.norm(predicted_vector)
            if norm_input == 0 or norm_pred == 0:
                dissonance = 1.0
            else:
                dissonance = cosine(input_vector, predicted_vector)
        elif best_metric == 'euclidean':
            dissonance = euclidean(input_vector, predicted_vector)
        elif best_metric == 'manhattan':
            dissonance = cityblock(input_vector, predicted_vector)
        dissonance_values.append(dissonance)

    # Compute retuned harmonic state (predicted class)
    retuned_harmonic = predicted_class_signals

    # Define class labels for confusion matrix
    class_labels = np.unique(y)

    # Plot all in a single window with multiple subplots
    plot_overview_single_window(
        time_steps,
        initial_harmonic=final_network.compute_class_signals(0, time_steps),  # Initial harmonic state (Class 0)
        input_signals=input_signals,
        retuned_harmonic=retuned_harmonic,
        dissonance_values=dissonance_values,
        true_class=true_class,
        predicted_class=predicted_class,
        accuracy=final_accuracy,
        conf_matrix=conf_matrix_final,
        class_labels=class_labels
    )

if __name__ == "__main__":
    main()