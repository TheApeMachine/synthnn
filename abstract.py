import numpy as np
import matplotlib.pyplot as plt

class ResonantNode:
    def __init__(self, freq=1.0, phase=0.0, amplitude=1.0, node_id=0):
        self.freq = freq
        self.phase = phase
        self.amplitude = amplitude
        self.node_id = node_id

    def compute(self, time_step):
        return self.amplitude * np.sin(2 * np.pi * self.freq * time_step + self.phase)

    def retune(self, new_freq, new_phase, new_amplitude):
        self.freq = new_freq
        self.phase = new_phase
        self.amplitude = new_amplitude

class ResonantNetwork:
    def __init__(self, num_nodes, base_freq, harmonic_ratios, target_phase, target_amplitude):
        """
        Initialize the nodes based on harmonic intervals relative to a base frequency.
        :param harmonic_ratios: List of harmonic ratios (e.g., [1, 1.5, 1.33]) to space nodes.
        """
        self.nodes = [ResonantNode(freq=base_freq * ratio, phase=target_phase, amplitude=target_amplitude, node_id=i+1)
                      for i, ratio in enumerate(harmonic_ratios)]
        self.num_nodes = num_nodes
        self.harmonic_outputs = []
        self.dissonant_outputs = []
        self.retuned_outputs = []
        self.mode = 'Ionian'  # Default mode
    
    def compute_harmonic_state(self, time_steps):
        """
        Compute the original harmonic state of the network.
        """
        for t in time_steps:
            outputs = [node.compute(t) for node in self.nodes]
            self.harmonic_outputs.append(outputs)
    
    def compute_dissonant_state(self, time_steps, foreign_signal):
        """
        Compute the dissonant state when a foreign signal is introduced.
        """
        for idx, t in enumerate(time_steps):
            outputs = [node.compute(t) + foreign_signal[idx] for node in self.nodes]
            self.dissonant_outputs.append(outputs)
    
    def retune_nodes(self, foreign_signal, time_steps):
        """
        Retune nodes to match the new harmonic structure formed by the foreign signal.
        """
        foreign_freq = self.analyze_foreign_signal(foreign_signal)
        self.mode = self.select_mode(foreign_freq)  # Choose the mode
        new_harmonic_structure = self.get_mode_intervals(self.mode, foreign_freq)
        
        for t in time_steps:
            for node, new_freq in zip(self.nodes, new_harmonic_structure):
                node.retune(new_freq, node.phase, node.amplitude)
            outputs = [node.compute(t) for node in self.nodes]
            self.retuned_outputs.append(outputs)
    
    def analyze_foreign_signal(self, foreign_signal):
        """
        Analyze the foreign signal to estimate its frequency.
        :return: Estimated frequency of the foreign signal.
        """
        estimated_freq = np.fft.fftfreq(len(foreign_signal), d=0.01)[np.argmax(np.abs(np.fft.fft(foreign_signal)))]
        return np.abs(estimated_freq)
    
    def select_mode(self, foreign_freq):
        """
        Based on the foreign signal's frequency, select an appropriate musical mode.
        :param foreign_freq: Frequency of the foreign signal.
        :return: The name of the mode selected (e.g., 'Ionian', 'Dorian').
        """
        # Simple logic to switch modes based on frequency (can be expanded)
        if foreign_freq < 1.2:
            return 'Ionian'
        elif foreign_freq < 1.5:
            return 'Dorian'
        elif foreign_freq < 1.8:
            return 'Lydian'
        else:
            return 'Mixolydian'
    
    def get_mode_intervals(self, mode, foreign_freq):
        """
        Get harmonic intervals for the nodes based on the selected mode.
        :param mode: The selected mode (e.g., 'Ionian', 'Dorian').
        :param foreign_freq: Frequency of the foreign signal.
        :return: List of new frequencies for the nodes in harmonic relation to the foreign signal.
        """
        # Harmonic ratios for each mode (these can be refined or expanded)
        modes = {
            'Ionian': [1, 1.25, 1.5, 2, 2.5],      # Major scale intervals
            'Dorian': [1, 1.22, 1.5, 2, 2.33],    # Dorian mode intervals
            'Lydian': [1, 1.33, 1.67, 2, 2.5],    # Lydian mode with raised fourth
            'Mixolydian': [1, 1.25, 1.5, 2, 2.33] # Mixolydian mode intervals
        }
        harmonic_ratios = modes.get(mode, modes['Ionian'])
        return [foreign_freq * ratio for ratio in harmonic_ratios]
    
    def compute_difference(self):
        """
        Compute the difference between the harmonic and dissonant states.
        :return: Array of differences between harmonic and dissonant outputs.
        """
        return np.array(self.harmonic_outputs) - np.array(self.dissonant_outputs)

class Plotter:
    def __init__(self, network, time_steps, foreign_signal):
        self.network = network
        self.time_steps = time_steps
        self.foreign_signal = foreign_signal
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
    
    def plot_harmonic_state(self):
        """
        Plot the harmonic state of the network nodes.
        """
        ax = self.axes[0, 0]
        for i in range(self.network.num_nodes):
            ax.plot(self.time_steps, [output[i] for output in self.network.harmonic_outputs], label=f'Node {i+1}')
        ax.set_title('Harmonically Tuned State')
        ax.legend()
        ax.grid(True)
    
    def plot_dissonant_state(self):
        """
        Plot the dissonant state of the network nodes with the foreign signal.
        """
        ax = self.axes[0, 1]
        for i in range(self.network.num_nodes):
            ax.plot(self.time_steps, [output[i] for output in self.network.dissonant_outputs], label=f'Node {i+1}')
        ax.plot(self.time_steps, self.foreign_signal, label='Foreign Signal', linestyle='--', color='red')
        ax.set_title(f'Foreign Signal with Dissonant State (Mode: {self.network.mode})')
        ax.legend()
        ax.grid(True)
    
    def plot_difference(self):
        """
        Plot the difference between the harmonic and dissonant states.
        """
        ax = self.axes[1, 0]
        differences = self.network.compute_difference()
        for i in range(self.network.num_nodes):
            ax.plot(self.time_steps, differences[:, i], label=f'Node {i+1}')
        ax.set_title('Difference Between Harmonic and Dissonant States')
        ax.legend()
        ax.grid(True)
    
    def plot_retuned_state(self):
        """
        Plot the retuned state of the network nodes harmonized with the foreign signal.
        """
        ax = self.axes[1, 1]
        for i in range(self.network.num_nodes):
            ax.plot(self.time_steps, [output[i] for output in self.network.retuned_outputs], label=f'Node {i+1}')
        ax.plot(self.time_steps, self.foreign_signal, label='Foreign Signal', linestyle='--', color='red')
        ax.set_title(f'Retuned State in Harmony with Foreign Signal (Mode: {self.network.mode})')
        ax.legend()
        ax.grid(True)
    
    def show(self):
        """
        Display the plots.
        """
        plt.tight_layout()
        plt.show()

# Example usage
base_freq = 1.0  # Base frequency for the harmonically tuned state
harmonic_ratios = [1, 1.5, 1.33, 2, 2.5]  # Harmonic intervals for the nodes
target_phase = 0.0
target_amplitude = 1.0
num_nodes = len(harmonic_ratios)
time_steps = np.linspace(0, 5, 500)
foreign_signal = np.sin(2 * np.pi * 1.5 * time_steps)  # A foreign signal with a different frequency

# Create and propagate network
network = ResonantNetwork(num_nodes, base_freq, harmonic_ratios, target_phase, target_amplitude)
network.compute_harmonic_state(time_steps)
network.compute_dissonant_state(time_steps, foreign_signal)
network.retune_nodes(foreign_signal, time_steps)

# Plot the results
plotter = Plotter(network, time_steps, foreign_signal)
plotter.plot_harmonic_state()
plotter.plot_dissonant_state()
plotter.plot_difference()
plotter.plot_retuned_state()
plotter.show()
