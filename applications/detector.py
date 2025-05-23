import numpy as np
from scipy.signal import find_peaks
from scipy.stats import entropy
from collections import defaultdict

class ModeDetector:
    """
    Analyzes input signals to determine the most appropriate musical mode
    based on harmonic content, spectral characteristics, and temporal patterns.
    """
    
    def __init__(self):
        # Define mode characteristics based on their harmonic intervals
        self.mode_intervals = {
            'Ionian': [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8],     # Major scale
            'Dorian': [1, 9/8, 6/5, 4/3, 3/2, 5/3, 9/5],      # Minor with raised 6th
            'Phrygian': [1, 16/15, 6/5, 4/3, 3/2, 8/5, 9/5],  # Minor with lowered 2nd
            'Lydian': [1, 9/8, 5/4, 45/32, 3/2, 5/3, 15/8],   # Major with raised 4th
            'Mixolydian': [1, 9/8, 5/4, 4/3, 3/2, 5/3, 16/9], # Major with lowered 7th
            'Aeolian': [1, 9/8, 6/5, 4/3, 3/2, 8/5, 9/5],     # Natural minor
            'Locrian': [1, 16/15, 6/5, 4/3, 64/45, 8/5, 9/5]  # Diminished
        }
        
        # Mode "mood" characteristics for matching signal properties
        self.mode_characteristics = {
            'Ionian': {'brightness': 0.9, 'stability': 0.95, 'complexity': 0.3},
            'Dorian': {'brightness': 0.6, 'stability': 0.8, 'complexity': 0.5},
            'Phrygian': {'brightness': 0.3, 'stability': 0.6, 'complexity': 0.7},
            'Lydian': {'brightness': 0.95, 'stability': 0.7, 'complexity': 0.6},
            'Mixolydian': {'brightness': 0.8, 'stability': 0.85, 'complexity': 0.4},
            'Aeolian': {'brightness': 0.4, 'stability': 0.7, 'complexity': 0.5},
            'Locrian': {'brightness': 0.1, 'stability': 0.3, 'complexity': 0.9}
        }
        
        # Cache for performance
        self.signal_cache = {}
        
    def analyze(self, signal, sample_rate=1.0):
        """
        Analyze a signal and determine the most appropriate mode.
        
        Args:
            signal: Input signal (numpy array)
            sample_rate: Sample rate of the signal
            
        Returns:
            str: Name of the detected mode
        """
        # Extract signal features
        features = self._extract_features(signal, sample_rate)
        
        # Calculate mode scores based on features
        mode_scores = self._calculate_mode_scores(features)
        
        # Return the mode with highest score
        return max(mode_scores.items(), key=lambda x: x[1])[0]
    
    def get_mode_probabilities(self, signal, sample_rate=1.0):
        """
        Get probability distribution over all modes for a given signal.
        
        Args:
            signal: Input signal (numpy array)
            sample_rate: Sample rate of the signal
            
        Returns:
            dict: Probability for each mode
        """
        features = self._extract_features(signal, sample_rate)
        mode_scores = self._calculate_mode_scores(features)
        
        # Convert scores to probabilities using softmax
        scores_array = np.array(list(mode_scores.values()))
        exp_scores = np.exp(scores_array - np.max(scores_array))
        probabilities = exp_scores / np.sum(exp_scores)
        
        return {mode: prob for mode, prob in zip(mode_scores.keys(), probabilities)}
    
    def _extract_features(self, signal, sample_rate):
        """
        Extract relevant features from the signal for mode detection.
        """
        features = {}
        
        # 1. Spectral features
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/sample_rate)
        magnitude = np.abs(fft)
        
        # Find peaks in spectrum
        peaks, properties = find_peaks(magnitude, height=np.max(magnitude)*0.1)
        peak_freqs = freqs[peaks]
        peak_mags = magnitude[peaks]
        
        # 2. Harmonic content analysis
        if len(peak_freqs) > 0:
            fundamental = peak_freqs[np.argmax(peak_mags)]
            
            # Check for harmonic relationships
            harmonic_ratios = []
            for freq in peak_freqs[1:]:
                ratio = freq / fundamental
                harmonic_ratios.append(ratio)
            
            features['harmonic_ratios'] = harmonic_ratios
            features['fundamental'] = fundamental
        else:
            features['harmonic_ratios'] = []
            features['fundamental'] = 0
        
        # 3. Spectral centroid (brightness indicator)
        if np.sum(magnitude) > 0:
            features['spectral_centroid'] = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            features['spectral_centroid'] = 0
        
        # 4. Spectral entropy (complexity indicator)
        if np.sum(magnitude) > 0:
            normalized_magnitude = magnitude / np.sum(magnitude)
            features['spectral_entropy'] = entropy(normalized_magnitude)
        else:
            features['spectral_entropy'] = 0
        
        # 5. Zero crossing rate (stability indicator)
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        features['zcr'] = zero_crossings / len(signal)
        
        # 6. Energy distribution
        energy = np.sum(signal**2)
        features['energy'] = energy
        
        # 7. Temporal patterns
        # Autocorrelation for periodicity detection
        autocorr = np.correlate(signal, signal, mode='same')
        features['periodicity'] = np.max(autocorr[len(autocorr)//2:]) / np.max(autocorr)
        
        return features
    
    def _calculate_mode_scores(self, features):
        """
        Calculate fitness scores for each mode based on extracted features.
        """
        mode_scores = {}
        
        for mode, intervals in self.mode_intervals.items():
            score = 0
            
            # 1. Harmonic matching score
            if 'harmonic_ratios' in features and features['harmonic_ratios']:
                harmonic_score = self._calculate_harmonic_fit(
                    features['harmonic_ratios'], 
                    intervals
                )
                score += harmonic_score * 0.4
            
            # 2. Brightness matching
            # Normalize spectral centroid to 0-1 range
            if features['fundamental'] > 0:
                brightness = min(features['spectral_centroid'] / (features['fundamental'] * 10), 1.0)
            else:
                brightness = 0.5
            
            brightness_diff = abs(brightness - self.mode_characteristics[mode]['brightness'])
            score += (1 - brightness_diff) * 0.2
            
            # 3. Complexity matching
            # Normalize entropy to 0-1 range
            complexity = min(features['spectral_entropy'] / 5.0, 1.0)
            complexity_diff = abs(complexity - self.mode_characteristics[mode]['complexity'])
            score += (1 - complexity_diff) * 0.2
            
            # 4. Stability matching
            # Use zero crossing rate as inverse stability measure
            stability = 1 - min(features['zcr'] * 10, 1.0)
            stability_diff = abs(stability - self.mode_characteristics[mode]['stability'])
            score += (1 - stability_diff) * 0.2
            
            mode_scores[mode] = score
        
        return mode_scores
    
    def _calculate_harmonic_fit(self, observed_ratios, mode_intervals):
        """
        Calculate how well observed harmonic ratios fit with mode intervals.
        """
        if not observed_ratios:
            return 0
        
        fit_score = 0
        matches = 0
        
        # Check each observed ratio against mode intervals
        for obs_ratio in observed_ratios:
            # Find closest interval in the mode
            min_diff = float('inf')
            for interval in mode_intervals:
                diff = abs(obs_ratio - interval)
                if diff < min_diff:
                    min_diff = diff
            
            # Score based on how close the match is
            if min_diff < 0.05:  # Very close match
                fit_score += 1.0
                matches += 1
            elif min_diff < 0.1:  # Good match
                fit_score += 0.7
                matches += 1
            elif min_diff < 0.2:  # Acceptable match
                fit_score += 0.3
                matches += 0.5
        
        # Normalize by number of ratios
        if len(observed_ratios) > 0:
            return fit_score / len(observed_ratios)
        return 0
    
    def suggest_mode_transition(self, current_mode, target_signal, sample_rate=1.0):
        """
        Suggest a smooth transition path from current mode to the mode
        best suited for the target signal.
        """
        target_mode = self.analyze(target_signal, sample_rate)
        
        if current_mode == target_mode:
            return [current_mode]
        
        # Define mode adjacency based on shared notes/intervals
        mode_adjacency = {
            'Ionian': ['Lydian', 'Mixolydian', 'Aeolian'],
            'Dorian': ['Aeolian', 'Mixolydian', 'Phrygian'],
            'Phrygian': ['Dorian', 'Aeolian', 'Locrian'],
            'Lydian': ['Ionian', 'Mixolydian'],
            'Mixolydian': ['Ionian', 'Dorian', 'Lydian'],
            'Aeolian': ['Dorian', 'Phrygian', 'Ionian'],
            'Locrian': ['Phrygian']
        }
        
        # Find shortest path using BFS
        from collections import deque
        
        queue = deque([(current_mode, [current_mode])])
        visited = {current_mode}
        
        while queue:
            mode, path = queue.popleft()
            
            if mode == target_mode:
                return path
            
            for adjacent in mode_adjacency.get(mode, []):
                if adjacent not in visited:
                    visited.add(adjacent)
                    queue.append((adjacent, path + [adjacent]))
        
        # If no path found, return direct transition
        return [current_mode, target_mode]