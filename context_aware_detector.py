import numpy as np
from detector import ModeDetector
from collections import deque
import json

class ContextAwareModeDetector(ModeDetector):
    """
    Enhanced mode detector that considers context from multiple sources:
    - Temporal context (what modes came before)
    - Semantic context (what the signal represents)
    - Structural context (patterns and relationships)
    """
    
    def __init__(self, context_window=10):
        super().__init__()
        self.context_window = context_window
        
        # Context tracking
        self.temporal_context = deque(maxlen=context_window)
        self.semantic_tags = []
        self.structural_patterns = {}
        
        # Learning components
        self.mode_transitions = self._initialize_transition_model()
        self.context_embeddings = {}
        self.pattern_library = PatternLibrary()
        
    def _initialize_transition_model(self):
        """Initialize mode transition probabilities"""
        # Based on musical theory and common progressions
        transitions = {
            'Ionian': {
                'Ionian': 0.3, 'Dorian': 0.15, 'Phrygian': 0.05,
                'Lydian': 0.2, 'Mixolydian': 0.2, 'Aeolian': 0.1
            },
            'Dorian': {
                'Dorian': 0.25, 'Ionian': 0.2, 'Phrygian': 0.15,
                'Aeolian': 0.25, 'Mixolydian': 0.15
            },
            'Phrygian': {
                'Phrygian': 0.3, 'Dorian': 0.25, 'Aeolian': 0.25,
                'Locrian': 0.1, 'Ionian': 0.1
            },
            'Lydian': {
                'Lydian': 0.3, 'Ionian': 0.35, 'Mixolydian': 0.25,
                'Dorian': 0.1
            },
            'Mixolydian': {
                'Mixolydian': 0.3, 'Ionian': 0.25, 'Dorian': 0.25,
                'Lydian': 0.1, 'Aeolian': 0.1
            },
            'Aeolian': {
                'Aeolian': 0.3, 'Dorian': 0.25, 'Phrygian': 0.2,
                'Ionian': 0.15, 'Locrian': 0.1
            },
            'Locrian': {
                'Locrian': 0.2, 'Phrygian': 0.4, 'Aeolian': 0.3,
                'Dorian': 0.1
            }
        }
        
        # Ensure all modes have entries for all other modes
        for mode in self.mode_intervals.keys():
            if mode not in transitions:
                transitions[mode] = {}
            for target in self.mode_intervals.keys():
                if target not in transitions[mode]:
                    transitions[mode][target] = 0.05  # Small default probability
                    
        return transitions
    
    def analyze_with_context(self, signal, sample_rate=1.0, semantic_tags=None, structural_hints=None):
        """
        Analyze signal with additional context information.
        
        Args:
            signal: Input signal
            sample_rate: Sample rate
            semantic_tags: List of semantic descriptors (e.g., ['melancholic', 'slow'])
            structural_hints: Dict of structural information (e.g., {'section': 'verse', 'position': 0.3})
            
        Returns:
            str: Detected mode considering context
        """
        # Get base analysis
        features = self._extract_features(signal, sample_rate)
        base_scores = self._calculate_mode_scores(features)
        
        # Apply temporal context
        if self.temporal_context:
            temporal_scores = self._apply_temporal_context(base_scores)
        else:
            temporal_scores = base_scores
            
        # Apply semantic context
        if semantic_tags:
            semantic_scores = self._apply_semantic_context(temporal_scores, semantic_tags)
        else:
            semantic_scores = temporal_scores
            
        # Apply structural context
        if structural_hints:
            final_scores = self._apply_structural_context(semantic_scores, structural_hints)
        else:
            final_scores = semantic_scores
            
        # Select mode and update context
        selected_mode = max(final_scores.items(), key=lambda x: x[1])[0]
        self.temporal_context.append({
            'mode': selected_mode,
            'confidence': final_scores[selected_mode],
            'features': features
        })
        
        return selected_mode
    
    def _apply_temporal_context(self, base_scores):
        """Apply temporal context using transition probabilities"""
        if not self.temporal_context:
            return base_scores
            
        # Get previous mode
        prev_mode = self.temporal_context[-1]['mode']
        
        # Apply transition probabilities
        adjusted_scores = {}
        for mode, score in base_scores.items():
            transition_prob = self.mode_transitions[prev_mode].get(mode, 0.05)
            # Weight combination: 70% current analysis, 30% transition probability
            adjusted_scores[mode] = 0.7 * score + 0.3 * transition_prob
            
        return adjusted_scores
    
    def _apply_semantic_context(self, scores, semantic_tags):
        """Adjust scores based on semantic context"""
        # Define semantic-mode associations
        semantic_mode_affinity = {
            'bright': {'Ionian': 0.3, 'Lydian': 0.4, 'Mixolydian': 0.2},
            'dark': {'Phrygian': 0.4, 'Locrian': 0.3, 'Aeolian': 0.2},
            'melancholic': {'Aeolian': 0.4, 'Dorian': 0.3, 'Phrygian': 0.2},
            'mysterious': {'Locrian': 0.3, 'Phrygian': 0.3, 'Lydian': 0.2},
            'heroic': {'Mixolydian': 0.4, 'Ionian': 0.3, 'Lydian': 0.2},
            'contemplative': {'Dorian': 0.5, 'Aeolian': 0.3},
            'ethereal': {'Lydian': 0.5, 'Ionian': 0.2, 'Dorian': 0.2},
            'tense': {'Locrian': 0.4, 'Phrygian': 0.3, 'Lydian': 0.2},
            'peaceful': {'Ionian': 0.4, 'Lydian': 0.3, 'Mixolydian': 0.2},
            'nostalgic': {'Dorian': 0.4, 'Aeolian': 0.3, 'Mixolydian': 0.2}
        }
        
        adjusted_scores = scores.copy()
        
        for tag in semantic_tags:
            if tag in semantic_mode_affinity:
                affinities = semantic_mode_affinity[tag]
                for mode, affinity in affinities.items():
                    if mode in adjusted_scores:
                        # Boost score based on semantic affinity
                        adjusted_scores[mode] *= (1 + affinity * 0.3)
                        
        # Renormalize
        total = sum(adjusted_scores.values())
        return {k: v/total for k, v in adjusted_scores.items()}
    
    def _apply_structural_context(self, scores, structural_hints):
        """Adjust scores based on structural position"""
        adjusted_scores = scores.copy()
        
        # Example structural rules
        if 'section' in structural_hints:
            section = structural_hints['section']
            
            if section == 'intro':
                # Intros often use more stable modes
                adjusted_scores['Ionian'] *= 1.2
                adjusted_scores['Lydian'] *= 1.1
                
            elif section == 'verse':
                # Verses might use more narrative modes
                adjusted_scores['Dorian'] *= 1.2
                adjusted_scores['Aeolian'] *= 1.1
                
            elif section == 'chorus':
                # Choruses often bright and memorable
                adjusted_scores['Ionian'] *= 1.3
                adjusted_scores['Mixolydian'] *= 1.2
                
            elif section == 'bridge':
                # Bridges often explore different territory
                adjusted_scores['Phrygian'] *= 1.2
                adjusted_scores['Lydian'] *= 1.2
                
        # Position-based adjustments
        if 'position' in structural_hints:
            position = structural_hints['position']  # 0.0 to 1.0
            
            if position < 0.2:  # Beginning
                # Favor establishing modes
                adjusted_scores['Ionian'] *= 1.1
                
            elif position > 0.8:  # Ending
                # Favor resolution
                adjusted_scores['Ionian'] *= 1.2
                adjusted_scores['Aeolian'] *= 1.1
                
        # Renormalize
        total = sum(adjusted_scores.values())
        return {k: v/total for k, v in adjusted_scores.items()}
    
    def learn_from_sequence(self, mode_sequence):
        """Update transition model based on observed sequences"""
        for i in range(len(mode_sequence) - 1):
            from_mode = mode_sequence[i]
            to_mode = mode_sequence[i + 1]
            
            # Increase transition probability
            old_prob = self.mode_transitions[from_mode][to_mode]
            self.mode_transitions[from_mode][to_mode] = old_prob * 0.9 + 0.1
            
            # Renormalize
            total = sum(self.mode_transitions[from_mode].values())
            for mode in self.mode_transitions[from_mode]:
                self.mode_transitions[from_mode][mode] /= total


class PatternLibrary:
    """Library of recognized patterns and their modal associations"""
    
    def __init__(self):
        self.patterns = {
            'ascending_scale': {
                'detector': self._detect_ascending_scale,
                'mode_affinities': {'Ionian': 0.3, 'Lydian': 0.4, 'Mixolydian': 0.2}
            },
            'descending_scale': {
                'detector': self._detect_descending_scale,
                'mode_affinities': {'Aeolian': 0.3, 'Phrygian': 0.3, 'Dorian': 0.2}
            },
            'chromatic_movement': {
                'detector': self._detect_chromatic,
                'mode_affinities': {'Locrian': 0.3, 'Phrygian': 0.3, 'Lydian': 0.2}
            },
            'pedal_tone': {
                'detector': self._detect_pedal,
                'mode_affinities': {'Mixolydian': 0.3, 'Dorian': 0.3, 'Ionian': 0.2}
            }
        }
    
    def detect_patterns(self, signal, sample_rate=1.0):
        """Detect patterns in the signal"""
        detected = {}
        for pattern_name, pattern_info in self.patterns.items():
            if pattern_info['detector'](signal, sample_rate):
                detected[pattern_name] = pattern_info['mode_affinities']
        return detected
    
    def _detect_ascending_scale(self, signal, sample_rate):
        """Detect ascending scale patterns"""
        # Simple heuristic: check if frequencies increase over time
        fft_windows = []
        window_size = len(signal) // 4
        
        for i in range(0, len(signal) - window_size, window_size // 2):
            window = signal[i:i + window_size]
            fft = np.fft.rfft(window)
            freqs = np.fft.rfftfreq(len(window), 1/sample_rate)
            
            # Find dominant frequency
            dominant_idx = np.argmax(np.abs(fft))
            dominant_freq = freqs[dominant_idx]
            fft_windows.append(dominant_freq)
        
        # Check if frequencies are ascending
        if len(fft_windows) > 1:
            diffs = np.diff(fft_windows)
            return np.sum(diffs > 0) > len(diffs) * 0.6
        return False
    
    def _detect_descending_scale(self, signal, sample_rate):
        """Detect descending scale patterns"""
        # Similar to ascending but check for decrease
        fft_windows = []
        window_size = len(signal) // 4
        
        for i in range(0, len(signal) - window_size, window_size // 2):
            window = signal[i:i + window_size]
            fft = np.fft.rfft(window)
            freqs = np.fft.rfftfreq(len(window), 1/sample_rate)
            
            dominant_idx = np.argmax(np.abs(fft))
            dominant_freq = freqs[dominant_idx]
            fft_windows.append(dominant_freq)
        
        if len(fft_windows) > 1:
            diffs = np.diff(fft_windows)
            return np.sum(diffs < 0) > len(diffs) * 0.6
        return False
    
    def _detect_chromatic(self, signal, sample_rate):
        """Detect chromatic movement"""
        # Look for many small frequency steps
        # This is a simplified detection
        fft = np.fft.rfft(signal)
        magnitude = np.abs(fft)
        
        # Count peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(magnitude, height=np.max(magnitude) * 0.1)
        
        # Many peaks might indicate chromatic movement
        return len(peaks) > 10
    
    def _detect_pedal(self, signal, sample_rate):
        """Detect pedal tone (sustained bass note)"""
        # Check for consistent low frequency component
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/sample_rate)
        magnitude = np.abs(fft)
        
        # Look at low frequency region
        low_freq_mask = freqs < 200  # Hz
        low_freq_energy = np.sum(magnitude[low_freq_mask])
        total_energy = np.sum(magnitude)
        
        # High ratio indicates pedal tone
        return low_freq_energy / total_energy > 0.3


def demonstrate_context_aware_detection():
    """Demonstrate context-aware mode detection"""
    detector = ContextAwareModeDetector(context_window=5)
    
    # Generate test signals with different contexts
    t = np.linspace(0, 2, 500)
    
    # Signal 1: Bright and ascending (should favor Lydian/Ionian)
    signal1 = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 550 * t)
    mode1 = detector.analyze_with_context(
        signal1, 
        sample_rate=250,
        semantic_tags=['bright', 'heroic'],
        structural_hints={'section': 'intro', 'position': 0.0}
    )
    
    # Signal 2: Dark and descending (should favor Phrygian/Aeolian)
    signal2 = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 233 * t)
    mode2 = detector.analyze_with_context(
        signal2,
        sample_rate=250,
        semantic_tags=['dark', 'mysterious'],
        structural_hints={'section': 'verse', 'position': 0.3}
    )
    
    # Signal 3: Resolution (should favor return to tonic)
    signal3 = np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 660 * t)
    mode3 = detector.analyze_with_context(
        signal3,
        sample_rate=250,
        semantic_tags=['peaceful'],
        structural_hints={'section': 'outro', 'position': 0.95}
    )
    
    print("Context-Aware Mode Detection Results:")
    print("=" * 50)
    print(f"Signal 1 (bright, heroic intro): {mode1}")
    print(f"Signal 2 (dark, mysterious verse): {mode2}")
    print(f"Signal 3 (peaceful resolution): {mode3}")
    
    # Show temporal context influence
    print("\nTemporal Context:")
    for ctx in detector.temporal_context:
        print(f"  {ctx['mode']} (confidence: {ctx['confidence']:.3f})")


if __name__ == "__main__":
    demonstrate_context_aware_detection() 