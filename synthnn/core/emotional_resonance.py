"""
Emotional Resonance Engine for SynthNN

Maps emotions to frequency/harmonic signatures and creates empathetic resonant responses.
This module enables emotion-aware music generation, mood detection, and cross-cultural 
emotional pattern recognition through resonance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .resonant_network import ResonantNetwork
from .resonant_node import ResonantNode
from .musical_extensions import MusicalResonantNetwork
from .signal_processor import SignalProcessor


class EmotionCategory(Enum):
    """Basic emotion categories based on psychological research."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    LOVE = "love"
    CALM = "calm"
    EXCITEMENT = "excitement"
    MELANCHOLY = "melancholy"
    HOPE = "hope"
    NOSTALGIA = "nostalgia"


@dataclass
class EmotionalSignature:
    """Defines the resonant signature of an emotion."""
    name: str
    base_frequency: float  # Characteristic frequency in Hz
    harmonic_series: List[float]  # Relative harmonic strengths [1.0, 0.5, 0.3, ...]
    tempo_range: Tuple[float, float]  # BPM range
    phase_coherence: float  # 0-1, how synchronized the network should be
    amplitude_envelope: str  # 'attack', 'sustain', 'decay', 'pulse'
    modal_preference: str  # Preferred musical mode
    color_association: Tuple[int, int, int]  # RGB for synesthetic mapping
    energy_level: float  # 0-1, overall energy/arousal level
    valence: float  # -1 to 1, negative to positive emotion


class EmotionalResonanceEngine:
    """
    Engine for emotion-aware resonance generation and analysis.
    """
    
    def __init__(self, base_freq: float = 440.0):
        """
        Initialize the Emotional Resonance Engine.
        
        Args:
            base_freq: Base frequency for calculations
        """
        self.base_freq = base_freq
        self.signal_processor = SignalProcessor()
        self.emotion_signatures = self._initialize_emotion_signatures()
        self.cultural_modifiers = self._initialize_cultural_modifiers()
        
    def _initialize_emotion_signatures(self) -> Dict[EmotionCategory, EmotionalSignature]:
        """Initialize emotional signatures based on psychoacoustic research."""
        return {
            EmotionCategory.JOY: EmotionalSignature(
                name="Joy",
                base_frequency=440.0,  # A4, bright and clear
                harmonic_series=[1.0, 0.7, 0.5, 0.3, 0.2],  # Rich harmonics
                tempo_range=(120, 140),  # Upbeat
                phase_coherence=0.8,  # High synchronization
                amplitude_envelope="attack",
                modal_preference="Ionian",
                color_association=(255, 223, 0),  # Yellow
                energy_level=0.8,
                valence=0.9
            ),
            EmotionCategory.SADNESS: EmotionalSignature(
                name="Sadness",
                base_frequency=220.0,  # A3, lower register
                harmonic_series=[1.0, 0.3, 0.2, 0.1, 0.05],  # Fewer harmonics
                tempo_range=(50, 70),  # Slow
                phase_coherence=0.4,  # Less synchronized
                amplitude_envelope="decay",
                modal_preference="Aeolian",
                color_association=(70, 130, 180),  # Steel blue
                energy_level=0.3,
                valence=-0.7
            ),
            EmotionCategory.ANGER: EmotionalSignature(
                name="Anger",
                base_frequency=130.81,  # C3, low and powerful
                harmonic_series=[1.0, 0.9, 0.8, 0.7, 0.6],  # Many strong harmonics
                tempo_range=(140, 180),  # Fast and aggressive
                phase_coherence=0.2,  # Chaotic
                amplitude_envelope="pulse",
                modal_preference="Phrygian",
                color_association=(220, 20, 60),  # Crimson
                energy_level=0.95,
                valence=-0.8
            ),
            EmotionCategory.FEAR: EmotionalSignature(
                name="Fear",
                base_frequency=415.30,  # G#4, tension frequency
                harmonic_series=[1.0, 0.1, 0.9, 0.1, 0.8],  # Irregular harmonics
                tempo_range=(100, 140),  # Variable, unstable
                phase_coherence=0.3,  # Unstable synchronization
                amplitude_envelope="pulse",
                modal_preference="Locrian",
                color_association=(128, 0, 128),  # Purple
                energy_level=0.7,
                valence=-0.6
            ),
            EmotionCategory.CALM: EmotionalSignature(
                name="Calm",
                base_frequency=256.0,  # C4, balanced
                harmonic_series=[1.0, 0.5, 0.25, 0.125, 0.0625],  # Smooth decay
                tempo_range=(60, 80),  # Relaxed
                phase_coherence=0.9,  # Very synchronized
                amplitude_envelope="sustain",
                modal_preference="Lydian",
                color_association=(175, 238, 238),  # Pale turquoise
                energy_level=0.4,
                valence=0.5
            ),
            EmotionCategory.LOVE: EmotionalSignature(
                name="Love",
                base_frequency=528.0,  # C5, "love frequency"
                harmonic_series=[1.0, 0.8, 0.6, 0.4, 0.3],  # Warm harmonics
                tempo_range=(70, 90),  # Moderate, heartbeat-like
                phase_coherence=0.85,  # Strong connection
                amplitude_envelope="sustain",
                modal_preference="Mixolydian",
                color_association=(255, 192, 203),  # Pink
                energy_level=0.6,
                valence=0.85
            ),
            EmotionCategory.NOSTALGIA: EmotionalSignature(
                name="Nostalgia",
                base_frequency=392.0,  # G4, bittersweet
                harmonic_series=[1.0, 0.6, 0.3, 0.4, 0.2],  # Complex mix
                tempo_range=(80, 100),  # Moderate
                phase_coherence=0.6,  # Partially synchronized
                amplitude_envelope="decay",
                modal_preference="Dorian",
                color_association=(218, 165, 32),  # Goldenrod
                energy_level=0.5,
                valence=0.2
            ),
            EmotionCategory.SURPRISE: EmotionalSignature(
                name="Surprise",
                base_frequency=523.25,  # C5, high and sudden
                harmonic_series=[1.0, 0.8, 0.6, 0.9, 0.4],  # Irregular pattern
                tempo_range=(120, 160),  # Quick tempo
                phase_coherence=0.3,  # Unpredictable
                amplitude_envelope="attack",
                modal_preference="Lydian",
                color_association=(255, 140, 0),  # Dark orange
                energy_level=0.85,
                valence=0.3
            ),
            EmotionCategory.DISGUST: EmotionalSignature(
                name="Disgust",
                base_frequency=185.0,  # F#3, low and grating
                harmonic_series=[1.0, 0.3, 0.8, 0.2, 0.7],  # Dissonant harmonics
                tempo_range=(70, 90),  # Slow to moderate
                phase_coherence=0.2,  # Very low coherence
                amplitude_envelope="pulse",
                modal_preference="Locrian",
                color_association=(154, 205, 50),  # Yellow-green
                energy_level=0.6,
                valence=-0.7
            ),
            EmotionCategory.EXCITEMENT: EmotionalSignature(
                name="Excitement",
                base_frequency=493.88,  # B4, bright and energetic
                harmonic_series=[1.0, 0.9, 0.8, 0.7, 0.6],  # Rich harmonics
                tempo_range=(140, 180),  # Fast
                phase_coherence=0.7,  # High but not perfect
                amplitude_envelope="attack",
                modal_preference="Mixolydian",
                color_association=(255, 69, 0),  # Red-orange
                energy_level=0.9,
                valence=0.8
            ),
            EmotionCategory.MELANCHOLY: EmotionalSignature(
                name="Melancholy",
                base_frequency=293.66,  # D4, middle register
                harmonic_series=[1.0, 0.4, 0.2, 0.3, 0.1],  # Sparse harmonics
                tempo_range=(60, 75),  # Slow
                phase_coherence=0.5,  # Moderate coherence
                amplitude_envelope="decay",
                modal_preference="Phrygian",
                color_association=(148, 0, 211),  # Dark violet
                energy_level=0.35,
                valence=-0.4
            ),
            EmotionCategory.HOPE: EmotionalSignature(
                name="Hope",
                base_frequency=349.23,  # F4, ascending feel
                harmonic_series=[1.0, 0.7, 0.5, 0.6, 0.4],  # Balanced harmonics
                tempo_range=(85, 105),  # Moderate
                phase_coherence=0.75,  # Good coherence
                amplitude_envelope="sustain",
                modal_preference="Ionian",
                color_association=(135, 206, 235),  # Sky blue
                energy_level=0.65,
                valence=0.7
            )
        }
        
    def _initialize_cultural_modifiers(self) -> Dict[str, Dict[EmotionCategory, float]]:
        """
        Initialize cultural modifiers for emotion expression.
        Different cultures express emotions with different intensities.
        """
        return {
            "western": {
                EmotionCategory.JOY: 1.0,
                EmotionCategory.SADNESS: 0.8,
                EmotionCategory.ANGER: 0.9,
            },
            "eastern": {
                EmotionCategory.JOY: 0.8,
                EmotionCategory.SADNESS: 0.6,
                EmotionCategory.CALM: 1.2,
            },
            "latin": {
                EmotionCategory.JOY: 1.2,
                EmotionCategory.LOVE: 1.3,
                EmotionCategory.EXCITEMENT: 1.4,
            },
            "nordic": {
                EmotionCategory.CALM: 1.3,
                EmotionCategory.MELANCHOLY: 1.2,
                EmotionCategory.JOY: 0.7,
            }
        }
        
    def create_emotional_network(self, emotion: EmotionCategory, 
                               intensity: float = 1.0,
                               cultural_context: Optional[str] = None) -> MusicalResonantNetwork:
        """
        Create a resonant network tuned to a specific emotion.
        
        Args:
            emotion: The target emotion
            intensity: Emotional intensity (0-1)
            cultural_context: Optional cultural modifier
            
        Returns:
            MusicalResonantNetwork configured for the emotion
        """
        signature = self.emotion_signatures[emotion]
        
        # Apply cultural modifier if specified
        if cultural_context and cultural_context in self.cultural_modifiers:
            if emotion in self.cultural_modifiers[cultural_context]:
                intensity *= self.cultural_modifiers[cultural_context][emotion]
        
        # Create network with appropriate mode
        network = MusicalResonantNetwork(
            name=f"{emotion.value}_network",
            base_freq=signature.base_frequency,
            mode=signature.modal_preference
        )
        
        # Create nodes based on harmonic series
        num_harmonics = len(signature.harmonic_series)
        for i, harmonic_strength in enumerate(signature.harmonic_series):
            freq = signature.base_frequency * (i + 1)
            amplitude = harmonic_strength * intensity * signature.energy_level
            
            node = ResonantNode(
                node_id=f"harmonic_{i}",
                frequency=freq,
                phase=0 if signature.phase_coherence > 0.7 else np.random.uniform(0, 2*np.pi),
                amplitude=amplitude
            )
            network.add_node(node)
        
        # Create connections based on phase coherence
        node_ids = list(network.nodes.keys())
        for i in range(len(node_ids)):
            for j in range(i+1, len(node_ids)):
                # Connection strength based on harmonic relationship and coherence
                harmonic_distance = abs(i - j)
                weight = signature.phase_coherence / (1 + harmonic_distance)
                
                # Add some randomness for organic feel
                weight *= np.random.uniform(0.8, 1.2)
                
                network.connect(node_ids[i], node_ids[j], weight)
                network.connect(node_ids[j], node_ids[i], weight * 0.7)
        
        # Set coupling strength based on coherence
        network.coupling_strength = signature.phase_coherence * 5.0
        
        return network
        
    def analyze_emotional_content(self, audio: np.ndarray, 
                                sample_rate: float = 44100) -> Dict[EmotionCategory, float]:
        """
        Analyze audio for emotional content.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
            
        Returns:
            Dictionary of emotion probabilities
        """
        # Extract features
        self.signal_processor.sample_rate = sample_rate
        
        # Get spectral centroid (brightness)
        freqs, mags = self.signal_processor.analyze_spectrum(audio)
        spectral_centroid = np.sum(freqs * mags) / np.sum(mags)
        
        # Get tempo estimate
        # Simple autocorrelation-based tempo estimation
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(60, min(len(autocorr)-1, int(sample_rate))):  # 60 samples to 1 second
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append(i)
        
        if peaks:
            avg_peak_distance = np.mean(np.diff(peaks[:10]))  # Use first 10 peaks
            tempo = 60 * sample_rate / avg_peak_distance
        else:
            tempo = 90  # Default tempo
        
        # Get energy level
        energy = np.sqrt(np.mean(audio**2))
        
        # Match against emotion signatures
        emotion_scores = {}
        
        for emotion, signature in self.emotion_signatures.items():
            # Frequency match
            freq_diff = abs(spectral_centroid - signature.base_frequency)
            freq_score = np.exp(-freq_diff / 200)  # Gaussian-like scoring
            
            # Tempo match
            tempo_in_range = signature.tempo_range[0] <= tempo <= signature.tempo_range[1]
            tempo_score = 1.0 if tempo_in_range else 0.5
            
            # Energy match
            energy_diff = abs(energy - signature.energy_level)
            energy_score = 1.0 - energy_diff
            
            # Combined score
            emotion_scores[emotion] = freq_score * tempo_score * energy_score
        
        # Normalize scores
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
        
        return emotion_scores
        
    def generate_empathetic_response(self, input_emotion: EmotionCategory,
                                   response_type: str = "complementary",
                                   duration: float = 3.0,
                                   sample_rate: float = 44100) -> np.ndarray:
        """
        Generate an empathetic audio response to an input emotion.
        
        Args:
            input_emotion: The detected input emotion
            response_type: "complementary", "matching", or "balancing"
            duration: Duration of response in seconds
            sample_rate: Sample rate
            
        Returns:
            Audio signal as numpy array
        """
        # Determine response emotion based on type
        if response_type == "matching":
            response_emotion = input_emotion
            intensity = 0.8  # Slightly lower to not overwhelm
        elif response_type == "complementary":
            # Map to complementary emotions
            complementary_map = {
                EmotionCategory.JOY: EmotionCategory.LOVE,
                EmotionCategory.SADNESS: EmotionCategory.CALM,
                EmotionCategory.ANGER: EmotionCategory.CALM,
                EmotionCategory.FEAR: EmotionCategory.LOVE,
                EmotionCategory.EXCITEMENT: EmotionCategory.JOY,
                EmotionCategory.SURPRISE: EmotionCategory.CALM,
                EmotionCategory.DISGUST: EmotionCategory.CALM,
                EmotionCategory.LOVE: EmotionCategory.JOY,
                EmotionCategory.CALM: EmotionCategory.HOPE,
                EmotionCategory.MELANCHOLY: EmotionCategory.HOPE,
                EmotionCategory.HOPE: EmotionCategory.JOY,
                EmotionCategory.NOSTALGIA: EmotionCategory.LOVE
            }
            response_emotion = complementary_map.get(input_emotion, EmotionCategory.CALM)
            intensity = 1.0
        else:  # balancing
            # Choose emotion with opposite valence
            input_valence = self.emotion_signatures[input_emotion].valence
            
            # Find emotion with most opposite valence
            best_emotion = EmotionCategory.CALM
            max_diff = 0
            
            for emotion, signature in self.emotion_signatures.items():
                valence_diff = abs(signature.valence - (-input_valence))
                if valence_diff < 0.3 and abs(signature.valence) > max_diff:  # Close to opposite
                    best_emotion = emotion
                    max_diff = abs(signature.valence)
            
            response_emotion = best_emotion
            intensity = 0.7
        
        # Create emotional network
        network = self.create_emotional_network(response_emotion, intensity)
        
        # Generate audio with appropriate envelope
        signature = self.emotion_signatures[response_emotion]
        num_samples = int(duration * sample_rate)
        audio = np.zeros(num_samples)
        
        # Generate base signal
        dt = 1.0 / sample_rate
        for i in range(num_samples):
            network.step(dt)
            signals = network.get_signals()
            audio[i] = sum(signals.values())
        
        # Apply amplitude envelope
        envelope = self._create_envelope(signature.amplitude_envelope, num_samples)
        audio *= envelope
        
        # Add tempo modulation
        tempo = np.mean(signature.tempo_range)
        beat_period = int(60 * sample_rate / tempo)
        
        # Create subtle tempo modulation
        for i in range(0, num_samples, beat_period):
            beat_env = np.exp(-np.linspace(0, 5, min(beat_period, num_samples - i)))
            audio[i:i+len(beat_env)] *= (1 + 0.2 * beat_env)
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
        
    def _create_envelope(self, envelope_type: str, num_samples: int) -> np.ndarray:
        """Create amplitude envelope based on type."""
        envelope = np.ones(num_samples)
        
        if envelope_type == "attack":
            # Fast attack, slow decay
            attack_samples = int(0.1 * num_samples)
            decay_samples = num_samples - attack_samples
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            envelope[attack_samples:] = np.exp(-np.linspace(0, 2, decay_samples))
            
        elif envelope_type == "decay":
            # Slow exponential decay
            envelope = np.exp(-np.linspace(0, 3, num_samples))
            
        elif envelope_type == "sustain":
            # Gentle fade in and out
            fade_samples = int(0.1 * num_samples)
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            
        elif envelope_type == "pulse":
            # Rhythmic pulsing
            pulse_period = num_samples // 8
            for i in range(0, num_samples, pulse_period):
                pulse_env = np.exp(-np.linspace(0, 5, min(pulse_period, num_samples - i)))
                envelope[i:i+len(pulse_env)] *= pulse_env
        
        return envelope
        
    def create_emotional_journey(self, emotions: List[Tuple[EmotionCategory, float]],
                               transition_time: float = 1.0,
                               total_duration: float = 10.0,
                               sample_rate: float = 44100) -> np.ndarray:
        """
        Create an audio journey through multiple emotions.
        
        Args:
            emotions: List of (emotion, duration) tuples
            transition_time: Time for transitions between emotions
            total_duration: Total duration if not specified by emotions
            sample_rate: Sample rate
            
        Returns:
            Audio signal
        """
        audio_segments = []
        
        for i, (emotion, duration) in enumerate(emotions):
            # Create network for this emotion
            network = self.create_emotional_network(emotion)
            
            # Generate audio for this segment
            segment_audio = self.generate_empathetic_response(
                emotion, "matching", duration, sample_rate
            )
            
            # Add transition if not the last segment
            if i < len(emotions) - 1:
                # Create transition by cross-fading
                transition_samples = int(transition_time * sample_rate)
                fade_out = np.linspace(1, 0, transition_samples)
                fade_in = np.linspace(0, 1, transition_samples)
                
                # Apply fade out to current segment end
                if len(segment_audio) > transition_samples:
                    segment_audio[-transition_samples:] *= fade_out
                
            audio_segments.append(segment_audio)
        
        # Concatenate all segments
        full_audio = np.concatenate(audio_segments)
        
        # Trim or pad to exact duration
        target_samples = int(total_duration * sample_rate)
        if len(full_audio) > target_samples:
            full_audio = full_audio[:target_samples]
        elif len(full_audio) < target_samples:
            padding = np.zeros(target_samples - len(full_audio))
            full_audio = np.concatenate([full_audio, padding])
        
        return full_audio
        
    def get_emotion_color(self, emotion: EmotionCategory) -> Tuple[int, int, int]:
        """Get RGB color associated with an emotion."""
        return self.emotion_signatures[emotion].color_association 