# Emotional Resonance Engine

The Emotional Resonance Engine is an innovative feature of SynthNN that maps human emotions to specific frequency patterns, harmonic signatures, and resonant behaviors. This enables emotion-aware music generation, empathetic audio responses, and cross-cultural emotional expression through sound.

## Overview

The engine is based on psychoacoustic research and the principle that different emotions naturally resonate at different frequencies and harmonic patterns. Each emotion has a unique "signature" consisting of:

- **Base Frequency**: The fundamental frequency that characterizes the emotion
- **Harmonic Series**: The relative strengths of overtones
- **Tempo Range**: Natural rhythm associated with the emotion
- **Phase Coherence**: How synchronized the resonant network should be
- **Musical Mode**: The scale that best expresses the emotion
- **Energy Level**: Overall arousal/activation level
- **Valence**: Positive or negative emotional quality

## Features

### 1. Emotion-to-Frequency Mapping

Each emotion is mapped to specific acoustic properties:

| Emotion | Base Freq       | Mode       | Energy | Valence | Color      |
| ------- | --------------- | ---------- | ------ | ------- | ---------- |
| Joy     | 440 Hz (A4)     | Ionian     | 0.8    | +0.9    | Yellow     |
| Sadness | 220 Hz (A3)     | Aeolian    | 0.3    | -0.7    | Steel Blue |
| Anger   | 130.81 Hz (C3)  | Phrygian   | 0.95   | -0.8    | Crimson    |
| Fear    | 415.30 Hz (G#4) | Locrian    | 0.7    | -0.6    | Purple     |
| Love    | 528 Hz (C5)     | Mixolydian | 0.6    | +0.85   | Pink       |
| Calm    | 256 Hz (C4)     | Lydian     | 0.4    | +0.5    | Turquoise  |

### 2. Empathetic Response Generation

The engine can generate three types of responses to emotional input:

- **Matching**: Reflects the same emotion back
- **Complementary**: Responds with a supportive emotion
- **Balancing**: Provides emotional equilibrium with opposite valence

### 3. Emotional Journey Creation

Create smooth transitions through multiple emotions over time, with automatic cross-fading between emotional states.

### 4. Emotion Analysis

Analyze any audio input to detect its emotional content based on:

- Spectral centroid (brightness)
- Tempo estimation
- Energy levels
- Harmonic content

### 5. Cultural Context

Different cultures express emotions with varying intensities. The engine includes cultural modifiers for:

- Western
- Eastern
- Latin
- Nordic

## Usage Examples

### Basic Emotion Generation

```python
from synthnn.core.emotional_resonance import EmotionalResonanceEngine, EmotionCategory

# Initialize the engine
engine = EmotionalResonanceEngine()

# Create an emotional network
network = engine.create_emotional_network(
    emotion=EmotionCategory.JOY,
    intensity=0.8,
    cultural_context="western"
)

# Generate audio
audio = engine.generate_empathetic_response(
    input_emotion=EmotionCategory.JOY,
    response_type="matching",
    duration=3.0
)
```

### Emotional Journey

```python
# Define an emotional journey
journey = [
    (EmotionCategory.CALM, 2.0),
    (EmotionCategory.JOY, 3.0),
    (EmotionCategory.NOSTALGIA, 2.0)
]

# Generate the journey audio
journey_audio = engine.create_emotional_journey(
    emotions=journey,
    transition_time=1.0,
    total_duration=8.0
)
```

### Emotion Analysis

```python
# Analyze emotional content of audio
emotion_scores = engine.analyze_emotional_content(audio_signal)

# Get the dominant emotion
dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
print(f"Detected emotion: {dominant_emotion[0].value} ({dominant_emotion[1]:.1%})")
```

## Technical Details

### Emotional Signatures

Each emotion signature contains:

- **Base Frequency**: Fundamental frequency in Hz
- **Harmonic Series**: List of relative harmonic strengths [1.0, 0.7, 0.5, ...]
- **Tempo Range**: (min_bpm, max_bpm) tuple
- **Phase Coherence**: 0-1 value for network synchronization
- **Amplitude Envelope**: "attack", "sustain", "decay", or "pulse"
- **Modal Preference**: Musical mode name
- **Color Association**: RGB tuple for synesthetic mapping
- **Energy Level**: 0-1 overall energy/arousal
- **Valence**: -1 to +1 for negative to positive emotion

### Network Generation

The engine creates MusicalResonantNetwork instances with:

1. Nodes tuned to emotion-specific harmonic series
2. Connections weighted by phase coherence
3. Coupling strength based on emotional coherence
4. Musical mode matching the emotion

### Audio Processing

Generated audio includes:

- Amplitude envelopes for natural expression
- Tempo modulation for rhythmic interest
- Phase relationships for emotional coherence
- Normalization for consistent output levels

## Applications

1. **Music Therapy**: Generate therapeutic soundscapes tailored to emotional needs
2. **Empathetic AI**: Create audio responses that understand and respond to human emotions
3. **Game Audio**: Dynamic emotional soundtracks that respond to player state
4. **Meditation Apps**: Guided emotional journeys through sound
5. **Art Installations**: Interactive emotional sound environments
6. **Mental Health Tools**: Emotion regulation through resonant frequencies

## Future Enhancements

- Real-time emotion tracking and response
- Biometric integration (heart rate, skin conductance)
- Personalized emotional profiles
- Multi-speaker spatial audio for immersive experiences
- Integration with visual elements (color, movement)
- Machine learning for improved emotion detection

## References

The Emotional Resonance Engine is inspired by:

- Psychoacoustic research on emotion and sound
- Music therapy principles
- Theories of emotional contagion
- Cross-cultural studies of emotional expression
- Resonance theory of consciousness
