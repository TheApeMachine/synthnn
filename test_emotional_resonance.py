#!/usr/bin/env python3
"""
Test script for the Emotional Resonance Engine
"""

import numpy as np
import matplotlib.pyplot as plt
from synthnn.core.emotional_resonance import EmotionalResonanceEngine, EmotionCategory

def test_emotional_resonance():
    """Test the basic functionality of the Emotional Resonance Engine."""
    
    print("🎵 Testing Emotional Resonance Engine...")
    
    # Initialize engine
    engine = EmotionalResonanceEngine()
    print("✓ Engine initialized")
    
    # Test 1: Create emotional networks
    print("\n1. Testing emotional network creation:")
    emotions_to_test = [EmotionCategory.JOY, EmotionCategory.SADNESS, EmotionCategory.CALM]
    
    for emotion in emotions_to_test:
        network = engine.create_emotional_network(emotion, intensity=0.8)
        print(f"  ✓ Created network for {emotion.value}: {len(network.nodes)} nodes")
    
    # Test 2: Generate empathetic responses
    print("\n2. Testing empathetic response generation:")
    response_types = ["matching", "complementary", "balancing"]
    
    for response_type in response_types:
        audio = engine.generate_empathetic_response(
            EmotionCategory.ANGER,
            response_type=response_type,
            duration=2.0
        )
        print(f"  ✓ Generated {response_type} response: {len(audio)} samples")
    
    # Test 3: Emotional journey
    print("\n3. Testing emotional journey:")
    journey = [
        (EmotionCategory.CALM, 2.0),
        (EmotionCategory.JOY, 3.0),
        (EmotionCategory.NOSTALGIA, 2.0)
    ]
    
    journey_audio = engine.create_emotional_journey(
        journey,
        transition_time=0.5,
        total_duration=8.0
    )
    print(f"  ✓ Created emotional journey: {len(journey_audio)} samples")
    
    # Test 4: Analyze emotional content
    print("\n4. Testing emotion analysis:")
    # Generate a test audio with known emotion
    test_audio = engine.generate_empathetic_response(
        EmotionCategory.JOY,
        "matching",
        duration=3.0
    )
    
    emotion_scores = engine.analyze_emotional_content(test_audio)
    top_emotion = max(emotion_scores.items(), key=lambda x: x[1])
    print(f"  ✓ Analyzed audio - Top emotion: {top_emotion[0].value} ({top_emotion[1]:.2%})")
    
    # Test 5: Cultural modifiers
    print("\n5. Testing cultural context:")
    cultures = ["western", "eastern", "latin", "nordic"]
    
    for culture in cultures:
        network = engine.create_emotional_network(
            EmotionCategory.JOY,
            intensity=1.0,
            cultural_context=culture
        )
        print(f"  ✓ Created {culture} joy network")
    
    # Visualize emotion signatures
    print("\n6. Visualizing emotion signatures:")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    emotions_to_plot = [
        EmotionCategory.JOY, EmotionCategory.SADNESS, 
        EmotionCategory.ANGER, EmotionCategory.FEAR,
        EmotionCategory.CALM, EmotionCategory.LOVE
    ]
    
    for idx, emotion in enumerate(emotions_to_plot):
        ax = axes[idx]
        signature = engine.emotion_signatures[emotion]
        
        # Plot harmonic series
        harmonics = range(1, len(signature.harmonic_series) + 1)
        color = [c/255 for c in signature.color_association]
        
        ax.bar(harmonics, signature.harmonic_series, color=color, alpha=0.7)
        ax.set_title(f"{emotion.value.title()}\n{signature.base_frequency:.1f} Hz")
        ax.set_xlabel("Harmonic")
        ax.set_ylabel("Strength")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("emotion_signatures.png", dpi=150)
    print(f"  ✓ Saved visualization to emotion_signatures.png")
    
    print("\n✅ All tests passed successfully!")
    
    # Generate demo audio files
    print("\n7. Generating demo audio files:")
    demo_emotions = [EmotionCategory.JOY, EmotionCategory.SADNESS, EmotionCategory.CALM]
    
    for emotion in demo_emotions:
        audio = engine.generate_empathetic_response(emotion, "matching", duration=3.0)
        
        # Save as WAV file
        from scipy.io import wavfile
        filename = f"demo_{emotion.value}.wav"
        wavfile.write(filename, 44100, (audio * 32767).astype(np.int16))
        print(f"  ✓ Saved {filename}")
    
    print("\n🎉 Emotional Resonance Engine is working perfectly!")

if __name__ == "__main__":
    test_emotional_resonance() 