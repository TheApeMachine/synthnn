"""
Demo of the fully migrated music generation system using AcceleratedMusicalNetwork.
This demonstrates the performance benefits and new features available after migration.
"""

import numpy as np
import time
from scipy.io import wavfile
import matplotlib.pyplot as plt

from applications.modal_music_generator import ModalMusicGenerator
from applications.interactive_modal_generator import InteractiveModalGenerator
from synthnn.performance import BackendManager
from synthnn.core import AcceleratedMusicalNetwork


def demonstrate_accelerated_music_generation():
    """Demonstrate the accelerated music generation capabilities"""
    print("=" * 70)
    print("SYNTHNN Accelerated Music Generation Demo")
    print("=" * 70)
    
    # Check available backends
    print("\n1. Checking available backends...")
    manager = BackendManager()
    available_backends = manager.available_backends
    print(f"Available backends: {[b.value for b in available_backends]}")
    print(f"Selected backend: {manager.backend_type.value}")
    
    # Create the modal music generator
    print("\n2. Creating Modal Music Generator...")
    generator = ModalMusicGenerator(base_freq=440.0, sample_rate=44100)
    
    # Generate a short composition
    print("\n3. Generating a 15-second composition...")
    print("   Structure: intro -> verse -> chorus -> outro")
    
    start_time = time.time()
    composition = generator.generate_composition(
        duration_seconds=15,
        structure='verse-chorus'
    )
    generation_time = time.time() - start_time
    
    print(f"   Generation completed in {generation_time:.2f} seconds")
    print(f"   Average generation speed: {15/generation_time:.2f}x realtime")
    
    # Save the composition
    filename = f"accelerated_composition_{manager.backend_type.value}.wav"
    generator.save_composition(composition, filename)
    print(f"   Saved to: {filename}")
    
    # Show generation history
    print("\n4. Generation History:")
    if generator.generation_history['modes']:
        print(f"   Modes used: {set(generator.generation_history['modes'])}")
        print(f"   Mode changes: {len(set(generator.generation_history['modes'])) - 1}")
    
    # Demonstrate interactive generation
    print("\n5. Creating Interactive Generator...")
    interactive_gen = InteractiveModalGenerator(base_freq=440.0, sample_rate=44100)
    
    # Generate phrases with different parameters
    print("\n6. Generating parametric phrases...")
    
    parameter_sets = [
        {'name': 'Simple & Bright', 'complexity': 0.2, 'brightness': 0.9, 'density': 0.3, 'variation': 0.1},
        {'name': 'Complex & Dark', 'complexity': 0.8, 'brightness': 0.2, 'density': 0.7, 'variation': 0.7},
        {'name': 'Balanced', 'complexity': 0.5, 'brightness': 0.6, 'density': 0.5, 'variation': 0.4}
    ]
    
    phrases = []
    for params_dict in parameter_sets:
        print(f"\n   Generating phrase: {params_dict['name']}")
        
        # Update parameters
        for param, value in params_dict.items():
            if param != 'name':
                interactive_gen.update_parameter(param, value)
        
        # Generate phrase
        start_time = time.time()
        phrase = interactive_gen.generate_interactive_phrase()
        gen_time = time.time() - start_time
        
        phrases.append(phrase)
        print(f"   Generated in {gen_time*1000:.1f}ms")
        print(f"   Length: {len(phrase)/44100:.2f}s")
    
    # Combine phrases into a demo track
    print("\n7. Combining phrases into demo track...")
    # Add small gaps between phrases
    gap = np.zeros(int(0.5 * 44100))  # 0.5 second gap
    demo_track = np.concatenate([phrases[0], gap, phrases[1], gap, phrases[2]])
    
    # Save demo track
    demo_filename = f"accelerated_demo_phrases_{manager.backend_type.value}.wav"
    demo_track_int = (demo_track * 32767).astype(np.int16)
    wavfile.write(demo_filename, 44100, demo_track_int)
    print(f"   Saved to: {demo_filename}")
    
    # Performance comparison
    print("\n8. Performance Comparison (Network Operations):")
    
    # Create a test network
    test_network = AcceleratedMusicalNetwork(
        name="performance_test",
        base_freq=440.0,
        mode='Ionian'
    )
    test_network.create_harmonic_nodes([1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8], amplitude=0.1)
    test_network.create_modal_connections("all_to_all", weight_scale=0.1)
    
    # Benchmark audio generation
    durations = [0.1, 0.5, 1.0]  # seconds
    for duration in durations:
        start_time = time.time()
        audio = test_network.generate_audio_accelerated(duration, 44100)
        gen_time = time.time() - start_time
        print(f"   {duration}s audio: {gen_time*1000:.1f}ms ({duration/gen_time:.1f}x realtime)")
    
    print("\n9. Advanced Features Available:")
    print("   - GPU acceleration (CUDA/Metal)")
    print("   - Real-time mode morphing")
    print("   - Chord progression generation")
    print("   - Multi-scale temporal processing")
    print("   - Context-aware mode detection")
    print("   - Batch signal processing")
    
    # Plot the generation history
    if generator.generation_history['modes']:
        print("\n10. Visualizing generation history...")
        generator.plot_generation_history()
    
    print("\n" + "=" * 70)
    print("Demo completed! Check the generated audio files.")
    print("=" * 70)


def benchmark_backends():
    """Compare performance across different backends"""
    print("\nBackend Performance Benchmark")
    print("-" * 40)
    
    from synthnn.performance import BackendType
    
    # Test parameters
    test_duration = 2.0  # seconds
    sample_rate = 44100
    
    results = {}
    
    for backend_type in [BackendType.CPU, BackendType.METAL, BackendType.CUDA]:
        try:
            # Force specific backend
            manager = BackendManager()
            if backend_type in manager.available_backends:
                manager.backend_type = backend_type
                
                # Create network
                network = AcceleratedMusicalNetwork(
                    name=f"benchmark_{backend_type.value}",
                    base_freq=440.0,
                    mode='Dorian'
                )
                network.create_harmonic_nodes([1, 9/8, 6/5, 4/3, 3/2, 5/3, 9/5], amplitude=0.1)
                network.create_modal_connections("nearest_neighbor", weight_scale=0.15)
                
                # Benchmark
                start_time = time.time()
                audio = network.generate_audio_accelerated(test_duration, sample_rate)
                gen_time = time.time() - start_time
                
                results[backend_type.value] = {
                    'time': gen_time,
                    'speedup': test_duration / gen_time
                }
                
                print(f"{backend_type.value:>6}: {gen_time*1000:>6.1f}ms ({test_duration/gen_time:>5.1f}x realtime)")
        except Exception as e:
            print(f"{backend_type.value:>6}: Not available - {str(e)}")
    
    return results


if __name__ == "__main__":
    # Run the main demo
    demonstrate_accelerated_music_generation()
    
    # Run backend benchmark if multiple backends available
    manager = BackendManager()
    if len(manager.available_backends) > 1:
        print("\n" + "=" * 70)
        benchmark_backends() 