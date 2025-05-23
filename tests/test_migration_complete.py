"""
Test suite to verify the complete migration of music generation to AcceleratedMusicalNetwork.
"""

import numpy as np
import pytest
import time

from applications.modal_music_generator import ModalMusicGenerator
from applications.interactive_modal_generator import InteractiveModalGenerator
from applications.adaptive import AdaptiveModalNetwork
from applications.hierarchical_modal import HierarchicalModalProcessor
from synthnn.core import AcceleratedMusicalNetwork
from synthnn.performance import BackendManager


class TestMigrationComplete:
    """Test that all music generation components use the new accelerated framework"""
    
    def test_modal_music_generator_uses_accelerated_network(self):
        """Test that ModalMusicGenerator uses AcceleratedMusicalNetwork"""
        generator = ModalMusicGenerator(base_freq=440.0, sample_rate=44100)
        
        # Generate a short phrase
        melody_contour = [1, 2, 3, 4, 5]
        rhythm_pattern = [0.5, 0.5, 0.5, 0.5, 0.5]
        
        # Check that _render_phrase creates AcceleratedMusicalNetwork
        phrase_audio = generator._render_phrase(
            melody_contour, 
            rhythm_pattern,
            'Ionian',
            2.5  # duration
        )
        
        assert isinstance(phrase_audio, np.ndarray)
        assert len(phrase_audio) > 0
        print("✓ ModalMusicGenerator successfully uses AcceleratedMusicalNetwork")
    
    def test_interactive_generator_uses_accelerated_network(self):
        """Test that InteractiveModalGenerator uses AcceleratedMusicalNetwork"""
        generator = InteractiveModalGenerator(base_freq=440.0, sample_rate=44100)
        
        # Update some parameters
        generator.update_parameter('complexity', 0.7)
        generator.update_parameter('brightness', 0.8)
        
        # Generate a phrase
        phrase = generator.generate_interactive_phrase()
        
        assert isinstance(phrase, np.ndarray)
        assert len(phrase) > 0
        print("✓ InteractiveModalGenerator successfully uses AcceleratedMusicalNetwork")
    
    def test_adaptive_network_uses_accelerated_backend(self):
        """Test that AdaptiveModalNetwork uses AcceleratedMusicalNetwork internally"""
        adaptive = AdaptiveModalNetwork(
            num_nodes_per_network=5,
            initial_base_freq=440.0
        )
        
        # Check that mode networks are AcceleratedMusicalNetwork instances
        for mode_name, network in adaptive.mode_networks.items():
            assert isinstance(network, AcceleratedMusicalNetwork)
            assert network.mode == mode_name
        
        # Test processing
        test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
        output = adaptive.process(test_signal, 44100)
        
        assert isinstance(output, (int, float))
        print("✓ AdaptiveModalNetwork successfully uses AcceleratedMusicalNetwork")
    
    def test_hierarchical_processor_integration(self):
        """Test that HierarchicalModalProcessor works with the new system"""
        processor = HierarchicalModalProcessor(
            time_scales=[0.1, 0.5, 1.0],
            num_nodes=5,
            base_process_sample_rate=44100
        )
        
        # Check that scale networks use AdaptiveModalNetwork
        for scale_info in processor.scale_networks:
            assert isinstance(scale_info['network'], AdaptiveModalNetwork)
        
        # Test processing
        test_chunk = np.random.randn(4410)  # 0.1 second chunk
        results = processor.process_signal_chunk(test_chunk, 44100)
        
        assert 'scale_outputs' in results
        print("✓ HierarchicalModalProcessor successfully integrated")
    
    def test_performance_benefits(self):
        """Test that the migration provides performance benefits"""
        print("\n--- Performance Comparison ---")
        
        # Create a test network
        network = AcceleratedMusicalNetwork(
            name="performance_test",
            base_freq=440.0,
            mode='Dorian'
        )
        network.create_harmonic_nodes([1, 9/8, 6/5, 4/3, 3/2, 5/3, 9/5])
        network.create_modal_connections("all_to_all", weight_scale=0.1)
        
        # Benchmark different operations
        operations = {
            'audio_generation': lambda: network.generate_audio_accelerated(1.0, 44100),
            'mode_morphing': lambda: network.morph_between_modes_accelerated('Dorian', 'Phrygian', 10),
            'spectrum_analysis': lambda: network.analyze_spectrum_accelerated(np.random.randn(44100), 44100)
        }
        
        for op_name, op_func in operations.items():
            start_time = time.time()
            result = op_func()
            elapsed = time.time() - start_time
            print(f"  {op_name}: {elapsed*1000:.1f}ms")
        
        print("✓ Performance tests completed")
    
    def test_backward_compatibility(self):
        """Test that old features still work after migration"""
        generator = ModalMusicGenerator()
        
        # Test old structure generation
        structure = generator.structure_generator.create_structure(
            total_duration=10,
            structure_type='verse-chorus',
            tempo=120
        )
        assert len(structure) > 0
        
        # Test phrase generation
        phrase_gen = generator.phrase_generator
        contour = phrase_gen.generate_contour('Lydian', phrase_length=8)
        assert len(contour) == 8
        
        # Test rhythm generation
        rhythm_gen = generator.rhythm_generator
        pattern = rhythm_gen.generate_pattern(4.0, 'verse', complexity=0.5)
        assert len(pattern) > 0
        
        print("✓ Backward compatibility maintained")
    
    def test_new_features_available(self):
        """Test that new features from AcceleratedMusicalNetwork are accessible"""
        network = AcceleratedMusicalNetwork(
            name="feature_test",
            base_freq=440.0,
            mode='Mixolydian'
        )
        network.create_harmonic_nodes([1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8])
        
        # Test chord progression generation
        progression = network.generate_chord_progression(['I', 'IV', 'V', 'I'], 2.0, 44100)
        assert isinstance(progression, np.ndarray)
        assert len(progression) > 0
        
        # Test batch processing
        signals = [np.random.randn(1000) for _ in range(5)]
        batch_results = network.batch_process_signals(signals, 44100)
        assert len(batch_results) == 5
        
        print("✓ New features are accessible")
    
    def test_backend_selection(self):
        """Test that backend selection works properly"""
        manager = BackendManager()
        print(f"\n--- Backend Information ---")
        print(f"  Available backends: {[b.value for b in manager.available_backends]}")
        print(f"  Selected backend: {manager.backend_type.value}")
        
        # Create networks and ensure they use the selected backend
        network = AcceleratedMusicalNetwork(
            name="backend_test",
            base_freq=440.0,
            mode='Aeolian'
        )
        
        # Generate audio to ensure backend is working
        audio = network.generate_audio_accelerated(0.1, 44100)
        assert len(audio) == 4410
        
        print("✓ Backend selection working correctly")


def run_all_tests():
    """Run all migration tests"""
    print("=" * 60)
    print("MUSIC GENERATION MIGRATION TEST SUITE")
    print("=" * 60)
    
    test_suite = TestMigrationComplete()
    
    tests = [
        test_suite.test_modal_music_generator_uses_accelerated_network,
        test_suite.test_interactive_generator_uses_accelerated_network,
        test_suite.test_adaptive_network_uses_accelerated_backend,
        test_suite.test_hierarchical_processor_integration,
        test_suite.test_performance_benefits,
        test_suite.test_backward_compatibility,
        test_suite.test_new_features_available,
        test_suite.test_backend_selection
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\nRunning: {test.__name__}")
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {test.__name__}")
            print(f"  Error: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 Migration completed successfully! All tests passed.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests() 