#!/usr/bin/env python3
"""
Demo of Enhanced Cascade Features
Shows the domain-neutral improvements added to cascade.py
"""

import numpy as np
import time
from cascade_enhanced import DHRCSystemEnhanced

def demo_enhanced_features():
    """Demonstrate the enhanced cascade features."""
    
    print("="*60)
    print("ENHANCED CASCADE FEATURES DEMO")
    print("="*60)
    
    # Configuration
    layer_configs = [
        {
            "name": "DetectorLayer (L1)", 
            "num_nodes": 40, 
            "freq_range": (30, 700),
            "connection_prob": 0.3, 
            "coupling_strength": 0.18, 
            "learning_rate_W": 0.0005, 
            "weight_decay_W": 0.00005, 
            "max_abs_weight_W": 0.6,
            "iters": 30,  # Reduced for demo
            "steps_per_iter": 5
        },
        {
            "name": "RelationLayer (L2)", 
            "num_nodes": 30, 
            "freq_range": (50, 1500),
            "connection_prob": 0.25, 
            "coupling_strength": 0.15,
            "learning_rate_W": 0.0003, 
            "weight_decay_W": 0.00003, 
            "max_abs_weight_W": 0.5,
            "iters": 25, 
            "steps_per_iter": 5
        },
        {
            "name": "AbstractHarmonyLayer (L3)", 
            "num_nodes": 20, 
            "freq_range": (1.0, 60.0),
            "fixed_node_frequencies": np.array([
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 48, 60
            ]) * 2.5,
            "connection_prob": 0.2, 
            "coupling_strength": 0.12,
            "learning_rate_W": 0.0001, 
            "weight_decay_W": 0.00001, 
            "max_abs_weight_W": 0.4,
            "iters": 20, 
            "steps_per_iter": 4
        },
    ]
    
    # Test inputs
    test_inputs = [
        {100.0: 0.9, 149.5: 0.7, 205.0: 0.6, 220.0: 0.35, 310.0: 0.5, 440.0: 0.25},
        {110.0: 0.85, 165.0: 0.65, 220.0: 0.55, 330.0: 0.45},  # Harmonic series
        {200.0: 0.9, 300.0: 0.7, 400.0: 0.5, 500.0: 0.3},      # Different pattern
    ]
    
    print("\n1. PATTERN MEMORY FEATURE")
    print("-"*40)
    
    # Create cascade with memory enabled
    cascade_with_memory = DHRCSystemEnhanced(
        layer_configs, 
        global_seed=42,
        use_acceleration=False,
        enable_memory=True
    )
    
    # Process multiple inputs
    for i, test_input in enumerate(test_inputs):
        print(f"\nProcessing input {i+1}...")
        result = cascade_with_memory.run(test_input, verbose_per_layer=False, measure_performance=False)
        
        if cascade_with_memory.global_memory:
            print(f"  Patterns in memory: {len(cascade_with_memory.global_memory.patterns)}")
            
            # Check for similar patterns
            if result[-1]:
                similar = cascade_with_memory.global_memory.find_similar_patterns(
                    result[-1], 
                    threshold=0.7
                )
                print(f"  Similar patterns found: {len(similar)}")
                for pattern_id, similarity in similar[:3]:
                    print(f"    - {pattern_id}: {similarity:.2%} similarity")
    
    print("\n2. PERFORMANCE MEASUREMENT")
    print("-"*40)
    
    # Create cascade without memory for performance testing
    cascade_perf = DHRCSystemEnhanced(
        layer_configs, 
        global_seed=42,
        use_acceleration=False,
        enable_memory=False
    )
    
    # Time multiple runs
    times = []
    for _ in range(3):
        start = time.time()
        cascade_perf.run(test_inputs[0], verbose_per_layer=False, measure_performance=False)
        times.append(time.time() - start)
    
    print(f"Average processing time: {np.mean(times):.3f}s (±{np.std(times):.3f}s)")
    
    print("\n3. PARAMETER OPTIMIZATION (Preview)")
    print("-"*40)
    
    print("The cascade can optimize these parameters:")
    print("  - coupling_strength: Controls inter-node coupling")
    print("  - learning_rate_W: Hebbian learning rate")
    print("  - weight_decay_W: Connection weight decay")
    print("  - harmonic_boost: Harmonic enhancement factor")
    print("  - dissonant_damping: Non-harmonic suppression")
    print("\nOptimization uses evolutionary algorithms with:")
    print("  - Population-based search")
    print("  - Crossover and mutation")
    print("  - Custom fitness metrics")
    
    # Small optimization demo (commented out as it takes time)
    # optimal_params = cascade_perf.optimize_parameters(
    #     test_inputs[:2], 
    #     fitness_metric='output_richness',
    #     generations=5
    # )
    
    print("\n4. ACCELERATION SUPPORT")
    print("-"*40)
    
    try:
        # Try to create accelerated cascade
        cascade_accel = DHRCSystemEnhanced(
            layer_configs, 
            global_seed=42,
            use_acceleration=True,
            enable_memory=False
        )
        print("✓ Acceleration framework integrated")
        print("  - Auto-detects best backend (CPU/GPU/Metal)")
        print("  - Transparent fallback to NumPy")
        print("  - Optimized phase coupling operations")
    except:
        print("✗ Acceleration not available in this environment")
        print("  - Would use GPU/Metal if available")
        print("  - Currently using NumPy backend")
    
    print("\n5. DOMAIN NEUTRALITY")
    print("-"*40)
    
    print("The enhanced cascade remains domain-neutral:")
    print("  ✓ Works with abstract frequency-amplitude-phase tuples")
    print("  ✓ No audio-specific assumptions")
    print("  ✓ Pattern memory stores generic harmonic signatures")
    print("  ✓ Optimization metrics are customizable")
    print("  ✓ Can be applied to:")
    print("    - Audio/music processing")
    print("    - Signal analysis")
    print("    - Pattern recognition")
    print("    - Any frequency-domain analysis")
    
    print("\n" + "="*60)
    print("SUMMARY: Enhanced cascade adds performance and")
    print("learning capabilities while maintaining generality")
    print("="*60)

if __name__ == "__main__":
    demo_enhanced_features() 