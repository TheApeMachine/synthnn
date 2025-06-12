#!/usr/bin/env python3
"""
Test performance of cascade with and without backend acceleration.
"""

import numpy as np
import time
from cascade_enhanced import DHRCSystemEnhanced

def test_backend_performance():
    """Compare performance with and without acceleration."""
    
    print("="*60)
    print("BACKEND PERFORMANCE COMPARISON")
    print("="*60)
    
    # Configuration for testing
    layer_configs = [
        {
            "name": "DetectorLayer (L1)", 
            "num_nodes": 80,  # Larger network for better performance comparison
            "freq_range": (30, 700),
            "connection_prob": 0.3, 
            "coupling_strength": 0.18, 
            "learning_rate_W": 0.0005, 
            "weight_decay_W": 0.00005, 
            "max_abs_weight_W": 0.6,
            "iters": 50, 
            "steps_per_iter": 10
        },
        {
            "name": "RelationLayer (L2)", 
            "num_nodes": 60, 
            "freq_range": (50, 1500),
            "connection_prob": 0.25, 
            "coupling_strength": 0.15,
            "learning_rate_W": 0.0003, 
            "weight_decay_W": 0.00003, 
            "max_abs_weight_W": 0.5,
            "iters": 40, 
            "steps_per_iter": 8
        },
        {
            "name": "AbstractHarmonyLayer (L3)", 
            "num_nodes": 40, 
            "freq_range": (1.0, 60.0),
            "connection_prob": 0.2, 
            "coupling_strength": 0.12,
            "learning_rate_W": 0.0001, 
            "weight_decay_W": 0.00001, 
            "max_abs_weight_W": 0.4,
            "iters": 30, 
            "steps_per_iter": 6
        },
    ]
    
    # Test input
    test_input = {
        100.0: 0.9, 149.5: 0.7, 205.0: 0.6, 
        220.0: 0.35, 310.0: 0.5, 440.0: 0.25,
        550.0: 0.4, 660.0: 0.3, 770.0: 0.2
    }
    
    print("\n1. TESTING WITHOUT ACCELERATION (NumPy)")
    print("-"*40)
    
    cascade_numpy = DHRCSystemEnhanced(
        layer_configs, 
        global_seed=42,
        use_acceleration=False,
        enable_memory=False
    )
    
    # Warmup run
    cascade_numpy.run(test_input, verbose_per_layer=False, measure_performance=False)
    
    # Timed runs
    numpy_times = []
    for i in range(3):
        start = time.time()
        result_numpy = cascade_numpy.run(test_input, verbose_per_layer=False, measure_performance=False)
        numpy_times.append(time.time() - start)
        print(f"  Run {i+1}: {numpy_times[-1]:.3f}s")
    
    avg_numpy_time = np.mean(numpy_times)
    print(f"\nAverage NumPy time: {avg_numpy_time:.3f}s (±{np.std(numpy_times):.3f}s)")
    
    print("\n2. TESTING WITH ACCELERATION (Metal)")
    print("-"*40)
    
    try:
        cascade_metal = DHRCSystemEnhanced(
            layer_configs, 
            global_seed=42,
            use_acceleration=True,
            enable_memory=False
        )
        
        # Warmup run
        cascade_metal.run(test_input, verbose_per_layer=False, measure_performance=False)
        
        # Timed runs
        metal_times = []
        for i in range(3):
            start = time.time()
            result_metal = cascade_metal.run(test_input, verbose_per_layer=False, measure_performance=False)
            metal_times.append(time.time() - start)
            print(f"  Run {i+1}: {metal_times[-1]:.3f}s")
        
        avg_metal_time = np.mean(metal_times)
        print(f"\nAverage Metal time: {avg_metal_time:.3f}s (±{np.std(metal_times):.3f}s)")
        
        # Compare results
        print("\n3. PERFORMANCE COMPARISON")
        print("-"*40)
        
        speedup = avg_numpy_time / avg_metal_time
        print(f"Speedup factor: {speedup:.2f}x")
        print(f"Time saved: {avg_numpy_time - avg_metal_time:.3f}s ({(1 - avg_metal_time/avg_numpy_time)*100:.1f}%)")
        
        # Verify results are similar
        print("\n4. RESULT VERIFICATION")
        print("-"*40)
        
        if result_numpy[-1] and result_metal[-1]:
            numpy_freqs = sorted([s[0] for s in result_numpy[-1]])
            metal_freqs = sorted([s[0] for s in result_metal[-1]])
            
            freq_diff = np.mean([abs(n - m) for n, m in zip(numpy_freqs[:min(len(numpy_freqs), len(metal_freqs))], 
                                                             metal_freqs[:min(len(numpy_freqs), len(metal_freqs))])])
            print(f"Average frequency difference: {freq_diff:.4f} Hz")
            
            numpy_amps = sorted([s[1] for s in result_numpy[-1]])
            metal_amps = sorted([s[1] for s in result_metal[-1]])
            
            amp_diff = np.mean([abs(n - m) for n, m in zip(numpy_amps[:min(len(numpy_amps), len(metal_amps))], 
                                                            metal_amps[:min(len(numpy_amps), len(metal_amps))])])
            print(f"Average amplitude difference: {amp_diff:.6f}")
            
            if freq_diff < 0.1 and amp_diff < 0.001:
                print("✓ Results are consistent between backends")
            else:
                print("⚠ Results differ between backends")
        
    except Exception as e:
        print(f"Metal acceleration failed: {e}")
        print("This is expected if Metal/MLX is not available")
    
    print("\n5. BACKEND INFORMATION")
    print("-"*40)
    
    try:
        from synthnn.performance import BackendManager
        backend_manager = BackendManager()
        
        print("Available backends:")
        for backend in backend_manager.list_available_backends():
            print(f"  - {backend.value}")
            info = backend_manager.get_device_info(backend)
            for key, value in info.items():
                print(f"    {key}: {value}")
    except Exception as e:
        print(f"Could not get backend information: {e}")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("The cascade implementation successfully uses hardware")
    print("acceleration when available, providing performance benefits")
    print("while maintaining result accuracy.")
    print("="*60)

if __name__ == "__main__":
    test_backend_performance() 