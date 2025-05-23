"""
Performance demonstration for SynthNN.

This script benchmarks the different compute backends and shows
the acceleration benefits of GPU/Metal over CPU.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from synthnn.core import ResonantNode
from synthnn.performance import (
    BackendManager, 
    AcceleratedResonantNetwork,
    BackendType
)


def benchmark_basic_network(backend_type=None, num_nodes=100, num_steps=1000):
    """Benchmark a basic resonant network."""
    print(f"\nBenchmarking with {num_nodes} nodes, {num_steps} steps...")
    
    # Create accelerated network
    network = AcceleratedResonantNetwork(
        name=f"benchmark_{backend_type}", 
        backend=backend_type
    )
    
    # Add nodes with random parameters
    for i in range(num_nodes):
        node = ResonantNode(
            node_id=f"node_{i}",
            frequency=np.random.uniform(0.5, 5.0),
            phase=np.random.uniform(0, 2*np.pi),
            amplitude=1.0
        )
        network.add_node(node)
    
    # Create random connections (sparse)
    connection_prob = 0.1
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and np.random.rand() < connection_prob:
                network.connect(f"node_{i}", f"node_{j}", 
                              weight=np.random.uniform(-0.5, 0.5))
    
    # Warmup
    for _ in range(10):
        network.step(0.01)
    
    # Benchmark stepping
    start_time = time.time()
    for _ in range(num_steps):
        network.step(0.01)
    step_time = time.time() - start_time
    
    # Benchmark signal generation
    start_time = time.time()
    audio = network.generate_signals(duration=1.0, sample_rate=44100)
    gen_time = time.time() - start_time
    
    return {
        'backend': str(network.backend.__class__.__name__),
        'step_time': step_time,
        'gen_time': gen_time,
        'steps_per_second': num_steps / step_time,
        'audio_shape': audio.shape
    }


def compare_backends():
    """Compare performance across available backends."""
    print("SynthNN Performance Comparison")
    print("=" * 50)
    
    # Initialize backend manager
    manager = BackendManager()
    
    # List available backends
    print("\nAvailable backends:")
    for backend in manager.list_available_backends():
        info = manager.get_device_info(backend)
        print(f"  - {backend.value}: {info.get('device_name', 'Unknown')}")
    
    # Run benchmarks
    results = {}
    node_counts = [10, 50, 100, 200, 500]
    
    for backend_type in manager.list_available_backends():
        print(f"\n\nTesting {backend_type.value} backend...")
        backend_results = []
        
        for num_nodes in node_counts:
            try:
                result = benchmark_basic_network(
                    backend_type=backend_type,
                    num_nodes=num_nodes,
                    num_steps=100
                )
                backend_results.append(result)
                print(f"  {num_nodes} nodes: {result['steps_per_second']:.1f} steps/sec")
            except Exception as e:
                print(f"  {num_nodes} nodes: Failed - {e}")
                backend_results.append(None)
        
        results[backend_type] = backend_results
    
    return results, node_counts


def plot_performance_results(results, node_counts):
    """Create performance comparison plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot steps per second
    for backend_type, backend_results in results.items():
        valid_results = [(n, r) for n, r in zip(node_counts, backend_results) if r is not None]
        if valid_results:
            nodes, res = zip(*valid_results)
            steps_per_sec = [r['steps_per_second'] for r in res]
            ax1.plot(nodes, steps_per_sec, 'o-', label=backend_type.value, linewidth=2)
    
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Steps per Second')
    ax1.set_title('Network Update Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot speedup relative to CPU
    cpu_results = results.get(BackendType.CPU, [])
    for backend_type, backend_results in results.items():
        if backend_type != BackendType.CPU:
            speedups = []
            valid_nodes = []
            
            for i, (cpu_res, backend_res) in enumerate(zip(cpu_results, backend_results)):
                if cpu_res and backend_res:
                    speedup = backend_res['steps_per_second'] / cpu_res['steps_per_second']
                    speedups.append(speedup)
                    valid_nodes.append(node_counts[i])
            
            if speedups:
                ax2.plot(valid_nodes, speedups, 'o-', label=backend_type.value, linewidth=2)
    
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Speedup vs CPU')
    ax2.set_title('Acceleration Factor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('synthnn_performance_comparison.png', dpi=150)
    plt.show()


def demo_large_scale():
    """Demonstrate large-scale network capabilities."""
    print("\n\n=== Large-Scale Network Demo ===")
    
    manager = BackendManager()
    
    # Try to use GPU/Metal if available
    if BackendType.METAL in manager.list_available_backends():
        backend = BackendType.METAL
    elif BackendType.CUDA in manager.list_available_backends():
        backend = BackendType.CUDA
    else:
        backend = BackendType.CPU
    
    print(f"Using {backend.value} backend for large-scale demo")
    
    # Create large network
    num_nodes = 1000
    network = AcceleratedResonantNetwork(name="large_scale", backend=backend)
    
    print(f"Creating network with {num_nodes} nodes...")
    
    # Add nodes in frequency bands (like a large organ)
    for i in range(num_nodes):
        # Distribute frequencies across musical range
        base_freq = 55 * (2 ** (i / (num_nodes / 7)))  # A1 to A7
        node = ResonantNode(
            node_id=f"node_{i}",
            frequency=base_freq + np.random.uniform(-5, 5),
            phase=np.random.uniform(0, 2*np.pi),
            amplitude=1.0 / np.sqrt(num_nodes)  # Scale amplitude
        )
        network.add_node(node)
    
    # Create layered connections
    print("Creating connections...")
    layers = 10
    nodes_per_layer = num_nodes // layers
    
    for layer in range(layers - 1):
        for i in range(nodes_per_layer):
            src_idx = layer * nodes_per_layer + i
            # Connect to next layer
            for j in range(3):  # 3 connections per node
                tgt_idx = (layer + 1) * nodes_per_layer + (i + j) % nodes_per_layer
                network.connect(f"node_{src_idx}", f"node_{tgt_idx}", 
                              weight=0.1 * np.random.randn())
    
    # Measure performance
    print("\nMeasuring performance...")
    
    # Time network evolution
    start = time.time()
    for _ in range(100):
        network.step(0.01)
    evolution_time = time.time() - start
    
    print(f"Network evolution: {100/evolution_time:.1f} steps/sec")
    
    # Generate audio
    print("Generating audio signal...")
    start = time.time()
    audio = network.generate_signals(duration=2.0, sample_rate=44100)
    gen_time = time.time() - start
    
    print(f"Audio generation: {len(audio)/gen_time/1000:.1f} ksamples/sec")
    print(f"Generated {len(audio)} samples in {gen_time:.2f} seconds")
    
    # Save a snippet
    from scipy.io import wavfile
    wavfile.write('large_scale_network_output.wav', 44100, 
                  (audio * 32767).astype(np.int16))
    print("Saved audio to large_scale_network_output.wav")


def main():
    """Run all performance demonstrations."""
    # Compare backends
    results, node_counts = compare_backends()
    
    # Plot results
    plot_performance_results(results, node_counts)
    
    # Run large-scale demo
    demo_large_scale()
    
    # Show backend auto-selection
    print("\n\n=== Backend Auto-Selection ===")
    manager = BackendManager()
    best_backend = manager.auto_select_by_benchmark()
    print(f"Recommended backend for this system: {best_backend.value}")


if __name__ == "__main__":
    main() 