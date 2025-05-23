#!/usr/bin/env python3
"""
Advanced Features Demo for SynthNN

Demonstrates:
1. 4D Resonance Fields - Wave propagation in 3D space
2. Collective Intelligence - Multiple networks making decisions together
3. Evolutionary Resonance - Networks evolving to optimize for harmony

Run this demo to see all three advanced features in action!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from synthnn.core import (
    ResonantNetwork, ResonantNode,
    ResonanceField4D, SpatialResonantNode, BoundaryCondition,
    CollectiveIntelligence, CommunicationMode, ConsensusMethod, NetworkRole,
    EvolutionaryResonance, FitnessMetric
)


def demo_4d_resonance_field():
    """Demonstrate 4D Resonance Fields with wave propagation."""
    print("\n" + "="*60)
    print("🌊 4D RESONANCE FIELD DEMO")
    print("="*60)
    
    # Create a resonance field
    print("\n1. Creating 4D resonance field (30x30x30)...")
    field = ResonanceField4D(
        dimensions=(30, 30, 30),
        resolution=0.5,  # 0.5 meters per grid point
        wave_speed=50.0,  # Slower for visualization
        boundary_condition=BoundaryCondition.ABSORBING
    )
    
    # Add some spatial nodes
    print("\n2. Adding resonant nodes at different positions...")
    
    # Central oscillator
    central_node = SpatialResonantNode(
        "central",
        position=np.array([7.5, 7.5, 7.5]),  # Center of field
        frequency=2.0,
        amplitude=1.0,
        radiation_pattern="omnidirectional"
    )
    field.add_spatial_node(central_node)
    
    # Corner oscillators with different frequencies
    corners = [
        ([2, 2, 2], 1.5, "node1"),
        ([13, 2, 2], 2.5, "node2"),
        ([2, 13, 2], 3.0, "node3"),
        ([13, 13, 2], 1.0, "node4")
    ]
    
    for pos, freq, node_id in corners:
        node = SpatialResonantNode(
            node_id,
            position=np.array(pos),
            frequency=freq,
            amplitude=0.5,
            radiation_pattern="dipole"
        )
        field.add_spatial_node(node)
    
    # Create a resonant cavity
    print("\n3. Creating a resonant cavity...")
    field.create_resonant_cavity(
        center=np.array([7.5, 7.5, 7.5]),
        dimensions=np.array([4, 4, 4]),
        impedance_ratio=5.0
    )
    
    # Simulate wave propagation
    print("\n4. Simulating wave propagation...")
    
    # Create figure for visualization
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("4D Resonance Field - Wave Propagation")
    
    # Function to update visualization
    def update_viz(frame):
        # Step the field
        for _ in range(5):  # Multiple steps per frame
            field.step()
        
        # Get slices for visualization
        xy_slice = field.extract_slice('z', 15)  # Middle z-plane
        xz_slice = field.extract_slice('y', 15)  # Middle y-plane
        yz_slice = field.extract_slice('x', 15)  # Middle x-plane
        
        # Clear axes
        for ax in axes.flat:
            ax.clear()
        
        # Plot slices
        im1 = axes[0, 0].imshow(xy_slice, cmap='seismic', vmin=-0.5, vmax=0.5)
        axes[0, 0].set_title(f'XY Plane (z=15), t={field.time:.2f}')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        
        im2 = axes[0, 1].imshow(xz_slice, cmap='seismic', vmin=-0.5, vmax=0.5)
        axes[0, 1].set_title('XZ Plane (y=15)')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Z')
        
        im3 = axes[1, 0].imshow(yz_slice, cmap='seismic', vmin=-0.5, vmax=0.5)
        axes[1, 0].set_title('YZ Plane (x=15)')
        axes[1, 0].set_xlabel('Y')
        axes[1, 0].set_ylabel('Z')
        
        # Plot field statistics
        stats = field.get_field_statistics()
        axes[1, 1].text(0.1, 0.8, f"Total Energy: {stats['total_energy']:.2f}", fontsize=12)
        axes[1, 1].text(0.1, 0.6, f"Max Amplitude: {stats['max_amplitude']:.3f}", fontsize=12)
        axes[1, 1].text(0.1, 0.4, f"Coherence: {stats['coherence']:.3f}", fontsize=12)
        axes[1, 1].text(0.1, 0.2, f"Entropy: {stats['entropy']:.2f}", fontsize=12)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Field Statistics')
        
        plt.tight_layout()
        return [im1, im2, im3]
    
    # Create animation
    print("\n5. Visualizing wave propagation (close window to continue)...")
    anim = FuncAnimation(fig, update_viz, frames=100, interval=50, blit=False)
    plt.show()
    
    # Find resonant modes
    print("\n6. Finding resonant modes of the cavity...")
    modes = field.find_resonant_modes((1.0, 5.0), num_modes=3)
    
    print("\nTop 3 resonant modes:")
    for i, mode in enumerate(modes):
        print(f"  Mode {i+1}: Frequency = {mode['frequency']:.2f} Hz, Q-factor = {mode['q_factor']:.2f}")
    
    print("\n✓ 4D Resonance Field demo complete!")


def demo_collective_intelligence():
    """Demonstrate Collective Intelligence with decision-making."""
    print("\n" + "="*60)
    print("🧠 COLLECTIVE INTELLIGENCE DEMO")
    print("="*60)
    
    # Create collective
    print("\n1. Creating collective intelligence system...")
    collective = CollectiveIntelligence(
        name="decision_collective",
        communication_mode=CommunicationMode.PHASE_COUPLING,
        consensus_method=ConsensusMethod.VOTING,
        use_spatial_field=False  # Simpler without spatial field for this demo
    )
    
    # Create specialized networks
    print("\n2. Adding specialized networks...")
    
    # Pattern detector network
    detector = ResonantNetwork("pattern_detector")
    for i in range(5):
        node = ResonantNode(f"detect_{i}", frequency=2.0 + i*0.5)
        detector.add_node(node)
    detector.create_ring_topology()
    
    role_detector = NetworkRole(
        name="detector",
        frequency_band=(2.0, 4.5),
        influence_weight=1.5,
        receptivity=0.8,
        specialization="pattern_detection"
    )
    collective.add_network("detector", detector, role_detector)
    
    # Memory network
    memory = ResonantNetwork("memory_network")
    for i in range(8):
        node = ResonantNode(f"mem_{i}", frequency=1.0, damping=0.05)
        memory.add_node(node)
    memory.create_hub_topology(hub_indices=[0, 4])
    
    role_memory = NetworkRole(
        name="memory",
        frequency_band=(0.5, 2.0),
        influence_weight=1.0,
        receptivity=1.2,
        specialization="memory_storage"
    )
    collective.add_network("memory", memory, role_memory)
    
    # Decision maker network
    decider = ResonantNetwork("decision_maker")
    for i in range(6):
        node = ResonantNode(f"decide_{i}", frequency=3.0 + i*0.3)
        decider.add_node(node)
    decider.create_small_world_topology(k=2, p=0.3)
    
    role_decider = NetworkRole(
        name="decider",
        frequency_band=(3.0, 5.0),
        influence_weight=2.0,
        receptivity=0.6,
        specialization="decision_making"
    )
    collective.add_network("decider", decider, role_decider)
    
    # Connect networks
    print("\n3. Creating communication channels...")
    collective.connect_networks("detector", "memory", weight=1.5)
    collective.connect_networks("memory", "decider", weight=1.2)
    collective.connect_networks("detector", "decider", weight=0.8)
    
    # Store some patterns in collective memory
    print("\n4. Storing patterns in distributed memory...")
    patterns = {
        "alpha": np.array([1, 0, 1, 0, 1]),
        "beta": np.array([0, 1, 1, 0, 0]),
        "gamma": np.array([1, 1, 0, 1, 1])
    }
    
    for pattern_name, pattern_data in patterns.items():
        collective.store_pattern(pattern_name, pattern_data, pattern_type="test_pattern")
        print(f"  Stored pattern '{pattern_name}'")
    
    # Simulate collective dynamics
    print("\n5. Running collective dynamics...")
    
    # Track synchronization over time
    sync_history = []
    for i in range(200):
        collective.step(0.01)
        if i % 10 == 0:
            sync_history.append(collective.synchronization_index)
    
    # Make collective decisions
    print("\n6. Making collective decisions...")
    
    # Decision 1: Choose best pattern
    print("\n  Decision 1: Which pattern to use?")
    options = ["alpha", "beta", "gamma", "delta"]
    decision = collective.make_collective_decision(options)
    print(f"  Collective decided: {decision}")
    
    # Change consensus method
    collective.consensus_method = ConsensusMethod.SYNCHRONIZATION
    
    # Decision 2: Action selection
    print("\n  Decision 2: What action to take?")
    actions = ["explore", "exploit", "wait", "communicate"]
    decision = collective.make_collective_decision(actions)
    print(f"  Collective decided: {decision}")
    
    # Test pattern recall
    print("\n7. Testing distributed memory recall...")
    for pattern_name in ["alpha", "gamma"]:
        recalled = collective.recall_pattern(pattern_name)
        if recalled is not None:
            original = patterns[pattern_name]
            similarity = np.corrcoef(original, recalled[:len(original)])[0, 1]
            print(f"  Pattern '{pattern_name}' recall similarity: {similarity:.3f}")
    
    # Visualize collective state
    print("\n8. Visualizing collective state...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot synchronization history
    ax1.plot(sync_history, linewidth=2)
    ax1.set_xlabel('Time Steps (x10)')
    ax1.set_ylabel('Synchronization Index')
    ax1.set_title('Collective Synchronization Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot network states
    viz_data = collective.visualize_collective_state()
    networks = list(viz_data['networks'].keys())
    metrics = ['mean_frequency', 'synchronization']
    
    x = np.arange(len(networks))
    width = 0.35
    
    freq_data = [viz_data['networks'][n]['mean_frequency'] for n in networks]
    sync_data = [viz_data['networks'][n]['synchronization'] for n in networks]
    
    ax2.bar(x - width/2, freq_data, width, label='Mean Frequency')
    ax2.bar(x + width/2, sync_data, width, label='Synchronization')
    ax2.set_xlabel('Network')
    ax2.set_ylabel('Value')
    ax2.set_title('Individual Network States')
    ax2.set_xticks(x)
    ax2.set_xticklabels(networks)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Collective Intelligence demo complete!")


def demo_evolutionary_resonance():
    """Demonstrate Evolutionary Resonance with network evolution."""
    print("\n" + "="*60)
    print("🧬 EVOLUTIONARY RESONANCE DEMO")
    print("="*60)
    
    # Create evolutionary system
    print("\n1. Creating evolutionary system...")
    evo_system = EvolutionaryResonance(
        population_size=20,
        fitness_metric=FitnessMetric.HARMONY,
        mutation_rate=0.15,
        crossover_rate=0.7,
        speciation_threshold=0.3
    )
    
    # Initialize population
    print("\n2. Initializing random population...")
    evo_system.initialize_population()
    
    # Evolution parameters
    num_generations = 20
    
    print(f"\n3. Evolving for {num_generations} generations...")
    print("   (Optimizing for harmonic resonance)")
    
    # Track best fitness
    best_fitness_history = []
    species_count_history = []
    
    # Evolution loop
    for gen in range(num_generations):
        # Evolve one generation
        evo_system.evolve_generation()
        
        # Get best genome
        best = evo_system.get_best_genome()
        if best:
            best_fitness_history.append(best.fitness)
        
        # Track species
        species_count_history.append(len(evo_system.species))
        
        # Print progress
        if gen % 5 == 0:
            print(f"\n  Generation {gen}:")
            print(f"    Best fitness: {best.fitness:.3f}")
            print(f"    Species count: {len(evo_system.species)}")
            
            # Show species info
            species_info = evo_system.get_species_info()
            for species_id, info in species_info.items():
                print(f"    {species_id}: {info['size']} members, "
                      f"avg fitness: {info['mean_fitness']:.3f}")
    
    # Get final best network
    print("\n4. Extracting best evolved network...")
    best_genome = evo_system.get_best_genome()
    best_network = best_genome.to_network()
    
    print(f"\n  Best network has:")
    print(f"    {len(best_network.nodes)} nodes")
    print(f"    {best_network.connections.number_of_edges()} connections")
    print(f"    Fitness: {best_genome.fitness:.3f}")
    print(f"    Mutations: {', '.join(best_genome.mutations[-3:]) if best_genome.mutations else 'None'}")
    
    # Test the best network
    print("\n5. Testing best network's harmonic properties...")
    
    # Run network and analyze
    signals = []
    for _ in range(500):
        best_network.step(0.01)
        signal = sum(node.oscillate(best_network.time) for node in best_network.nodes.values())
        signals.append(signal)
    
    # Frequency analysis
    from scipy.fft import fft, fftfreq
    
    fft_vals = fft(signals)
    freqs = fftfreq(len(signals), 0.01)
    
    # Find dominant frequencies
    power = np.abs(fft_vals[:len(signals)//2])
    dominant_freq_idx = np.argsort(power)[-5:]
    dominant_freqs = freqs[dominant_freq_idx]
    
    print("\n  Dominant frequencies:")
    for i, (idx, freq) in enumerate(zip(dominant_freq_idx[-5:], dominant_freqs[-5:])):
        if freq > 0:
            print(f"    {freq:.2f} Hz (power: {power[idx]:.1f})")
    
    # Check frequency ratios
    positive_freqs = [f for f in dominant_freqs if f > 0.1]
    if len(positive_freqs) >= 2:
        positive_freqs.sort()
        print("\n  Frequency ratios:")
        for i in range(len(positive_freqs)-1):
            ratio = positive_freqs[i+1] / positive_freqs[i]
            print(f"    {positive_freqs[i+1]:.2f} / {positive_freqs[i]:.2f} = {ratio:.3f}")
    
    # Visualize evolution results
    print("\n6. Visualizing evolution results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot fitness evolution
    axes[0, 0].plot(best_fitness_history, linewidth=2, color='green')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Best Fitness')
    axes[0, 0].set_title('Fitness Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot species count
    axes[0, 1].plot(species_count_history, linewidth=2, color='blue')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Number of Species')
    axes[0, 1].set_title('Species Diversity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot network signal
    axes[1, 0].plot(signals[-200:], linewidth=1, color='purple')
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Signal')
    axes[1, 0].set_title('Best Network Output Signal')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot frequency spectrum
    axes[1, 1].semilogy(freqs[:len(signals)//2], power, linewidth=1, color='red')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Power')
    axes[1, 1].set_title('Frequency Spectrum')
    axes[1, 1].set_xlim(0, 20)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save best network
    print("\n7. Saving evolution results...")
    evo_system.save_population("evolved_population.pkl")
    print("  Population saved to 'evolved_population.pkl'")
    
    print("\n✓ Evolutionary Resonance demo complete!")


def main():
    """Run all advanced feature demos."""
    print("\n" + "🌟"*30)
    print("SYNTHNN ADVANCED FEATURES DEMO")
    print("🌟"*30)
    print("\nThis demo showcases three cutting-edge features:")
    print("1. 4D Resonance Fields - Waves in 3D space + time")
    print("2. Collective Intelligence - Networks working together")
    print("3. Evolutionary Resonance - Networks that evolve")
    
    # Run demos
    demo_4d_resonance_field()
    demo_collective_intelligence()
    demo_evolutionary_resonance()
    
    print("\n" + "🎉"*30)
    print("ALL DEMOS COMPLETE!")
    print("🎉"*30)
    print("\nThese advanced features enable:")
    print("- Spatial wave propagation and interference")
    print("- Distributed decision-making and memory")
    print("- Self-optimizing networks through evolution")
    print("\nCombine them to create truly intelligent resonant systems!")


if __name__ == "__main__":
    main() 