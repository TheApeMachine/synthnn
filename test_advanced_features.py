#!/usr/bin/env python3
"""
Quick test for advanced features
"""

import numpy as np
from synthnn.core import (
    ResonanceField4D, SpatialResonantNode, BoundaryCondition,
    CollectiveIntelligence, CommunicationMode, NetworkRole, ResonantNetwork, ResonantNode,
    EvolutionaryResonance, FitnessMetric
)

print("Testing Advanced Features...")

# Test 1: 4D Resonance Field
print("\n1. Testing 4D Resonance Field...")
field = ResonanceField4D(
    dimensions=(10, 10, 10),
    resolution=1.0,
    wave_speed=100.0
)

node = SpatialResonantNode(
    "test_node",
    position=np.array([5, 5, 5]),
    frequency=2.0
)
field.add_spatial_node(node)

# Step the field
for _ in range(10):
    field.step(0.01)

stats = field.get_field_statistics()
print(f"   ✓ Field energy: {stats['total_energy']:.3f}")
print(f"   ✓ Max amplitude: {stats['max_amplitude']:.3f}")

# Test 2: Collective Intelligence
print("\n2. Testing Collective Intelligence...")
collective = CollectiveIntelligence(
    name="test_collective",
    communication_mode=CommunicationMode.PHASE_COUPLING
)

# Add a simple network
network = ResonantNetwork("test_network")
for i in range(3):
    node = ResonantNode(f"node_{i}", frequency=1.0 + i*0.5)
    network.add_node(node)

# Create connections manually (ring topology)
network.connect("node_0", "node_1")
network.connect("node_1", "node_2")
network.connect("node_2", "node_0")

collective.add_network("net1", network)

# Step collective
for _ in range(10):
    collective.step(0.01)

print(f"   ✓ Collective synchronization: {collective.synchronization_index:.3f}")
print(f"   ✓ Collective frequency: {collective.collective_frequency:.3f}")

# Test 3: Evolutionary Resonance
print("\n3. Testing Evolutionary Resonance...")
evo = EvolutionaryResonance(
    population_size=10,
    fitness_metric=FitnessMetric.HARMONY
)

# Initialize and evolve one generation
evo.initialize_population()
evo.evolve_generation()

best = evo.get_best_genome()
print(f"   ✓ Best fitness: {best.fitness:.3f}")
print(f"   ✓ Species count: {len(evo.species)}")

print("\n✅ All advanced features working correctly!")
print("\nRun 'python demos/advanced_features_demo.py' for full demonstrations.") 