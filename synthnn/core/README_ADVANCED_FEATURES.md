# Advanced Features in SynthNN

This document describes three advanced features that extend SynthNN's capabilities into new dimensions of complexity and intelligence.

## 🌊 4D Resonance Fields

The 4D Resonance Fields module (`resonance_field.py`) extends resonant networks into spatial dimensions, enabling wave propagation, interference patterns, and emergent spatial dynamics. The "4D" represents 3D space + time.

### Key Concepts

- **Spatial Resonant Nodes**: Nodes that exist at specific 3D positions and radiate waves
- **Wave Propagation**: Waves travel through the field at finite speed following the wave equation
- **Boundary Conditions**: Control how waves behave at field edges (absorbing, reflecting, periodic, radiating)
- **Resonant Cavities**: Regions with different medium properties that can trap and amplify waves

### Usage Example

```python
from synthnn.core import ResonanceField4D, SpatialResonantNode, BoundaryCondition

# Create a 3D resonance field
field = ResonanceField4D(
    dimensions=(50, 50, 50),      # 50x50x50 grid
    resolution=1.0,               # 1 meter per grid point
    wave_speed=343.0,             # Speed of sound in air
    boundary_condition=BoundaryCondition.ABSORBING
)

# Add spatial nodes
node = SpatialResonantNode(
    "oscillator",
    position=np.array([25, 25, 25]),  # Center of field
    frequency=440.0,                   # A4 note
    amplitude=1.0,
    radiation_pattern="omnidirectional"
)
field.add_spatial_node(node)

# Create a resonant cavity
field.create_resonant_cavity(
    center=np.array([25, 25, 25]),
    dimensions=np.array([10, 10, 10]),
    impedance_ratio=10.0
)

# Simulate wave propagation
for _ in range(1000):
    field.step(0.001)  # 1ms time steps

# Extract 2D slices for visualization
xy_slice = field.extract_slice('z', 25)  # Horizontal slice at z=25
```

### Applications

- **Acoustic Modeling**: Simulate sound propagation in rooms and concert halls
- **Neural Field Theory**: Model spatially extended neural networks
- **Wave-based Computing**: Perform computations using wave interference
- **Holographic Storage**: Store patterns as interference patterns

## 🧠 Collective Intelligence Framework

The Collective Intelligence module (`collective_intelligence.py`) enables multiple resonant networks to work together as a collective, exhibiting swarm intelligence, distributed decision-making, and emergent behaviors.

### Key Concepts

- **Network Roles**: Each network has a specialized role with influence weights
- **Communication Modes**: Networks can communicate via phase coupling, frequency modulation, or direct signals
- **Consensus Methods**: Decisions made through synchronization, voting, emergence, or hierarchy
- **Distributed Memory**: Patterns stored across multiple networks for redundancy

### Usage Example

```python
from synthnn.core import (
    CollectiveIntelligence, CommunicationMode,
    ConsensusMethod, NetworkRole, ResonantNetwork
)

# Create collective
collective = CollectiveIntelligence(
    name="decision_maker",
    communication_mode=CommunicationMode.PHASE_COUPLING,
    consensus_method=ConsensusMethod.VOTING
)

# Add specialized networks
sensor_network = ResonantNetwork("sensors")
# ... configure sensor_network ...

role = NetworkRole(
    name="sensor",
    frequency_band=(1.0, 10.0),
    influence_weight=1.5,
    receptivity=0.8,
    specialization="sensing"
)
collective.add_network("sensor", sensor_network, role)

# Connect networks
collective.connect_networks("sensor", "processor", weight=2.0)

# Make collective decisions
options = ["action_a", "action_b", "action_c"]
decision = collective.make_collective_decision(options)

# Store and recall patterns
collective.store_pattern("important_pattern", pattern_data)
recalled = collective.recall_pattern("important_pattern")
```

### Applications

- **Distributed AI**: Create AI systems with specialized sub-networks
- **Consensus Algorithms**: Novel approaches to distributed consensus
- **Swarm Robotics**: Coordinate multiple agents through resonance
- **Brain Modeling**: Model interactions between brain regions

## 🧬 Evolutionary Resonance

The Evolutionary Resonance module (`evolutionary_resonance.py`) implements genetic algorithms for resonant networks, allowing them to evolve, adapt, and speciate based on fitness criteria.

### Key Concepts

- **Genomes**: Genetic representation of network topology and parameters
- **Species**: Groups of similar networks that can interbreed
- **Fitness Metrics**: Harmony, efficiency, information capacity, adaptability, creativity, memory
- **Mutations**: Changes to frequencies, connections, topology
- **Crossover**: Combining parent networks to create offspring

### Usage Example

```python
from synthnn.core import EvolutionaryResonance, FitnessMetric

# Create evolutionary system
evo = EvolutionaryResonance(
    population_size=50,
    fitness_metric=FitnessMetric.HARMONY,
    mutation_rate=0.1,
    crossover_rate=0.7,
    speciation_threshold=0.3
)

# Initialize random population
evo.initialize_population()

# Evolve for multiple generations
for generation in range(100):
    evo.evolve_generation()

    # Get best network
    best = evo.get_best_genome()
    print(f"Gen {generation}: Best fitness = {best.fitness:.3f}")

# Extract evolved network
best_network = best.to_network()

# Custom fitness function
def my_fitness(network):
    # Evaluate network for specific task
    return score

evo_custom = EvolutionaryResonance(
    fitness_metric=FitnessMetric.CUSTOM,
    custom_fitness=my_fitness
)
```

### Fitness Metrics

1. **HARMONY**: Optimizes for consonant frequency ratios and synchronization
2. **EFFICIENCY**: Minimizes energy loss while maintaining function
3. **INFORMATION**: Maximizes information processing capacity
4. **ADAPTABILITY**: Ability to recover from perturbations
5. **CREATIVITY**: Generates diverse, complex patterns
6. **MEMORY**: Pattern storage and recall capacity
7. **CUSTOM**: User-defined fitness function

### Applications

- **Network Optimization**: Automatically find optimal network configurations
- **Music Generation**: Evolve networks that produce harmonic music
- **Problem Solving**: Evolve networks to solve specific tasks
- **Artificial Life**: Study emergence and speciation in artificial systems

## 🔗 Combining the Features

These three features can be combined for even more powerful applications:

### Example: Evolving Collective Spatial Intelligence

```python
# Create evolutionary system with collective intelligence
evo = EvolutionaryResonance(
    population_size=30,
    fitness_metric=FitnessMetric.CUSTOM,
    use_collective=True  # Enable collective intelligence
)

# Custom fitness that evaluates collective behavior in space
def spatial_collective_fitness(network):
    # Add network to a collective in a spatial field
    collective = CollectiveIntelligence(use_spatial_field=True)
    collective.add_network("test", network, position=np.random.randn(3)*10)

    # Evaluate collective's ability to synchronize spatially
    for _ in range(100):
        collective.step(0.01)

    # Return spatial coherence as fitness
    return collective.spatial_field.get_field_statistics()['coherence']

evo.custom_fitness = spatial_collective_fitness

# Evolve networks that work well together in space
evo.initialize_population()
for gen in range(50):
    evo.evolve_generation()
```

### Example: Emotional Resonance in 4D Fields

```python
from synthnn.core import EmotionalResonanceEngine, EmotionCategory

# Create emotional engine
emotion_engine = EmotionalResonanceEngine()

# Create spatial field for emotions
field = ResonanceField4D(dimensions=(40, 40, 40))

# Map emotions to spatial regions
emotions = [EmotionCategory.JOY, EmotionCategory.SADNESS, EmotionCategory.CALM]
positions = [np.array([10, 20, 20]), np.array([30, 20, 20]), np.array([20, 20, 20])]

for emotion, pos in zip(emotions, positions):
    # Create emotional network
    network = emotion_engine.create_emotional_network(emotion, intensity=0.8)

    # Add to spatial field
    for node_id, node in network.nodes.items():
        spatial_node = SpatialResonantNode(
            f"{emotion.value}_{node_id}",
            position=pos + np.random.randn(3)*2,
            frequency=node.frequency,
            amplitude=node.amplitude
        )
        field.add_spatial_node(spatial_node)

# Emotions propagate and interact through space
for _ in range(1000):
    field.step(0.01)
```

## 🚀 Performance Considerations

### 4D Resonance Fields

- Computational complexity: O(n³) for n×n×n grid
- Memory usage: ~8 bytes per grid point per field variable
- Use smaller grids or adaptive resolution for real-time applications

### Collective Intelligence

- Scales with number of networks and connections
- Communication overhead can be reduced with sparse topologies
- Consider using GPU acceleration for large collectives

### Evolutionary Resonance

- Population size vs. diversity trade-off
- Parallel evaluation of fitness can speed up evolution
- Save/load populations to continue evolution across sessions

## 📚 Further Reading

- **Wave Physics**: See `docs/wave_propagation.md` for mathematical details
- **Collective Behavior**: Read about emergent intelligence in `docs/emergence.md`
- **Evolutionary Algorithms**: Learn about NEAT and genetic algorithms
- **Integration Guide**: See `docs/advanced_integration.md` for combining features

## 🎯 Next Steps

1. Run the demo: `python demos/advanced_features_demo.py`
2. Experiment with different parameters
3. Create your own fitness functions
4. Combine features for your specific use case
5. Share your discoveries with the community!

These advanced features open up entirely new possibilities for resonant computing, from spatial wave computers to evolving collective intelligence. The only limit is your imagination!
