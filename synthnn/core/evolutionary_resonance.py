"""
Evolutionary Resonance for SynthNN

Implements evolutionary algorithms for resonant networks, allowing them to
evolve, adapt, and speciate based on fitness criteria. Networks can evolve
their topology, parameters, and behaviors to optimize for specific tasks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import copy
import random
from collections import defaultdict

from .resonant_network import ResonantNetwork
from .resonant_node import ResonantNode
from .collective_intelligence import CollectiveIntelligence, NetworkRole


class FitnessMetric(Enum):
    """Types of fitness metrics for evolution."""
    HARMONY = "harmony"              # Consonance and synchronization
    EFFICIENCY = "efficiency"        # Energy efficiency
    INFORMATION = "information"      # Information processing capacity
    ADAPTABILITY = "adaptability"    # Response to perturbations
    CREATIVITY = "creativity"        # Novel pattern generation
    MEMORY = "memory"               # Pattern storage capacity
    CUSTOM = "custom"               # User-defined fitness function


class MutationType(Enum):
    """Types of mutations that can occur."""
    NODE_FREQUENCY = "node_freq"     # Change node frequencies
    NODE_DAMPING = "node_damp"       # Change damping factors
    CONNECTION_WEIGHT = "conn_weight" # Change connection strengths
    ADD_NODE = "add_node"            # Add new nodes
    REMOVE_NODE = "remove_node"      # Remove nodes
    ADD_CONNECTION = "add_conn"      # Add connections
    REMOVE_CONNECTION = "remove_conn" # Remove connections
    TOPOLOGY = "topology"            # Major topology changes


@dataclass
class Genome:
    """
    Genetic representation of a resonant network.
    """
    network_id: str
    node_genes: Dict[str, Dict[str, float]]  # node_id -> {freq, phase, amp, damp}
    connection_genes: List[Tuple[str, str, float]]  # (from, to, weight)
    metadata: Dict[str, Any] = field(default_factory=dict)
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutations: List[str] = field(default_factory=list)
    
    def to_network(self) -> ResonantNetwork:
        """Convert genome to ResonantNetwork."""
        network = ResonantNetwork(name=self.network_id)
        
        # Create nodes
        for node_id, params in self.node_genes.items():
            node = ResonantNode(
                node_id=node_id,
                frequency=params['freq'],
                phase=params['phase'],
                amplitude=params['amp'],
                damping=params['damp']
            )
            network.add_node(node)
            
        # Create connections
        for from_id, to_id, weight in self.connection_genes:
            if from_id in network.nodes and to_id in network.nodes:
                network.connect(from_id, to_id, weight)
                
        return network
        
    @staticmethod
    def from_network(network: ResonantNetwork, network_id: str) -> 'Genome':
        """Create genome from ResonantNetwork."""
        node_genes = {}
        for node_id, node in network.nodes.items():
            node_genes[node_id] = {
                'freq': node.frequency,
                'phase': node.phase,
                'amp': node.amplitude,
                'damp': node.damping
            }
            
        connection_genes = []
        # network.connections is a dict with (source, target) keys
        for (from_id, to_id), connection in network.connections.items():
            weight = connection.weight
            connection_genes.append((from_id, to_id, weight))
            
        return Genome(
            network_id=network_id,
            node_genes=node_genes,
            connection_genes=connection_genes
        )


@dataclass
class Species:
    """
    A species is a group of similar genomes that can interbreed.
    """
    species_id: str
    representative: Genome  # Representative genome for comparison
    members: List[Genome] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)
    stagnation_counter: int = 0
    created_generation: int = 0
    
    def add_member(self, genome: Genome) -> None:
        """Add a genome to this species."""
        self.members.append(genome)
        
    def update_fitness(self) -> None:
        """Update species fitness based on members."""
        if self.members:
            avg_fitness = np.mean([g.fitness for g in self.members])
            self.fitness_history.append(avg_fitness)
            
            # Check for stagnation
            if len(self.fitness_history) > 5:
                recent = self.fitness_history[-5:]
                if max(recent) - min(recent) < 0.01:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0
                    
    def select_parents(self, num_parents: int) -> List[Genome]:
        """Select parents for reproduction based on fitness."""
        if len(self.members) <= num_parents:
            return self.members.copy()
            
        # Tournament selection
        parents = []
        for _ in range(num_parents):
            tournament = random.sample(self.members, min(3, len(self.members)))
            winner = max(tournament, key=lambda g: g.fitness)
            parents.append(winner)
            
        return parents


class EvolutionaryResonance:
    """
    Evolutionary system for resonant networks.
    """
    
    def __init__(self, population_size: int = 50,
                 fitness_metric: FitnessMetric = FitnessMetric.HARMONY,
                 custom_fitness: Optional[Callable] = None,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 speciation_threshold: float = 0.3,
                 use_collective: bool = False):
        """
        Initialize evolutionary system.
        
        Args:
            population_size: Number of networks in population
            fitness_metric: How to evaluate fitness
            custom_fitness: Custom fitness function if metric is CUSTOM
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            speciation_threshold: Distance threshold for speciation
            use_collective: Whether to use collective intelligence
        """
        self.population_size = population_size
        self.fitness_metric = fitness_metric
        self.custom_fitness = custom_fitness
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.speciation_threshold = speciation_threshold
        
        # Population
        self.population: List[Genome] = []
        self.species: Dict[str, Species] = {}
        self.generation = 0
        
        # Evolution history
        self.fitness_history = []
        self.species_history = []
        self.innovation_history = []
        
        # Collective intelligence integration
        self.use_collective = use_collective
        if use_collective:
            self.collective = CollectiveIntelligence(
                name="evolutionary_collective",
                use_spatial_field=True,
                field_dimensions=(30, 30, 30)
            )
        else:
            self.collective = None
            
        # Innovation tracking
        self.innovation_number = 0
        self.innovations = {}  # (from, to) -> innovation_number
        
    def initialize_population(self, template_network: Optional[ResonantNetwork] = None) -> None:
        """
        Initialize the population with random or template-based networks.
        
        Args:
            template_network: Optional template to base population on
        """
        for i in range(self.population_size):
            if template_network:
                # Create variation of template
                network = self._create_variation(template_network)
            else:
                # Create random network
                network = self._create_random_network()
                
            genome = Genome.from_network(network, f"genome_{i}")
            genome.generation = 0
            self.population.append(genome)
            
        # Initial speciation
        self._speciate()
        
    def _create_random_network(self) -> ResonantNetwork:
        """Create a random resonant network."""
        network = ResonantNetwork(name="random")
        
        # Random number of nodes (3-10)
        num_nodes = random.randint(3, 10)
        
        for i in range(num_nodes):
            node = ResonantNode(
                node_id=f"node_{i}",
                frequency=random.uniform(0.1, 10.0),
                phase=random.uniform(0, 2*np.pi),
                amplitude=random.uniform(0.5, 1.0),
                damping=random.uniform(0.01, 0.2)
            )
            network.add_node(node)
            
        # Random connections (ensure connected)
        nodes = list(network.nodes.keys())
        
        # Create a connected backbone
        for i in range(len(nodes) - 1):
            weight = random.uniform(0.1, 1.0)
            network.connect(nodes[i], nodes[i+1], weight)
            
        # Add some random connections
        num_extra = random.randint(0, num_nodes)
        for _ in range(num_extra):
            from_node = random.choice(nodes)
            to_node = random.choice(nodes)
            if from_node != to_node:
                weight = random.uniform(0.1, 1.0)
                network.connect(from_node, to_node, weight)
                
        return network
        
    def _create_variation(self, template: ResonantNetwork) -> ResonantNetwork:
        """Create a variation of a template network."""
        # Deep copy the template
        network = ResonantNetwork(name="variation")
        
        # Copy nodes with slight variations
        for node_id, node in template.nodes.items():
            new_node = ResonantNode(
                node_id=node_id,
                frequency=node.frequency * random.uniform(0.9, 1.1),
                phase=node.phase + random.uniform(-0.1, 0.1),
                amplitude=node.amplitude * random.uniform(0.95, 1.05),
                damping=node.damping * random.uniform(0.9, 1.1)
            )
            network.add_node(new_node)
            
        # Copy connections with slight variations
        for (from_id, to_id), connection in template.connections.items():
            weight = connection.weight * random.uniform(0.9, 1.1)
            network.connect(from_id, to_id, weight)
            
        return network
        
    def evolve_generation(self) -> None:
        """Run one generation of evolution."""
        # Step 1: Evaluate fitness
        self._evaluate_fitness()
        
        # Step 2: Update species fitness
        for species in self.species.values():
            species.update_fitness()
            
        # Step 3: Record statistics
        self._record_statistics()
        
        # Step 4: Selection and reproduction
        new_population = []
        
        for species_id, species in self.species.items():
            # Calculate offspring quota based on species fitness
            species_fitness = np.mean([g.fitness for g in species.members])
            total_fitness = sum(np.mean([g.fitness for g in s.members]) 
                              for s in self.species.values())
            
            if total_fitness > 0:
                offspring_quota = int(self.population_size * species_fitness / total_fitness)
            else:
                offspring_quota = self.population_size // len(self.species)
                
            # Select parents
            parents = species.select_parents(offspring_quota * 2)
            
            # Create offspring
            for i in range(0, len(parents) - 1, 2):
                if len(new_population) >= self.population_size:
                    break
                    
                parent1, parent2 = parents[i], parents[i+1]
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)
                    
                # Mutation
                if random.random() < self.mutation_rate:
                    self._mutate(child)
                    
                child.generation = self.generation + 1
                child.parent_ids = [parent1.network_id, parent2.network_id]
                new_population.append(child)
                
        # Step 5: Replace population
        self.population = new_population
        
        # Step 6: Re-speciate
        self._speciate()
        
        # Step 7: Update collective if used
        if self.use_collective:
            self._update_collective()
            
        self.generation += 1
        
    def _evaluate_fitness(self) -> None:
        """Evaluate fitness for all genomes in population."""
        for genome in self.population:
            network = genome.to_network()
            
            if self.fitness_metric == FitnessMetric.HARMONY:
                genome.fitness = self._evaluate_harmony(network)
            elif self.fitness_metric == FitnessMetric.EFFICIENCY:
                genome.fitness = self._evaluate_efficiency(network)
            elif self.fitness_metric == FitnessMetric.INFORMATION:
                genome.fitness = self._evaluate_information(network)
            elif self.fitness_metric == FitnessMetric.ADAPTABILITY:
                genome.fitness = self._evaluate_adaptability(network)
            elif self.fitness_metric == FitnessMetric.CREATIVITY:
                genome.fitness = self._evaluate_creativity(network)
            elif self.fitness_metric == FitnessMetric.MEMORY:
                genome.fitness = self._evaluate_memory(network)
            elif self.fitness_metric == FitnessMetric.CUSTOM and self.custom_fitness:
                genome.fitness = self.custom_fitness(network)
                
    def _evaluate_harmony(self, network: ResonantNetwork) -> float:
        """Evaluate network harmony and consonance."""
        # Run network for a period
        for _ in range(100):
            network.step(0.01)
            
        # Measure synchronization
        sync = network.measure_synchronization()
        
        # Measure frequency ratios (prefer simple ratios)
        frequencies = [node.frequency for node in network.nodes.values()]
        harmony_score = 0
        
        for i, freq1 in enumerate(frequencies):
            for freq2 in frequencies[i+1:]:
                ratio = max(freq1, freq2) / min(freq1, freq2)
                
                # Check for harmonic ratios
                harmonic_ratios = [1, 2, 3/2, 4/3, 5/4, 6/5, 5/3, 8/5]
                for hr in harmonic_ratios:
                    if abs(ratio - hr) < 0.1:
                        harmony_score += 1 / (1 + abs(ratio - hr))
                        
        # Normalize
        max_pairs = len(frequencies) * (len(frequencies) - 1) / 2
        if max_pairs > 0:
            harmony_score /= max_pairs
            
        # Combined fitness
        return 0.5 * sync + 0.5 * harmony_score
        
    def _evaluate_efficiency(self, network: ResonantNetwork) -> float:
        """Evaluate energy efficiency."""
        initial_energy = sum(node.amplitude**2 for node in network.nodes.values())
        
        # Run network
        for _ in range(100):
            network.step(0.01)
            
        final_energy = sum(node.amplitude**2 for node in network.nodes.values())
        
        # Efficiency is maintaining energy without loss
        if initial_energy > 0:
            efficiency = final_energy / initial_energy
        else:
            efficiency = 0
            
        # Also consider computational efficiency (fewer nodes/connections)
        complexity = len(network.nodes) + network.connections.number_of_edges()
        complexity_penalty = 1 / (1 + complexity / 20)
        
        return efficiency * complexity_penalty
        
    def _evaluate_information(self, network: ResonantNetwork) -> float:
        """Evaluate information processing capacity."""
        # Test with different input patterns
        test_patterns = [
            np.sin(np.linspace(0, 2*np.pi, 10)),
            np.cos(np.linspace(0, 4*np.pi, 10)),
            np.random.randn(10)
        ]
        
        responses = []
        for pattern in test_patterns:
            # Reset network
            for node in network.nodes.values():
                node.phase = 0
                node.amplitude = 1
                
            # Apply pattern as external force
            nodes = list(network.nodes.values())
            for i, value in enumerate(pattern[:len(nodes)]):
                nodes[i].amplitude = abs(value)
                nodes[i].phase = np.angle(value + 1j*0.1)
                
            # Let network process
            trajectory = []
            for _ in range(50):
                network.step(0.01)
                state = [node.amplitude * np.exp(1j*node.phase) 
                        for node in network.nodes.values()]
                trajectory.append(state)
                
            responses.append(trajectory)
            
        # Measure information content (variance in responses)
        info_score = 0
        for response in responses:
            response_array = np.array(response)
            variance = np.var(response_array)
            info_score += variance
            
        return info_score / len(test_patterns)
        
    def _evaluate_adaptability(self, network: ResonantNetwork) -> float:
        """Evaluate adaptability to perturbations."""
        # Baseline synchronization
        for _ in range(50):
            network.step(0.01)
        baseline_sync = network.measure_synchronization()
        
        # Apply perturbation
        for node in network.nodes.values():
            node.phase += random.uniform(-np.pi/2, np.pi/2)
            node.frequency *= random.uniform(0.8, 1.2)
            
        # Measure recovery
        recovery_scores = []
        for i in range(100):
            network.step(0.01)
            if i % 10 == 0:
                current_sync = network.measure_synchronization()
                recovery = current_sync / baseline_sync if baseline_sync > 0 else 0
                recovery_scores.append(recovery)
                
        # Adaptability is speed and quality of recovery
        if recovery_scores:
            final_recovery = recovery_scores[-1]
            recovery_speed = sum(recovery_scores) / len(recovery_scores)
            return 0.5 * final_recovery + 0.5 * recovery_speed
        else:
            return 0
            
    def _evaluate_creativity(self, network: ResonantNetwork) -> float:
        """Evaluate ability to generate novel patterns."""
        # Record patterns over time
        patterns = []
        
        for _ in range(200):
            network.step(0.01)
            
            # Record state pattern
            state = tuple(round(node.phase, 1) for node in network.nodes.values())
            patterns.append(state)
            
        # Creativity is diversity of patterns
        unique_patterns = len(set(patterns))
        pattern_diversity = unique_patterns / len(patterns)
        
        # Also measure pattern complexity
        if patterns:
            pattern_changes = sum(1 for i in range(1, len(patterns)) 
                                if patterns[i] != patterns[i-1])
            pattern_complexity = pattern_changes / len(patterns)
        else:
            pattern_complexity = 0
            
        return 0.5 * pattern_diversity + 0.5 * pattern_complexity
        
    def _evaluate_memory(self, network: ResonantNetwork) -> float:
        """Evaluate pattern storage capacity."""
        # Try to store patterns
        stored_patterns = []
        
        for i in range(5):
            # Create pattern
            pattern = [random.uniform(0, 2*np.pi) for _ in range(len(network.nodes))]
            
            # Store pattern by setting node phases
            for j, node in enumerate(network.nodes.values()):
                if j < len(pattern):
                    node.phase = pattern[j]
                    
            # Let network settle
            for _ in range(20):
                network.step(0.01)
                
            # Record settled state
            settled = [node.phase for node in network.nodes.values()]
            stored_patterns.append((pattern, settled))
            
        # Test recall
        recall_score = 0
        for original, stored in stored_patterns:
            # Set to stored state with noise
            for j, node in enumerate(network.nodes.values()):
                if j < len(stored):
                    node.phase = stored[j] + random.uniform(-0.1, 0.1)
                    
            # Let network evolve
            for _ in range(30):
                network.step(0.01)
                
            # Check if it recovered the pattern
            final = [node.phase for node in network.nodes.values()]
            
            # Calculate similarity
            similarity = 0
            for j in range(min(len(original), len(final))):
                phase_diff = abs(original[j] - final[j])
                similarity += np.cos(phase_diff)
                
            recall_score += similarity / len(original)
            
        return recall_score / len(stored_patterns) if stored_patterns else 0
        
    def _crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Perform crossover between two parent genomes."""
        child = Genome(
            network_id=f"genome_{self.generation}_{random.randint(1000, 9999)}",
            node_genes={},
            connection_genes=[]
        )
        
        # Crossover nodes
        all_nodes = set(parent1.node_genes.keys()) | set(parent2.node_genes.keys())
        
        for node_id in all_nodes:
            if node_id in parent1.node_genes and node_id in parent2.node_genes:
                # Both parents have this node - randomly choose
                if random.random() < 0.5:
                    child.node_genes[node_id] = copy.deepcopy(parent1.node_genes[node_id])
                else:
                    child.node_genes[node_id] = copy.deepcopy(parent2.node_genes[node_id])
            elif node_id in parent1.node_genes:
                # Only parent1 has this node
                if random.random() < 0.7:  # Bias toward including
                    child.node_genes[node_id] = copy.deepcopy(parent1.node_genes[node_id])
            else:
                # Only parent2 has this node
                if random.random() < 0.7:
                    child.node_genes[node_id] = copy.deepcopy(parent2.node_genes[node_id])
                    
        # Crossover connections
        parent1_conns = set((c[0], c[1]) for c in parent1.connection_genes)
        parent2_conns = set((c[0], c[1]) for c in parent2.connection_genes)
        all_conns = parent1_conns | parent2_conns
        
        for conn in all_conns:
            # Only include if both nodes exist in child
            if conn[0] in child.node_genes and conn[1] in child.node_genes:
                # Find the connection in parents
                conn1 = next((c for c in parent1.connection_genes if c[0] == conn[0] and c[1] == conn[1]), None)
                conn2 = next((c for c in parent2.connection_genes if c[0] == conn[0] and c[1] == conn[1]), None)
                
                if conn1 and conn2:
                    # Both parents have it - average weight
                    weight = (conn1[2] + conn2[2]) / 2
                    child.connection_genes.append((conn[0], conn[1], weight))
                elif conn1:
                    child.connection_genes.append(conn1)
                elif conn2:
                    child.connection_genes.append(conn2)
                    
        return child
        
    def _mutate(self, genome: Genome) -> None:
        """Apply mutations to a genome."""
        mutation_types = list(MutationType)
        mutation = random.choice(mutation_types)
        
        if mutation == MutationType.NODE_FREQUENCY:
            # Mutate node frequencies
            if genome.node_genes:
                node_id = random.choice(list(genome.node_genes.keys()))
                genome.node_genes[node_id]['freq'] *= random.uniform(0.8, 1.2)
                genome.mutations.append(f"freq_mut_{node_id}")
                
        elif mutation == MutationType.NODE_DAMPING:
            # Mutate damping
            if genome.node_genes:
                node_id = random.choice(list(genome.node_genes.keys()))
                genome.node_genes[node_id]['damp'] *= random.uniform(0.8, 1.2)
                genome.node_genes[node_id]['damp'] = np.clip(genome.node_genes[node_id]['damp'], 0.001, 0.5)
                genome.mutations.append(f"damp_mut_{node_id}")
                
        elif mutation == MutationType.CONNECTION_WEIGHT:
            # Mutate connection weight
            if genome.connection_genes:
                idx = random.randint(0, len(genome.connection_genes) - 1)
                old_conn = genome.connection_genes[idx]
                new_weight = old_conn[2] * random.uniform(0.5, 1.5)
                genome.connection_genes[idx] = (old_conn[0], old_conn[1], new_weight)
                genome.mutations.append(f"weight_mut_{old_conn[0]}_{old_conn[1]}")
                
        elif mutation == MutationType.ADD_NODE:
            # Add a new node
            new_node_id = f"node_{len(genome.node_genes)}_{random.randint(100, 999)}"
            genome.node_genes[new_node_id] = {
                'freq': random.uniform(0.1, 10.0),
                'phase': random.uniform(0, 2*np.pi),
                'amp': random.uniform(0.5, 1.0),
                'damp': random.uniform(0.01, 0.2)
            }
            
            # Connect to existing nodes
            if genome.node_genes:
                existing = random.choice(list(genome.node_genes.keys()))
                genome.connection_genes.append((existing, new_node_id, random.uniform(0.1, 1.0)))
                
            genome.mutations.append(f"add_node_{new_node_id}")
            
        elif mutation == MutationType.REMOVE_NODE:
            # Remove a node (if more than 2 nodes)
            if len(genome.node_genes) > 2:
                node_id = random.choice(list(genome.node_genes.keys()))
                del genome.node_genes[node_id]
                
                # Remove connections involving this node
                genome.connection_genes = [
                    c for c in genome.connection_genes 
                    if c[0] != node_id and c[1] != node_id
                ]
                
                genome.mutations.append(f"remove_node_{node_id}")
                
        elif mutation == MutationType.ADD_CONNECTION:
            # Add a new connection
            if len(genome.node_genes) >= 2:
                nodes = list(genome.node_genes.keys())
                from_node = random.choice(nodes)
                to_node = random.choice(nodes)
                
                if from_node != to_node:
                    # Check if connection already exists
                    exists = any(c[0] == from_node and c[1] == to_node 
                               for c in genome.connection_genes)
                    
                    if not exists:
                        weight = random.uniform(0.1, 1.0)
                        genome.connection_genes.append((from_node, to_node, weight))
                        
                        # Track innovation
                        innovation_key = (from_node, to_node)
                        if innovation_key not in self.innovations:
                            self.innovations[innovation_key] = self.innovation_number
                            self.innovation_number += 1
                            
                        genome.mutations.append(f"add_conn_{from_node}_{to_node}")
                        
        elif mutation == MutationType.REMOVE_CONNECTION:
            # Remove a connection (keep at least one)
            if len(genome.connection_genes) > 1:
                idx = random.randint(0, len(genome.connection_genes) - 1)
                removed = genome.connection_genes.pop(idx)
                genome.mutations.append(f"remove_conn_{removed[0]}_{removed[1]}")
                
    def _speciate(self) -> None:
        """Organize population into species."""
        # Clear existing species members
        for species in self.species.values():
            species.members = []
            
        # Assign genomes to species
        for genome in self.population:
            assigned = False
            
            for species_id, species in self.species.items():
                distance = self._calculate_distance(genome, species.representative)
                
                if distance < self.speciation_threshold:
                    species.add_member(genome)
                    assigned = True
                    break
                    
            if not assigned:
                # Create new species
                new_species_id = f"species_{len(self.species)}"
                new_species = Species(
                    species_id=new_species_id,
                    representative=genome,
                    created_generation=self.generation
                )
                new_species.add_member(genome)
                self.species[new_species_id] = new_species
                
        # Remove empty species
        self.species = {sid: s for sid, s in self.species.items() if s.members}
        
        # Update representatives
        for species in self.species.values():
            if species.members:
                # Choose member closest to average as representative
                species.representative = random.choice(species.members)
                
    def _calculate_distance(self, genome1: Genome, genome2: Genome) -> float:
        """Calculate genetic distance between two genomes."""
        distance = 0
        
        # Node differences
        nodes1 = set(genome1.node_genes.keys())
        nodes2 = set(genome2.node_genes.keys())
        
        # Disjoint nodes
        disjoint_nodes = len(nodes1 ^ nodes2)
        distance += disjoint_nodes * 0.5
        
        # Common nodes - parameter differences
        common_nodes = nodes1 & nodes2
        for node_id in common_nodes:
            params1 = genome1.node_genes[node_id]
            params2 = genome2.node_genes[node_id]
            
            freq_diff = abs(params1['freq'] - params2['freq']) / 10.0
            damp_diff = abs(params1['damp'] - params2['damp']) / 0.5
            
            distance += (freq_diff + damp_diff) * 0.1
            
        # Connection differences
        conns1 = set((c[0], c[1]) for c in genome1.connection_genes)
        conns2 = set((c[0], c[1]) for c in genome2.connection_genes)
        
        disjoint_conns = len(conns1 ^ conns2)
        distance += disjoint_conns * 0.3
        
        # Normalize by size
        total_elements = len(nodes1) + len(nodes2) + len(conns1) + len(conns2)
        if total_elements > 0:
            distance /= total_elements
            
        return distance
        
    def _record_statistics(self) -> None:
        """Record evolution statistics."""
        # Fitness statistics
        fitnesses = [g.fitness for g in self.population]
        self.fitness_history.append({
            'generation': self.generation,
            'mean': np.mean(fitnesses),
            'max': np.max(fitnesses),
            'min': np.min(fitnesses),
            'std': np.std(fitnesses)
        })
        
        # Species statistics
        self.species_history.append({
            'generation': self.generation,
            'num_species': len(self.species),
            'species_sizes': {s.species_id: len(s.members) for s in self.species.values()}
        })
        
        # Innovation statistics
        self.innovation_history.append({
            'generation': self.generation,
            'total_innovations': self.innovation_number,
            'new_innovations': len([g for g in self.population if g.mutations])
        })
        
    def _update_collective(self) -> None:
        """Update collective intelligence with evolved networks."""
        if not self.collective:
            return
            
        # Clear old networks
        self.collective.networks.clear()
        
        # Add top performers from each species
        for species in self.species.values():
            if species.members:
                # Get best member
                best = max(species.members, key=lambda g: g.fitness)
                network = best.to_network()
                
                # Create role based on species characteristics
                role = NetworkRole(
                    name=f"{species.species_id}_elite",
                    frequency_band=(0.1, 100.0),
                    influence_weight=best.fitness,
                    receptivity=1.0,
                    specialization=species.species_id
                )
                
                # Random position in 3D space
                position = np.random.randn(3) * 10
                
                self.collective.add_network(
                    best.network_id,
                    network,
                    role,
                    position
                )
                
        # Connect networks based on species relationships
        for i, species1 in enumerate(self.species.values()):
            for j, species2 in enumerate(list(self.species.values())[i+1:], i+1):
                if species1.members and species2.members:
                    # Calculate species distance
                    distance = self._calculate_distance(
                        species1.representative,
                        species2.representative
                    )
                    
                    # Connect if similar enough
                    if distance < self.speciation_threshold * 2:
                        weight = 1.0 / (1.0 + distance)
                        best1 = max(species1.members, key=lambda g: g.fitness)
                        best2 = max(species2.members, key=lambda g: g.fitness)
                        
                        self.collective.connect_networks(
                            best1.network_id,
                            best2.network_id,
                            weight
                        )
                        
    def get_best_genome(self) -> Optional[Genome]:
        """Get the best genome in the current population."""
        if self.population:
            return max(self.population, key=lambda g: g.fitness)
        return None
        
    def get_species_info(self) -> Dict[str, Any]:
        """Get information about current species."""
        info = {}
        
        for species_id, species in self.species.items():
            if species.members:
                fitnesses = [g.fitness for g in species.members]
                info[species_id] = {
                    'size': len(species.members),
                    'mean_fitness': np.mean(fitnesses),
                    'max_fitness': np.max(fitnesses),
                    'stagnation': species.stagnation_counter,
                    'age': self.generation - species.created_generation,
                    'representative_nodes': len(species.representative.node_genes),
                    'representative_connections': len(species.representative.connection_genes)
                }
                
        return info
        
    def save_population(self, filename: str) -> None:
        """Save the current population to file."""
        import pickle
        
        data = {
            'population': self.population,
            'species': self.species,
            'generation': self.generation,
            'innovation_number': self.innovation_number,
            'innovations': self.innovations,
            'fitness_history': self.fitness_history,
            'species_history': self.species_history
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
            
    def load_population(self, filename: str) -> None:
        """Load a population from file."""
        import pickle
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        self.population = data['population']
        self.species = data['species']
        self.generation = data['generation']
        self.innovation_number = data['innovation_number']
        self.innovations = data['innovations']
        self.fitness_history = data['fitness_history']
        self.species_history = data['species_history'] 