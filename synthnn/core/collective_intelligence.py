"""
Collective Intelligence Framework for SynthNN

Enables multiple resonant networks to work together as a collective,
exhibiting swarm intelligence, distributed decision-making, and emergent behaviors.
Networks can exist in physical space (using ResonanceField4D) or abstract spaces.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict
import threading
import queue

from .resonant_network import ResonantNetwork
from .resonant_node import ResonantNode
from .resonance_field import ResonanceField4D, SpatialResonantNode
from .signal_processor import SignalProcessor


class CommunicationMode(Enum):
    """How networks communicate with each other."""
    DIRECT = "direct"              # Direct signal passing
    FIELD_MEDIATED = "field"       # Through shared resonance field
    PHASE_COUPLING = "phase"       # Phase synchronization
    FREQUENCY_MODULATION = "freq"  # FM encoding


class ConsensusMethod(Enum):
    """Methods for reaching collective decisions."""
    SYNCHRONIZATION = "sync"       # Decision when synchronized
    VOTING = "vote"               # Weighted voting by amplitude
    EMERGENCE = "emerge"          # Emergent from dynamics
    HIERARCHICAL = "hierarchy"    # Top-down from leader networks


@dataclass
class NetworkRole:
    """Defines a network's role in the collective."""
    name: str
    frequency_band: Tuple[float, float]  # Preferred frequency range
    influence_weight: float              # How much influence on others
    receptivity: float                   # How much influenced by others
    specialization: Optional[str] = None # Task specialization


@dataclass 
class CollectiveMemory:
    """Distributed memory across the collective."""
    pattern_id: str
    storage_networks: Set[str]     # Which networks store this
    encoding_phase: np.ndarray     # Phase relationships encoding the pattern
    encoding_frequency: np.ndarray # Frequency relationships
    strength: float                # Memory strength
    last_accessed: float          # Time last accessed
    metadata: Dict[str, Any] = field(default_factory=dict)


class CollectiveIntelligence:
    """
    A collective of resonant networks that can work together,
    communicate, and exhibit emergent intelligent behaviors.
    """
    
    def __init__(self, name: str = "collective",
                 communication_mode: CommunicationMode = CommunicationMode.PHASE_COUPLING,
                 consensus_method: ConsensusMethod = ConsensusMethod.SYNCHRONIZATION,
                 use_spatial_field: bool = False,
                 field_dimensions: Optional[Tuple[int, int, int]] = None):
        """
        Initialize a collective intelligence system.
        
        Args:
            name: Name of the collective
            communication_mode: How networks communicate
            consensus_method: How decisions are made
            use_spatial_field: Whether to use 4D resonance field
            field_dimensions: Size of spatial field if used
        """
        self.name = name
        self.communication_mode = communication_mode
        self.consensus_method = consensus_method
        
        # Networks in the collective
        self.networks: Dict[str, ResonantNetwork] = {}
        self.network_roles: Dict[str, NetworkRole] = {}
        
        # Communication infrastructure
        self.communication_graph = nx.DiGraph()
        self.message_queue = queue.Queue()
        
        # Spatial field if used
        self.use_spatial_field = use_spatial_field
        if use_spatial_field:
            dims = field_dimensions or (50, 50, 50)
            self.spatial_field = ResonanceField4D(
                dimensions=dims,
                resolution=1.0,
                wave_speed=100.0  # Faster for neural signals
            )
        else:
            self.spatial_field = None
            
        # Collective state
        self.collective_phase = 0.0
        self.collective_frequency = 1.0
        self.synchronization_index = 0.0
        
        # Distributed memory
        self.memories: Dict[str, CollectiveMemory] = {}
        self.memory_index = defaultdict(set)  # pattern_type -> set of pattern_ids
        
        # Emergent properties tracking
        self.emergence_history = []
        self.decision_history = []
        
        # Time and dynamics
        self.time = 0.0
        self.signal_processor = SignalProcessor()
        
    def add_network(self, network_id: str, network: ResonantNetwork,
                   role: Optional[NetworkRole] = None,
                   position: Optional[np.ndarray] = None) -> None:
        """
        Add a network to the collective.
        
        Args:
            network_id: Unique identifier for the network
            network: The ResonantNetwork instance
            role: Optional role definition
            position: Optional 3D position if using spatial field
        """
        self.networks[network_id] = network
        
        # Assign role
        if role:
            self.network_roles[network_id] = role
        else:
            # Default role
            self.network_roles[network_id] = NetworkRole(
                name="generic",
                frequency_band=(0.1, 100.0),
                influence_weight=1.0,
                receptivity=1.0
            )
            
        # Add to communication graph
        self.communication_graph.add_node(network_id)
        
        # If using spatial field, create spatial nodes for network
        if self.use_spatial_field and position is not None:
            self._add_network_to_field(network_id, network, position)
            
    def _add_network_to_field(self, network_id: str, network: ResonantNetwork,
                             position: np.ndarray) -> None:
        """Add network's nodes to the spatial field."""
        # Create a spatial node for each resonant node
        for node_id, node in network.nodes.items():
            spatial_node = SpatialResonantNode(
                node_id=f"{network_id}_{node_id}",
                position=position + np.random.randn(3) * 0.1,  # Small offset
                frequency=node.frequency,
                phase=node.phase,
                amplitude=node.amplitude
            )
            self.spatial_field.add_spatial_node(spatial_node)
            
    def connect_networks(self, source_id: str, target_id: str,
                        weight: float = 1.0,
                        bidirectional: bool = True) -> None:
        """
        Create communication channel between networks.
        
        Args:
            source_id: Source network ID
            target_id: Target network ID  
            weight: Connection strength
            bidirectional: Whether connection goes both ways
        """
        self.communication_graph.add_edge(source_id, target_id, weight=weight)
        if bidirectional:
            self.communication_graph.add_edge(target_id, source_id, weight=weight)
            
    def create_hierarchy(self, hierarchy: Dict[str, List[str]]) -> None:
        """
        Create hierarchical organization.
        
        Args:
            hierarchy: Dict mapping parent IDs to lists of child IDs
        """
        for parent, children in hierarchy.items():
            for child in children:
                # Parent influences child strongly
                self.connect_networks(parent, child, weight=2.0, bidirectional=False)
                # Child influences parent weakly
                self.connect_networks(child, parent, weight=0.5, bidirectional=False)
                
    def step(self, dt: float = 0.01) -> None:
        """
        Advance the collective by one time step.
        """
        # Step 1: Update individual networks
        for network_id, network in self.networks.items():
            # Get influences from other networks
            influences = self._calculate_network_influences(network_id)
            
            # Apply influences based on communication mode
            if self.communication_mode == CommunicationMode.PHASE_COUPLING:
                self._apply_phase_coupling(network_id, influences)
            elif self.communication_mode == CommunicationMode.FREQUENCY_MODULATION:
                self._apply_frequency_modulation(network_id, influences)
            elif self.communication_mode == CommunicationMode.DIRECT:
                self._apply_direct_signals(network_id, influences)
                
            # Step the network
            network.step(dt)
            
        # Step 2: Update spatial field if used
        if self.use_spatial_field:
            self.spatial_field.step(dt)
            self._sync_field_to_networks()
            
        # Step 3: Process collective dynamics
        self._update_collective_state()
        self._process_message_queue()
        
        # Step 4: Check for emergent behaviors
        self._detect_emergence()
        
        # Step 5: Memory consolidation
        self._consolidate_memories()
        
        self.time += dt
        
    def _calculate_network_influences(self, network_id: str) -> Dict[str, float]:
        """Calculate influences from connected networks."""
        influences = {}
        
        # Get predecessors in communication graph
        for source_id in self.communication_graph.predecessors(network_id):
            edge_data = self.communication_graph[source_id][network_id]
            weight = edge_data['weight']
            
            # Get source network state
            source_network = self.networks[source_id]
            source_role = self.network_roles[source_id]
            target_role = self.network_roles[network_id]
            
            # Calculate influence based on synchronization
            sync = self._measure_network_sync(source_network, self.networks[network_id])
            
            # Modulate by roles
            influence = weight * sync * source_role.influence_weight * target_role.receptivity
            
            influences[source_id] = influence
            
        return influences
        
    def _apply_phase_coupling(self, network_id: str, influences: Dict[str, float]) -> None:
        """Apply phase coupling from other networks."""
        network = self.networks[network_id]
        
        for source_id, influence in influences.items():
            source_network = self.networks[source_id]
            
            # Calculate phase difference
            phase_diff = self._calculate_phase_difference(source_network, network)
            
            # Apply Kuramoto-like coupling to nodes
            for node in network.nodes.values():
                coupling = influence * np.sin(phase_diff)
                node.phase += coupling * 0.01  # Small update
                
    def _apply_frequency_modulation(self, network_id: str, influences: Dict[str, float]) -> None:
        """Apply frequency modulation from other networks."""
        network = self.networks[network_id]
        
        for source_id, influence in influences.items():
            source_network = self.networks[source_id]
            
            # Get dominant frequency from source
            source_freq = self._get_dominant_frequency(source_network)
            
            # Modulate target network frequencies
            for node in network.nodes.values():
                freq_shift = influence * (source_freq - node.frequency) * 0.1
                node.frequency += freq_shift
                
    def _apply_direct_signals(self, network_id: str, influences: Dict[str, float]) -> None:
        """Apply direct signal injection from other networks."""
        network = self.networks[network_id]
        
        for source_id, influence in influences.items():
            source_network = self.networks[source_id]
            
            # Get output signals from source
            source_signals = source_network.get_signals()
            
            # Inject into target network nodes
            for node_id, signal in source_signals.items():
                if node_id in network.nodes:
                    network.nodes[node_id].amplitude += signal * influence * 0.1
                    
    def _measure_network_sync(self, network1: ResonantNetwork, 
                            network2: ResonantNetwork) -> float:
        """Measure synchronization between two networks."""
        # Get phase distributions
        phases1 = [node.phase for node in network1.nodes.values()]
        phases2 = [node.phase for node in network2.nodes.values()]
        
        # Calculate Kuramoto order parameter
        r1 = np.abs(np.mean([np.exp(1j * phase) for phase in phases1]))
        r2 = np.abs(np.mean([np.exp(1j * phase) for phase in phases2]))
        
        # Inter-network synchronization
        phase_diff = np.mean(phases1) - np.mean(phases2)
        inter_sync = np.cos(phase_diff)
        
        return r1 * r2 * inter_sync
        
    def _calculate_phase_difference(self, network1: ResonantNetwork,
                                  network2: ResonantNetwork) -> float:
        """Calculate mean phase difference between networks."""
        mean_phase1 = np.mean([node.phase for node in network1.nodes.values()])
        mean_phase2 = np.mean([node.phase for node in network2.nodes.values()])
        return mean_phase1 - mean_phase2
        
    def _get_dominant_frequency(self, network: ResonantNetwork) -> float:
        """Get dominant frequency of a network."""
        # Weighted average by amplitude
        total_weight = 0
        weighted_freq = 0
        
        for node in network.nodes.values():
            weighted_freq += node.frequency * node.amplitude
            total_weight += node.amplitude
            
        if total_weight > 0:
            return weighted_freq / total_weight
        return 1.0
        
    def _sync_field_to_networks(self) -> None:
        """Synchronize spatial field state back to networks."""
        if not self.use_spatial_field:
            return
            
        # Update network nodes based on field state
        for network_id, network in self.networks.items():
            for node_id, node in network.nodes.items():
                field_node_id = f"{network_id}_{node_id}"
                
                if field_node_id in self.spatial_field.nodes:
                    spatial_node = self.spatial_field.nodes[field_node_id]
                    
                    # Get field value at node position
                    grid_pos = self.spatial_field._position_to_grid(spatial_node.position)
                    if self.spatial_field._in_bounds(grid_pos):
                        field_amplitude = self.spatial_field.amplitude_field[grid_pos]
                        field_phase = self.spatial_field.phase_field[grid_pos]
                        
                        # Blend with node state
                        node.amplitude = 0.9 * node.amplitude + 0.1 * field_amplitude
                        node.phase = node.phase + 0.1 * (field_phase - node.phase)
                        
    def _update_collective_state(self) -> None:
        """Update collective-level state variables."""
        # Calculate collective synchronization
        all_phases = []
        all_frequencies = []
        all_amplitudes = []
        
        for network in self.networks.values():
            for node in network.nodes.values():
                all_phases.append(node.phase)
                all_frequencies.append(node.frequency)
                all_amplitudes.append(node.amplitude)
                
        # Kuramoto order parameter for collective
        if all_phases:
            complex_phases = [np.exp(1j * phase) for phase in all_phases]
            self.synchronization_index = np.abs(np.mean(complex_phases))
            self.collective_phase = np.angle(np.mean(complex_phases))
            
        # Weighted average frequency
        if all_frequencies and all_amplitudes:
            total_weight = sum(all_amplitudes)
            if total_weight > 0:
                self.collective_frequency = sum(
                    f * a for f, a in zip(all_frequencies, all_amplitudes)
                ) / total_weight
                
    def _process_message_queue(self) -> None:
        """Process inter-network messages."""
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get_nowait()
                self._handle_message(message)
            except queue.Empty:
                break
                
    def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle a message between networks."""
        # Message format: {'from': id, 'to': id, 'type': str, 'data': any}
        if message['type'] == 'memory_request':
            self._handle_memory_request(message)
        elif message['type'] == 'decision_proposal':
            self._handle_decision_proposal(message)
            
    def _detect_emergence(self) -> None:
        """Detect emergent behaviors in the collective."""
        # Check for spontaneous synchronization
        if self.synchronization_index > 0.8:
            self.emergence_history.append({
                'time': self.time,
                'type': 'synchronization',
                'value': self.synchronization_index
            })
            
        # Check for pattern formation
        if self.use_spatial_field:
            field_stats = self.spatial_field.get_field_statistics()
            if field_stats['entropy'] < 2.0:  # Low entropy = pattern
                self.emergence_history.append({
                    'time': self.time,
                    'type': 'spatial_pattern',
                    'stats': field_stats
                })
                
    def _consolidate_memories(self) -> None:
        """Consolidate distributed memories across the collective."""
        # Strengthen frequently accessed memories
        current_time = self.time
        for memory in self.memories.values():
            time_since_access = current_time - memory.last_accessed
            
            # Decay unused memories
            if time_since_access > 100.0:
                memory.strength *= 0.99
                
            # Remove very weak memories
            if memory.strength < 0.1:
                self._forget_memory(memory.pattern_id)
                
    def make_collective_decision(self, options: List[str],
                               context: Optional[Dict[str, Any]] = None) -> str:
        """
        Make a collective decision among options.
        
        Args:
            options: List of option strings
            context: Optional context information
            
        Returns:
            Selected option
        """
        if self.consensus_method == ConsensusMethod.SYNCHRONIZATION:
            return self._decide_by_synchronization(options, context)
        elif self.consensus_method == ConsensusMethod.VOTING:
            return self._decide_by_voting(options, context)
        elif self.consensus_method == ConsensusMethod.EMERGENCE:
            return self._decide_by_emergence(options, context)
        elif self.consensus_method == ConsensusMethod.HIERARCHICAL:
            return self._decide_hierarchically(options, context)
            
    def _decide_by_synchronization(self, options: List[str], 
                                  context: Optional[Dict[str, Any]]) -> str:
        """Make decision based on which option leads to most synchronization."""
        best_option = options[0]
        best_sync = 0.0
        
        for option in options:
            # Temporarily bias networks toward this option
            self._bias_networks_toward_option(option, context)
            
            # Run a few steps
            initial_sync = self.synchronization_index
            for _ in range(10):
                self.step(0.01)
                
            final_sync = self.synchronization_index
            sync_increase = final_sync - initial_sync
            
            if sync_increase > best_sync:
                best_sync = sync_increase
                best_option = option
                
            # Reset biases
            self._reset_network_biases()
            
        # Record decision
        self.decision_history.append({
            'time': self.time,
            'options': options,
            'selected': best_option,
            'method': 'synchronization',
            'confidence': best_sync
        })
        
        return best_option
        
    def _decide_by_voting(self, options: List[str],
                        context: Optional[Dict[str, Any]]) -> str:
        """Make decision by weighted voting from networks."""
        votes = defaultdict(float)
        
        for network_id, network in self.networks.items():
            role = self.network_roles[network_id]
            
            # Each network votes based on its state
            network_vote = self._get_network_vote(network, options, context)
            
            # Weight by influence and current amplitude
            vote_weight = role.influence_weight * np.mean([
                node.amplitude for node in network.nodes.values()
            ])
            
            votes[network_vote] += vote_weight
            
        # Select option with most votes
        best_option = max(votes.items(), key=lambda x: x[1])[0]
        
        self.decision_history.append({
            'time': self.time,
            'options': options,
            'selected': best_option,
            'method': 'voting',
            'votes': dict(votes)
        })
        
        return best_option
        
    def _bias_networks_toward_option(self, option: str,
                                   context: Optional[Dict[str, Any]]) -> None:
        """Temporarily bias networks toward an option."""
        # Simple implementation: modulate frequencies based on option hash
        option_freq = 1.0 + (hash(option) % 100) / 100.0
        
        for network in self.networks.values():
            for node in network.nodes.values():
                node.frequency *= option_freq
                
    def _reset_network_biases(self) -> None:
        """Reset any temporary biases."""
        # Reset to natural frequencies
        # This is simplified - in practice you'd store original frequencies
        for network in self.networks.values():
            for node in network.nodes.values():
                node.frequency = node.frequency / 1.5 + 0.5  # Rough reset
                
    def _get_network_vote(self, network: ResonantNetwork, options: List[str],
                        context: Optional[Dict[str, Any]]) -> str:
        """Get a network's vote based on its state."""
        # Simple voting based on phase alignment with option hashes
        best_alignment = -1
        best_option = options[0]
        
        mean_phase = np.mean([node.phase for node in network.nodes.values()])
        
        for option in options:
            option_phase = (hash(option) % 628) / 100.0  # 0 to 2π
            alignment = np.cos(mean_phase - option_phase)
            
            if alignment > best_alignment:
                best_alignment = alignment
                best_option = option
                
        return best_option
        
    def store_pattern(self, pattern_id: str, pattern_data: Any,
                     pattern_type: str = "generic") -> None:
        """
        Store a pattern in distributed memory.
        
        Args:
            pattern_id: Unique identifier for the pattern
            pattern_data: The pattern data to store
            pattern_type: Type of pattern for indexing
        """
        # Encode pattern as phase/frequency relationships
        encoded_phase, encoded_freq = self._encode_pattern(pattern_data)
        
        # Select networks to store the pattern
        storage_networks = self._select_storage_networks(pattern_type)
        
        # Create memory entry
        memory = CollectiveMemory(
            pattern_id=pattern_id,
            storage_networks=storage_networks,
            encoding_phase=encoded_phase,
            encoding_frequency=encoded_freq,
            strength=1.0,
            last_accessed=self.time,
            metadata={'type': pattern_type, 'size': len(encoded_phase)}
        )
        
        # Store in selected networks
        for network_id in storage_networks:
            self._store_in_network(network_id, memory)
            
        # Add to memory index
        self.memories[pattern_id] = memory
        self.memory_index[pattern_type].add(pattern_id)
        
    def recall_pattern(self, pattern_id: str) -> Optional[Any]:
        """
        Recall a pattern from distributed memory.
        
        Args:
            pattern_id: ID of pattern to recall
            
        Returns:
            Reconstructed pattern data or None if not found
        """
        if pattern_id not in self.memories:
            return None
            
        memory = self.memories[pattern_id]
        memory.last_accessed = self.time
        
        # Collect pattern pieces from storage networks
        pattern_pieces = []
        for network_id in memory.storage_networks:
            if network_id in self.networks:
                piece = self._retrieve_from_network(network_id, memory)
                if piece is not None:
                    pattern_pieces.append(piece)
                    
        # Reconstruct pattern
        if pattern_pieces:
            return self._reconstruct_pattern(pattern_pieces, memory)
            
        return None
        
    def _encode_pattern(self, pattern_data: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Encode pattern as phase/frequency relationships."""
        # Convert pattern to numeric representation
        if isinstance(pattern_data, np.ndarray):
            numeric_data = pattern_data.flatten()
        elif isinstance(pattern_data, (list, tuple)):
            numeric_data = np.array(pattern_data)
        else:
            # Hash-based encoding for other types
            numeric_data = np.array([hash(str(pattern_data))])
            
        # Normalize to 0-1 range
        if len(numeric_data) > 0:
            numeric_data = (numeric_data - np.min(numeric_data)) / (
                np.max(numeric_data) - np.min(numeric_data) + 1e-10
            )
            
        # Encode as phases (0 to 2π)
        encoded_phase = numeric_data * 2 * np.pi
        
        # Encode magnitudes as frequencies (0.1 to 10 Hz)
        encoded_freq = 0.1 + numeric_data * 9.9
        
        return encoded_phase, encoded_freq
        
    def _select_storage_networks(self, pattern_type: str) -> Set[str]:
        """Select which networks should store a pattern."""
        storage_networks = set()
        
        # Select networks based on specialization
        for network_id, role in self.network_roles.items():
            if role.specialization == pattern_type or role.specialization is None:
                storage_networks.add(network_id)
                
        # Ensure redundancy (at least 3 networks)
        if len(storage_networks) < 3:
            # Add random networks for redundancy
            available = set(self.networks.keys()) - storage_networks
            needed = min(3 - len(storage_networks), len(available))
            storage_networks.update(np.random.choice(list(available), needed, replace=False))
            
        return storage_networks
        
    def _store_in_network(self, network_id: str, memory: CollectiveMemory) -> None:
        """Store memory in a specific network."""
        network = self.networks[network_id]
        
        # Modulate network state to encode memory
        nodes = list(network.nodes.values())
        
        for i, node in enumerate(nodes):
            if i < len(memory.encoding_phase):
                # Blend memory into node state
                node.phase = 0.7 * node.phase + 0.3 * memory.encoding_phase[i]
                node.frequency = 0.9 * node.frequency + 0.1 * memory.encoding_frequency[i]
                
    def _retrieve_from_network(self, network_id: str,
                             memory: CollectiveMemory) -> Optional[np.ndarray]:
        """Retrieve memory component from a network."""
        network = self.networks[network_id]
        nodes = list(network.nodes.values())
        
        # Extract encoded pattern from node states
        retrieved_phases = []
        retrieved_freqs = []
        
        for i in range(min(len(nodes), len(memory.encoding_phase))):
            retrieved_phases.append(nodes[i].phase)
            retrieved_freqs.append(nodes[i].frequency)
            
        if retrieved_phases:
            return np.array(retrieved_phases)
            
        return None
        
    def _reconstruct_pattern(self, pattern_pieces: List[np.ndarray],
                           memory: CollectiveMemory) -> Any:
        """Reconstruct pattern from pieces."""
        # Average the pieces (simple reconstruction)
        reconstructed = np.mean(pattern_pieces, axis=0)
        
        # Convert back from phase encoding
        numeric_data = reconstructed / (2 * np.pi)
        
        return numeric_data
        
    def _forget_memory(self, pattern_id: str) -> None:
        """Remove a memory from the collective."""
        if pattern_id in self.memories:
            memory = self.memories[pattern_id]
            
            # Remove from index
            if memory.metadata.get('type') in self.memory_index:
                self.memory_index[memory.metadata['type']].discard(pattern_id)
                
            # Remove from storage
            del self.memories[pattern_id]
            
    def visualize_collective_state(self) -> Dict[str, Any]:
        """Get visualization data for the collective state."""
        viz_data = {
            'networks': {},
            'synchronization': self.synchronization_index,
            'collective_frequency': self.collective_frequency,
            'collective_phase': self.collective_phase,
            'communication_graph': nx.node_link_data(self.communication_graph),
            'memory_count': len(self.memories),
            'emergence_events': len(self.emergence_history)
        }
        
        # Add individual network states
        for network_id, network in self.networks.items():
            role = self.network_roles[network_id]
            viz_data['networks'][network_id] = {
                'role': role.name,
                'node_count': len(network.nodes),
                'mean_frequency': self._get_dominant_frequency(network),
                'synchronization': network.measure_synchronization()
            }
            
        # Add spatial field slice if available
        if self.use_spatial_field:
            viz_data['field_slice'] = self.spatial_field.extract_slice('z')
            viz_data['field_stats'] = self.spatial_field.get_field_statistics()
            
        return viz_data 