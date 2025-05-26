"""
Core ResonantNetwork implementation for the SynthNN framework.

This module provides the network layer that manages collections of ResonantNodes,
their connections, and emergent behaviors through phase coupling and resonance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
import networkx as nx

from .resonant_node import ResonantNode


class Connection:
    """Represents a weighted, optionally delayed connection between nodes."""
    
    def __init__(self, weight: float = 1.0, delay: float = 0.0):
        self.weight = weight
        self.delay = delay
        self.signal_buffer = []  # For delayed signals
    
    def propagate(self, signal: float, dt: float) -> float:
        """Propagate signal through connection with optional delay."""
        if self.delay <= 0:
            return self.weight * signal
        
        # Add to buffer for delayed propagation
        self.signal_buffer.append((signal, self.delay))
        
        # Process buffer and return delayed signals
        output = 0.0
        remaining = []
        for sig, remaining_delay in self.signal_buffer:
            new_delay = remaining_delay - dt
            if new_delay <= 0:
                output += self.weight * sig
            else:
                remaining.append((sig, new_delay))
        
        self.signal_buffer = remaining
        return output


class ResonantNetwork:
    """
    A network of interconnected ResonantNodes with emergent dynamics.
    
    This class manages the collective behavior of resonant nodes, including
    their connections, phase coupling, and synchronization dynamics.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.nodes: Dict[str, ResonantNode] = {}
        self.connections: Dict[Tuple[str, str], Connection] = {}
        self.time = 0.0
        self.history = defaultdict(list)  # Track network dynamics
        
        # Network-wide parameters
        self.global_damping = 0.01
        self.coupling_strength = 0.1
        self.adaptation_rate = 0.05
    
    def add_node(self, node: ResonantNode) -> None:
        """Add a node to the network."""
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already exists in network")
        self.nodes[node.node_id] = node
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its connections."""
        if node_id not in self.nodes:
            return
        
        # Remove all connections involving this node
        connections_to_remove = [
            (src, tgt) for src, tgt in self.connections
            if src == node_id or tgt == node_id
        ]
        for conn in connections_to_remove:
            del self.connections[conn]
        
        # Remove the node
        del self.nodes[node_id]
    
    def connect(self, source_id: str, target_id: str, 
                weight: float = 1.0, delay: float = 0.0) -> None:
        """Create a connection between two nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Both nodes must exist in the network")
        
        self.connections[(source_id, target_id)] = Connection(weight, delay)
    
    def disconnect(self, source_id: str, target_id: str) -> None:
        """Remove a connection between two nodes."""
        key = (source_id, target_id)
        if key in self.connections:
            del self.connections[key]
    
    def get_neighbors(self, node_id: str) -> Set[str]:
        """Get all nodes connected from the given node."""
        return {tgt for src, tgt in self.connections if src == node_id}
    
    def get_inputs(self, node_id: str) -> Set[str]:
        """Get all nodes that connect to the given node."""
        return {src for src, tgt in self.connections if tgt == node_id}
    
    def compute_phase_coupling(self, node_id: str) -> float:
        """
        Compute the total phase coupling influence on a node from its inputs.
        
        Uses Kuramoto-style coupling with weighted connections.
        """
        target_node = self.nodes[node_id]
        coupling = 0.0
        
        for source_id in self.get_inputs(node_id):
            source_node = self.nodes[source_id]
            connection = self.connections[(source_id, node_id)]
            
            # Kuramoto coupling: weight * sin(source_phase - target_phase)
            phase_diff = source_node.phase - target_node.phase
            signal = np.sin(phase_diff)
            
            # Apply connection weight and delay
            coupling += connection.propagate(signal, self.time)
        
        return coupling * self.coupling_strength
    
    def step(self, dt: float, external_inputs: Optional[Dict[str, float]] = None) -> None:
        """
        Advance the network by one time step.
        
        Args:
            dt: Time step size
            external_inputs: Optional external signals for specific nodes
        """
        # Apply external inputs
        if external_inputs:
            for node_id, signal in external_inputs.items():
                if node_id in self.nodes:
                    self.nodes[node_id].apply_stimulus(signal)
        
        # Compute phase updates for all nodes
        phase_updates = {}
        for node_id in self.nodes:
            coupling = self.compute_phase_coupling(node_id)
            phase_updates[node_id] = coupling
        
        # Update all nodes
        for node_id, node in self.nodes.items():
            node.update_phase(dt, phase_updates[node_id])
            
            # Apply global damping
            node.amplitude *= (1 - self.global_damping * dt)
        
        # Record history
        self._record_state()
        
        # Increment time
        self.time += dt
    
    def _record_state(self) -> None:
        """Record current network state for analysis."""
        self.history['time'].append(self.time)
        
        # Record node states
        for node_id, node in self.nodes.items():
            self.history[f'phase_{node_id}'].append(node.phase)
            self.history[f'amplitude_{node_id}'].append(node.amplitude)
            self.history[f'signal_{node_id}'].append(node.oscillate(self.time))

        # Record global metrics
        self.history['total_energy'].append(self.measure_total_energy())
    
    def get_signals(self, node_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Get current signals from specified nodes (or all nodes)."""
        if node_ids is None:
            node_ids = list(self.nodes.keys())
        
        return {
            node_id: self.nodes[node_id].oscillate(self.time)
            for node_id in node_ids
            if node_id in self.nodes
        }
    
    def measure_synchronization(self, node_group: Optional[List[str]] = None) -> float:
        """
        Measure the synchronization level of a group of nodes.
        
        Returns a value between 0 (no sync) and 1 (perfect sync).
        """
        if node_group is None:
            node_group = list(self.nodes.keys())
        
        if len(node_group) < 2:
            return 1.0
        
        # Calculate average phase coherence
        phases = [self.nodes[node_id].phase for node_id in node_group]
        
        # Kuramoto order parameter
        complex_phases = np.exp(1j * np.array(phases))
        order_param = np.abs(np.mean(complex_phases))
        
        return order_param
    
    def adapt_connections(self, target_sync: float = 0.8) -> None:
        """
        Adapt connection weights to achieve target synchronization using STDP-like rules.
        """
        current_sync = self.measure_synchronization()
        sync_error = target_sync - current_sync
        
        for (src_id, tgt_id), connection in self.connections.items():
            src_node = self.nodes[src_id]
            tgt_node = self.nodes[tgt_id]
            
            # Phase difference determines weight update
            phase_diff = src_node.phase - tgt_node.phase
            
            # STDP-inspired rule
            if abs(phase_diff) < np.pi/2:  # Nodes are somewhat synchronized
                weight_change = self.adaptation_rate * sync_error * np.cos(phase_diff)
            else:  # Nodes are out of phase
                weight_change = -self.adaptation_rate * sync_error * np.cos(phase_diff)
            
            # Update weight with bounds
            connection.weight += weight_change
            connection.weight = np.clip(connection.weight, -2.0, 2.0)
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for analysis and visualization."""
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, 
                      frequency=node.frequency,
                      phase=node.phase,
                      amplitude=node.amplitude)
        
        # Add edges with weights
        for (src, tgt), conn in self.connections.items():
            G.add_edge(src, tgt, weight=conn.weight, delay=conn.delay)
        
        return G
    
    def save_state(self) -> Dict[str, Any]:
        """Save network state to dictionary."""
        return {
            'name': self.name,
            'time': self.time,
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'connections': {
                f"{src}->{tgt}": {'weight': conn.weight, 'delay': conn.delay}
                for (src, tgt), conn in self.connections.items()
            },
            'parameters': {
                'global_damping': self.global_damping,
                'coupling_strength': self.coupling_strength,
                'adaptation_rate': self.adaptation_rate
            }
        }
    
    @classmethod
    def load_state(cls, state: Dict[str, Any]) -> 'ResonantNetwork':
        """Load network from saved state."""
        network = cls(name=state['name'])
        network.time = state['time']
        
        # Load parameters
        params = state['parameters']
        network.global_damping = params['global_damping']
        network.coupling_strength = params['coupling_strength']
        network.adaptation_rate = params['adaptation_rate']
        
        # Load nodes
        for node_data in state['nodes'].values():
            node = ResonantNode.from_dict(node_data)
            network.add_node(node)
        
        # Load connections
        for conn_str, conn_data in state['connections'].items():
            src, tgt = conn_str.split('->')
            network.connect(src, tgt, conn_data['weight'], conn_data['delay'])

        return network

    def measure_total_energy(self, node_group: Optional[List[str]] = None) -> float:
        """Calculate the total energy of the network or a subset of nodes."""
        if node_group is None:
            node_group = list(self.nodes.keys())

        return sum(
            self.nodes[nid].energy() for nid in node_group if nid in self.nodes
        )

    def get_network_statistics(self) -> Dict[str, float]:
        """Return basic statistics about the current network state."""
        return {
            'num_nodes': len(self.nodes),
            'num_connections': len(self.connections),
            'synchronization': self.measure_synchronization(),
            'total_energy': self.measure_total_energy(),
        }
