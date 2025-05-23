#!/usr/bin/env python3
"""
Multi-Modal SynthNN Demonstration

This example showcases how Synthetic Resonant Neural Networks can be applied
beyond music generation to various AI tasks including:
- Text analysis and generation
- Image pattern recognition
- Cross-modal translation
- Emergent behavior studies
- Novel AI architectures
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time

from synthnn.core import (
    ResonantNode, 
    ResonantNetwork,
    SignalProcessor,
    UniversalPatternCodec,
    TextPatternEncoder,
    ImagePatternEncoder,
    AudioPatternEncoder
)
from synthnn.performance import AcceleratedResonantNetwork, BackendManager


class MultiModalResonantSystem:
    """
    A multi-modal AI system using resonant networks for diverse tasks.
    
    This demonstrates how resonance-based computation can be applied
    to various domains beyond music.
    """
    
    def __init__(self, use_acceleration: bool = True):
        self.use_acceleration = use_acceleration
        self.signal_processor = SignalProcessor()
        self.codec = UniversalPatternCodec()
        
        # Create specialized networks for different modalities
        self.networks = {
            'language': self._create_language_network(),
            'vision': self._create_vision_network(),
            'reasoning': self._create_reasoning_network(),
            'memory': self._create_memory_network()
        }
        
        # Store cross-modal connection definitions
        self.cross_modal_links: List[Dict[str, Any]] = [] 
        self._define_cross_modal_connections()
        
    def _create_language_network(self) -> ResonantNetwork:
        """Create a network specialized for language processing."""
        if self.use_acceleration:
            network = AcceleratedResonantNetwork(name="language_network")
        else:
            network = ResonantNetwork(name="language_network")
            
        # Create nodes representing different linguistic features
        # Frequency bands represent different aspects of language
        linguistic_features = {
            'phonemes': np.linspace(100, 500, 10),      # Low frequencies for sound units
            'morphemes': np.linspace(500, 1000, 10),    # Mid frequencies for meaning units
            'syntax': np.linspace(1000, 1500, 10),      # Higher for grammatical structures
            'semantics': np.linspace(1500, 2000, 10),   # Highest for meaning
        }
        
        for feature_type, frequencies in linguistic_features.items():
            for i, freq in enumerate(frequencies):
                node = ResonantNode(
                    node_id=f"{feature_type}_{i}",
                    frequency=freq,
                    phase=np.random.uniform(0, 2*np.pi),
                    amplitude=0.5
                )
                network.add_node(node)
                
        # Create hierarchical connections within language network
        self._create_hierarchical_connections(network, linguistic_features)
        
        return network
    
    def _create_vision_network(self) -> ResonantNetwork:
        """Create a network for visual pattern processing."""
        if self.use_acceleration:
            network = AcceleratedResonantNetwork(name="vision_network")
        else:
            network = ResonantNetwork(name="vision_network")
            
        # Visual features at different scales (similar to CNN layers)
        visual_features = {
            'edges': np.linspace(200, 400, 8),           # Edge detectors
            'textures': np.linspace(400, 800, 8),       # Texture patterns
            'shapes': np.linspace(800, 1200, 8),        # Shape recognition
            'objects': np.linspace(1200, 1600, 8),      # Object-level features
        }
        
        for feature_type, frequencies in visual_features.items():
            for i, freq in enumerate(frequencies):
                node = ResonantNode(
                    node_id=f"{feature_type}_{i}",
                    frequency=freq,
                    phase=np.random.uniform(0, 2*np.pi),
                    amplitude=0.5,
                    damping=0.05  # Visual features decay slower
                )
                network.add_node(node)
                
        # Create spatial connections (neighboring features connect)
        self._create_spatial_connections(network, visual_features)
        
        return network
    
    def _create_reasoning_network(self) -> ResonantNetwork:
        """Create a network for logical reasoning and pattern inference."""
        if self.use_acceleration:
            network = AcceleratedResonantNetwork(name="reasoning_network")
        else:
            network = ResonantNetwork(name="reasoning_network")
            
        # Reasoning operates at lower frequencies (slower, deliberate processing)
        reasoning_features = {
            'pattern_match': np.linspace(10, 50, 6),
            'inference': np.linspace(50, 100, 6),
            'abstraction': np.linspace(100, 150, 6),
            'synthesis': np.linspace(150, 200, 6),
        }
        
        for feature_type, frequencies in reasoning_features.items():
            for i, freq in enumerate(frequencies):
                node = ResonantNode(
                    node_id=f"{feature_type}_{i}",
                    frequency=freq,
                    phase=0,  # Start synchronized for logical operations
                    amplitude=1.0,
                    damping=0.01  # Very low damping for sustained reasoning
                )
                network.add_node(node)
                
        # Dense connections for reasoning (everything connects)
        self._create_dense_connections(network, weight_scale=0.3)
        
        return network
    
    def _create_memory_network(self) -> ResonantNetwork:
        """Create a network for memory storage and retrieval."""
        if self.use_acceleration:
            network = AcceleratedResonantNetwork(name="memory_network")
        else:
            network = ResonantNetwork(name="memory_network")
            
        # Memory uses resonant frequencies that can store patterns
        memory_banks = {
            'short_term': np.linspace(300, 500, 5),      # Fast access
            'working': np.linspace(150, 300, 5),         # Active manipulation
            'long_term': np.linspace(50, 150, 5),        # Stable storage
            'associative': np.linspace(200, 400, 5),     # Pattern associations
        }
        
        for memory_type, frequencies in memory_banks.items():
            for i, freq in enumerate(frequencies):
                node = ResonantNode(
                    node_id=f"{memory_type}_{i}",
                    frequency=freq,
                    phase=0,
                    amplitude=0.1,  # Start with low amplitude
                    damping=0.001 if 'long_term' in memory_type else 0.1
                )
                network.add_node(node)
                
        # Associative connections for memory
        self._create_associative_connections(network, memory_banks)
        
        return network
    
    def _create_hierarchical_connections(self, network: ResonantNetwork, 
                                       feature_groups: Dict[str, np.ndarray]):
        """Create hierarchical connections within a network."""
        # Connect within groups
        for group_name, frequencies in feature_groups.items():
            group_nodes = [f"{group_name}_{i}" for i in range(len(frequencies))]
            for i in range(len(group_nodes) - 1):
                network.connect(group_nodes[i], group_nodes[i+1], weight=0.5)
                network.connect(group_nodes[i+1], group_nodes[i], weight=0.3)
        
        # Connect between adjacent hierarchical levels
        group_names = list(feature_groups.keys())
        for i in range(len(group_names) - 1):
            lower_group = group_names[i]
            higher_group = group_names[i + 1]
            
            # Sample connections between levels
            for j in range(0, len(feature_groups[lower_group]), 2):
                for k in range(0, len(feature_groups[higher_group]), 2):
                    network.connect(f"{lower_group}_{j}", f"{higher_group}_{k}", weight=0.2)
    
    def _create_spatial_connections(self, network: ResonantNetwork,
                                  feature_groups: Dict[str, np.ndarray]):
        """Create spatially-aware connections (for vision network)."""
        for group_name, frequencies in feature_groups.items():
            group_nodes = [f"{group_name}_{i}" for i in range(len(frequencies))]
            
            # Grid-like connectivity (assuming square arrangement)
            grid_size = int(np.sqrt(len(group_nodes)))
            for i in range(len(group_nodes)):
                row = i // grid_size
                col = i % grid_size
                
                # Connect to neighbors
                neighbors = [
                    (row-1, col), (row+1, col),  # vertical
                    (row, col-1), (row, col+1),  # horizontal
                    (row-1, col-1), (row-1, col+1),  # diagonals
                    (row+1, col-1), (row+1, col+1)
                ]
                
                for nr, nc in neighbors:
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        neighbor_idx = nr * grid_size + nc
                        if neighbor_idx < len(group_nodes):
                            network.connect(group_nodes[i], group_nodes[neighbor_idx], 
                                          weight=0.3 / (abs(nr-row) + abs(nc-col)))
    
    def _create_dense_connections(self, network: ResonantNetwork, weight_scale: float = 0.1):
        """Create dense all-to-all connections."""
        node_ids = list(network.nodes.keys())
        for i, src in enumerate(node_ids):
            for j, tgt in enumerate(node_ids):
                if i != j:
                    weight = weight_scale / (1 + abs(i - j))
                    network.connect(src, tgt, weight)
    
    def _create_associative_connections(self, network: ResonantNetwork,
                                      memory_groups: Dict[str, np.ndarray]):
        """Create associative memory connections."""
        # Connect short-term to working memory
        for i in range(len(memory_groups['short_term'])):
            for j in range(len(memory_groups['working'])):
                network.connect(f"short_term_{i}", f"working_{j}", weight=0.4)
        
        # Connect working to long-term (consolidation)
        for i in range(len(memory_groups['working'])):
            for j in range(len(memory_groups['long_term'])):
                network.connect(f"working_{i}", f"long_term_{j}", weight=0.2)
        
        # Associative connections link all memory types
        for memory_type in memory_groups:
            for i in range(len(memory_groups['associative'])):
                for j in range(len(memory_groups[memory_type])):
                    if memory_type != 'associative':
                        network.connect(f"associative_{i}", f"{memory_type}_{j}", weight=0.3)
    
    def _define_cross_modal_connections(self):
        """Defines connections between different modal networks."""
        self.cross_modal_links = [] # Reset before defining

        # Language <-> Vision connections (for describing what we see)
        lang_vision_pairs = [
            ('semantics_5', 'objects_3'),
            ('syntax_3', 'shapes_2'),
            ('morphemes_7', 'textures_4')
        ]
        
        for lang_node_id, vis_node_id in lang_vision_pairs:
            if lang_node_id in self.networks['language'].nodes and \
               vis_node_id in self.networks['vision'].nodes:
                # Define bidirectional links
                self.cross_modal_links.append({
                    'source_network': 'language',
                    'source_node': lang_node_id,
                    'target_network': 'vision',
                    'target_node': vis_node_id,
                    'weight': 0.15
                })
                self.cross_modal_links.append({
                    'source_network': 'vision',
                    'source_node': vis_node_id,
                    'target_network': 'language',
                    'target_node': lang_node_id,
                    'weight': 0.15
                })
        
        # All modalities connect to reasoning
        for modality in ['language', 'vision', 'memory']:
            # Ensure nodes exist before attempting to link
            if modality not in self.networks or 'reasoning' not in self.networks:
                continue
            source_net_nodes = list(self.networks[modality].nodes.keys())
            reasoning_net_nodes = list(self.networks['reasoning'].nodes.keys())

            sample_nodes = source_net_nodes[::5]
            reasoning_nodes = reasoning_net_nodes[::3]
            
            for mod_node_id in sample_nodes[:3]:
                for reas_node_id in reasoning_nodes[:3]:
                    if mod_node_id in self.networks[modality].nodes and \
                       reas_node_id in self.networks['reasoning'].nodes:
                        self.cross_modal_links.append({
                            'source_network': modality,
                            'source_node': mod_node_id,
                            'target_network': 'reasoning',
                            'target_node': reas_node_id,
                            'weight': 0.1
                        })
        
        # Memory connects to all modalities (for storage/retrieval)
        for modality in ['language', 'vision', 'reasoning']:
            if modality not in self.networks or 'memory' not in self.networks:
                continue
            source_net_nodes = list(self.networks[modality].nodes.keys())
            memory_net_nodes = list(self.networks['memory'].nodes.keys())

            sample_nodes = source_net_nodes[::4]
            memory_nodes = memory_net_nodes[::2]
            
            for mod_node_id in sample_nodes[:3]:
                for mem_node_id in memory_nodes[:3]:
                    if mod_node_id in self.networks[modality].nodes and \
                       mem_node_id in self.networks['memory'].nodes:
                        self.cross_modal_links.append({
                            'source_network': 'memory',
                            'source_node': mem_node_id,
                            'target_network': modality,
                            'target_node': mod_node_id,
                            'weight': 0.2
                        })
    
    def _apply_cross_modal_influence(self):
        """Apply influence between networks based on defined links."""
        for link in self.cross_modal_links:
            source_network = self.networks.get(link['source_network'])
            target_network = self.networks.get(link['target_network'])
            
            if not source_network or not target_network:
                continue
                
            source_node_id = link['source_node']
            target_node_id = link['target_node']
            
            if source_node_id in source_network.nodes and \
               target_node_id in target_network.nodes:
                
                source_node = source_network.nodes[source_node_id]
                target_node = target_network.nodes[target_node_id]
                
                # Get signal from source node
                # For AcceleratedResonantNetwork, we might need to sync from device first
                # or get signal directly if backend supports it without full sync.
                # For simplicity, assume node.oscillate() works.
                # If using AcceleratedResonantNetwork, ensure _sync_from_device was called
                # if node state on CPU is needed, or get signal from device representation.
                source_signal = source_node.oscillate(source_network.time)
                
                # Apply weighted stimulus to target node
                target_node.apply_stimulus(source_signal * link['weight'])

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text through the language network."""
        print(f"\nProcessing text: '{text[:50]}...'")
        
        # Encode text to network parameters
        encoder = TextPatternEncoder()
        encoded_node_params = encoder.encode(text) # This returns Dict[node_id, params]
        
        # Apply encoded pattern to language network
        lang_network = self.networks['language']
        
        # Clear old states or re-initialize nodes if necessary
        # For simplicity, we'll assume the network has nodes corresponding to potential text length
        # or that we are dynamically adding/removing them (more complex).
        # Here, we just update existing nodes if their IDs match.

        active_node_ids_from_encoding = set()
        for node_id_in_encoding, params in encoded_node_params.items():
            # If the language network is pre-defined with specific node IDs, ensure they match
            # or map them. For this demo, we directly use encoded node IDs if they exist.
            if node_id_in_encoding in lang_network.nodes:
                node = lang_network.nodes[node_id_in_encoding]
                node.frequency = params['frequency']
                node.phase = params['phase']
                node.amplitude = params['amplitude']
                active_node_ids_from_encoding.add(node_id_in_encoding)
            else:
                # Optionally, create the node if it doesn't exist and matches naming convention
                if node_id_in_encoding.startswith(tuple(self.networks['language'].nodes.keys())[0].split('_')[0] if self.networks['language'].nodes else "") : # Check prefix like 'phonemes'
                    new_node = ResonantNode(
                        node_id=node_id_in_encoding, 
                        frequency=params['frequency'], 
                        phase=params['phase'], 
                        amplitude=params['amplitude']
                    )
                    lang_network.add_node(new_node)
                    active_node_ids_from_encoding.add(node_id_in_encoding)

        # Dampen nodes not activated by the current text
        for node_id, node in lang_network.nodes.items():
            if node_id not in active_node_ids_from_encoding:
                node.amplitude *= 0.1 # Reduce amplitude of unused nodes
        
        # Simulate network dynamics
        for _ in range(100):
            lang_network.step(0.01)
            self._apply_cross_modal_influence() # Apply influence at each step
        
        # Measure emergent patterns
        sync_level = lang_network.measure_synchronization()
        
        # Extract dominant patterns
        signals = lang_network.get_signals()
        dominant_freqs = self._extract_dominant_frequencies(signals)
        
        return {
            'synchronization': sync_level,
            'dominant_frequencies': dominant_freqs,
            'network_state': lang_network.save_state(),
            'interpretation': self._interpret_language_patterns(dominant_freqs)
        }
    
    def process_image(self, image_data: np.ndarray) -> Dict[str, Any]:
        """Process image through the vision network."""
        print(f"\nProcessing image of shape: {image_data.shape}")
        
        # Encode image to network parameters
        encoder = ImagePatternEncoder(grid_size=(8,8)) # Ensure grid matches vision network structure
        encoded_node_params = encoder.encode(image_data)
        
        # Apply to vision network
        vision_network = self.networks['vision']

        active_node_ids_from_encoding = set()
        for node_id_in_encoding, params in encoded_node_params.items():
            if node_id_in_encoding in vision_network.nodes:
                node = vision_network.nodes[node_id_in_encoding]
                node.frequency = params['frequency']
                node.phase = params['phase']
                node.amplitude = params['amplitude']
                active_node_ids_from_encoding.add(node_id_in_encoding)
            # else: # Vision network nodes are typically pre-defined by feature type
                # print(f"Warning: Encoded image node {node_id_in_encoding} not in vision network.")

        for node_id, node in vision_network.nodes.items():
            if node_id not in active_node_ids_from_encoding:
                node.amplitude *= 0.1
        
        # Process through network
        for _ in range(150):
            vision_network.step(0.01)
            self._apply_cross_modal_influence() # Apply influence at each step
        
        # Extract visual features
        edge_activity = self._measure_group_activity(vision_network, 'edges')
        shape_activity = self._measure_group_activity(vision_network, 'shapes')
        
        return {
            'edge_detection': edge_activity,
            'shape_recognition': shape_activity,
            'visual_coherence': vision_network.measure_synchronization(),
            'interpretation': self._interpret_visual_patterns(edge_activity, shape_activity)
        }
    
    def cross_modal_translation(self, source_data: Any, 
                              source_modality: str, 
                              target_modality: str) -> Any:
        """Translate between modalities using resonant coupling."""
        print(f"\nTranslating from {source_modality} to {target_modality}")
        
        # Process source data
        if source_modality == 'text':
            source_result = self.process_text(source_data)
        elif source_modality == 'image':
            source_result = self.process_image(source_data)
        else:
            raise ValueError(f"Unsupported source modality: {source_modality}")
        
        # Transfer patterns through cross-modal connections
        source_network = self.networks['language' if source_modality == 'text' else 'vision']
        target_network = self.networks['vision' if target_modality == 'image' else 'language']
        
        # Propagate activity
        for _ in range(200):
            # Step individual networks first
            source_network.step(0.01)
            if target_network != source_network: # Avoid double-stepping if same network
                target_network.step(0.01)
            
            # Then apply cross-modal influences
            self._apply_cross_modal_influence()
        
        # Decode from target network
        if target_modality == 'text':
            return self._decode_to_text(target_network)
        elif target_modality == 'image':
            return self._decode_to_image(target_network)
    
    def demonstrate_emergent_behavior(self):
        """Show how complex behaviors emerge from simple resonant interactions."""
        print("\n=== Demonstrating Emergent Behaviors ===")
        
        # Start with random initialization
        for network in self.networks.values():
            for node in network.nodes.values():
                node.phase = np.random.uniform(0, 2*np.pi)
                node.amplitude = np.random.uniform(0.1, 0.5)
        
        # Track network states over time
        history = {name: {'sync': [], 'energy': []} for name in self.networks}
        
        # Simulate extended interaction
        for step_idx in range(500):
            # Update all networks
            for name, network in self.networks.items():
                network.step(0.01)
            
            # Apply cross-modal influences after all networks have stepped
            self._apply_cross_modal_influence()
            
            # Record metrics
            for name, network in self.networks.items():
                sync = network.measure_synchronization()
                energy = sum(node.energy() for node in network.nodes.values())
                
                history[name]['sync'].append(sync)
                history[name]['energy'].append(energy)
            
            # Occasional perturbation (simulating input)
            if step_idx % 100 == 50:
                random_network = np.random.choice(list(self.networks.values()))
                random_node = np.random.choice(list(random_network.nodes.values()))
                random_node.apply_stimulus(np.random.randn() * 2)
        
        # Visualize emergent patterns
        self._plot_emergent_behavior(history)
        
        # Identify emergent properties
        emergent_properties = self._analyze_emergence(history)
        
        return emergent_properties
    
    def _extract_dominant_frequencies(self, signals: Dict[str, float]) -> List[float]:
        """Extract dominant frequencies from network signals."""
        # Combine all signals
        combined = np.array(list(signals.values()))
        
        # FFT to find dominant frequencies
        fft_result = np.fft.fft(combined)
        freqs = np.fft.fftfreq(len(combined))
        
        # Get top 5 frequencies
        magnitude = np.abs(fft_result)
        top_indices = np.argsort(magnitude)[-5:]
        
        return [freqs[i] for i in top_indices]
    
    def _measure_group_activity(self, network: ResonantNetwork, group: str) -> float:
        """Measure activity level of a node group."""
        group_nodes = [n for n in network.nodes if group in n]
        if not group_nodes:
            return 0.0
            
        activities = [network.nodes[n].amplitude * network.nodes[n].energy() 
                     for n in group_nodes]
        return np.mean(activities)
    
    def _interpret_language_patterns(self, frequencies: List[float]) -> str:
        """Interpret language network patterns."""
        freq_ranges = {
            (0, 500): "phonetic processing dominant",
            (500, 1000): "morphological analysis active", 
            (1000, 1500): "syntactic structure detected",
            (1500, 2000): "semantic comprehension engaged"
        }
        
        interpretations = []
        for freq in frequencies:
            for (low, high), interp in freq_ranges.items():
                if low <= abs(freq) < high:
                    interpretations.append(interp)
                    break
        
        return "; ".join(set(interpretations))
    
    def _interpret_visual_patterns(self, edge_activity: float, shape_activity: float) -> str:
        """Interpret visual network patterns."""
        interpretations = []
        
        if edge_activity > 0.7:
            interpretations.append("strong edge features detected")
        if shape_activity > 0.6:
            interpretations.append("clear shape patterns recognized")
        if edge_activity < 0.3 and shape_activity < 0.3:
            interpretations.append("uniform or texture-dominant image")
            
        return "; ".join(interpretations)
    
    def _decode_to_text(self, network: ResonantNetwork) -> str:
        """Decode network state back to text."""
        # Extract network state
        nodes = list(network.nodes.values())
        
        pattern = {
            'frequencies': [n.frequency for n in nodes],
            'phases': [n.phase for n in nodes],
            'amplitudes': [n.amplitude for n in nodes]
        }
        
        # Use decoder
        decoder = self.codec.decoders.get('text', self.codec.decoders['audio'])
        return decoder.decode(pattern)
    
    def _decode_to_image(self, network: ResonantNetwork) -> np.ndarray:
        """Decode network state back to image."""
        # Simple example: create grayscale image from network state
        nodes = list(network.nodes.values())
        size = int(np.sqrt(len(nodes)))
        
        # Map node amplitudes to pixel values
        pixel_values = [n.amplitude * n.oscillate(network.time) for n in nodes]
        
        # Reshape to 2D
        image = np.array(pixel_values[:size*size]).reshape(size, size)
        
        # Normalize to 0-255
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        return image
    
    def _plot_emergent_behavior(self, history: Dict[str, Dict[str, List[float]]]):
        """Visualize emergent behaviors."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Emergent Behaviors in Multi-Modal Resonant System")
        
        for idx, (name, data) in enumerate(history.items()):
            ax = axes[idx // 2, idx % 2]
            
            # Plot synchronization and energy
            ax2 = ax.twinx()
            
            line1 = ax.plot(data['sync'], label='Synchronization', color='blue', alpha=0.7)
            line2 = ax2.plot(data['energy'], label='Total Energy', color='red', alpha=0.7)
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Synchronization', color='blue')
            ax2.set_ylabel('Energy', color='red')
            ax.set_title(f'{name.capitalize()} Network')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right')
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('emergent_behaviors.png', dpi=150)
        plt.show()
    
    def _analyze_emergence(self, history: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Analyze emergent properties from simulation history."""
        properties = {}
        
        for name, data in history.items():
            sync_data = np.array(data['sync'])
            energy_data = np.array(data['energy'])
            
            properties[name] = {
                'mean_synchronization': np.mean(sync_data),
                'sync_stability': 1.0 / (1.0 + np.std(sync_data)),
                'energy_efficiency': np.mean(sync_data) / (np.mean(energy_data) + 1e-6),
                'phase_transitions': self._detect_phase_transitions(sync_data),
                'attractor_states': self._find_attractors(sync_data, energy_data)
            }
        
        # Cross-network correlations
        network_names = list(history.keys())
        correlations = {}
        
        for i in range(len(network_names)):
            for j in range(i+1, len(network_names)):
                name1, name2 = network_names[i], network_names[j]
                sync_corr = np.corrcoef(history[name1]['sync'], history[name2]['sync'])[0, 1]
                correlations[f"{name1}-{name2}"] = sync_corr
        
        properties['cross_modal_coupling'] = correlations
        
        return properties
    
    def _detect_phase_transitions(self, data: np.ndarray) -> List[int]:
        """Detect phase transitions in synchronization data."""
        # Simple detection: large changes in derivative
        diff = np.diff(data)
        threshold = 2 * np.std(diff)
        transitions = np.where(np.abs(diff) > threshold)[0]
        return transitions.tolist()
    
    def _find_attractors(self, sync_data: np.ndarray, energy_data: np.ndarray) -> int:
        """Find number of attractor states."""
        # Combine sync and energy into state space
        states = np.column_stack([sync_data[:-1], energy_data[:-1]])
        next_states = np.column_stack([sync_data[1:], energy_data[1:]])
        
        # Find recurring states (simplified)
        unique_states = []
        tolerance = 0.1
        
        for state in states:
            is_new = True
            for unique in unique_states:
                if np.linalg.norm(state - unique) < tolerance:
                    is_new = False
                    break
            if is_new:
                unique_states.append(state)
        
        return len(unique_states)


def demonstrate_ai_architecture():
    """Demonstrate SynthNN as a novel AI architecture."""
    print("="*60)
    print("SynthNN: A Novel AI Architecture Based on Resonance")
    print("="*60)
    
    # Create multi-modal system
    print("\n1. Creating Multi-Modal Resonant System...")
    system = MultiModalResonantSystem(use_acceleration=True)
    
    # Test different modalities
    print("\n2. Testing Individual Modalities:")
    
    # Language processing
    text_result = system.process_text(
        "The concept of resonance in neural networks opens new possibilities for AI."
    )
    print(f"   Language processing synchronization: {text_result['synchronization']:.3f}")
    print(f"   Interpretation: {text_result['interpretation']}")
    
    # Vision processing
    test_image = np.random.rand(32, 32) * 255  # Random test image
    image_result = system.process_image(test_image)
    print(f"   Vision processing coherence: {image_result['visual_coherence']:.3f}")
    print(f"   Interpretation: {image_result['interpretation']}")
    
    # Cross-modal translation
    print("\n3. Cross-Modal Translation:")
    try:
        translated = system.cross_modal_translation(
            "A beautiful sunset over the ocean",
            source_modality='text',
            target_modality='image'
        )
        print(f"   Generated image shape: {translated.shape}")
    except Exception as e:
        print(f"   Translation demonstration: {e}")
    
    # Emergent behaviors
    print("\n4. Studying Emergent Behaviors:")
    emergent = system.demonstrate_emergent_behavior()
    
    print("\n5. Emergent Properties Analysis:")
    for network_name, props in emergent.items():
        if isinstance(props, dict) and 'mean_synchronization' in props:
            print(f"\n   {network_name} network:")
            print(f"     - Mean synchronization: {props['mean_synchronization']:.3f}")
            print(f"     - Stability: {props['sync_stability']:.3f}")
            print(f"     - Energy efficiency: {props['energy_efficiency']:.3f}")
            print(f"     - Phase transitions detected: {len(props['phase_transitions'])}")
            print(f"     - Attractor states: {props['attractor_states']}")
    
    # Show cross-modal coupling
    if 'cross_modal_coupling' in emergent:
        print("\n   Cross-modal coupling strengths:")
        for pair, correlation in emergent['cross_modal_coupling'].items():
            print(f"     - {pair}: {correlation:.3f}")
    
    print("\n" + "="*60)
    print("Key Advantages of Resonance-Based AI:")
    print("="*60)
    print("""
    1. BIOLOGICAL PLAUSIBILITY: Mimics neural oscillations found in the brain
    2. ENERGY EFFICIENCY: Information encoded in phase relationships
    3. EMERGENT COMPUTATION: Complex behaviors from simple interactions
    4. CONTINUOUS LEARNING: No discrete training/inference phases
    5. MULTI-MODAL INTEGRATION: Natural cross-modal connections
    6. INTERPRETABILITY: Frequency analysis reveals processing
    7. SCALABILITY: Hierarchical organization of resonant modules
    8. ROBUSTNESS: Distributed representation across oscillators
    """)


def demonstrate_practical_applications():
    """Show practical applications beyond music."""
    print("\n" + "="*60)
    print("Practical Applications of SynthNN")
    print("="*60)
    
    applications = {
        "Signal Processing": {
            "description": "Real-time analysis of complex signals",
            "use_cases": ["EEG/Brain signal analysis", "Seismic data", "Network traffic patterns"],
            "advantage": "Natural frequency decomposition without FFT windows"
        },
        "Pattern Recognition": {
            "description": "Detecting patterns through resonance matching",
            "use_cases": ["Anomaly detection", "Voice recognition", "Gesture recognition"],
            "advantage": "Continuous adaptation to new patterns"
        },
        "Control Systems": {
            "description": "Adaptive control through phase synchronization", 
            "use_cases": ["Robotics", "Drone swarms", "Smart grid management"],
            "advantage": "Emergent coordination without central control"
        },
        "Data Compression": {
            "description": "Encoding data as resonant patterns",
            "use_cases": ["Audio/Video compression", "Time series compression", "Sensor data"],
            "advantage": "Lossy compression that preserves perceptual features"
        },
        "Generative AI": {
            "description": "Creating new content through resonant synthesis",
            "use_cases": ["Music/Audio", "Textures", "Abstract patterns", "Code generation"],
            "advantage": "Smooth interpolation in frequency space"
        },
        "Quantum Computing Interface": {
            "description": "Bridge between classical and quantum systems",
            "use_cases": ["Quantum state preparation", "Error correction", "Hybrid algorithms"],
            "advantage": "Natural mapping to quantum oscillations"
        }
    }
    
    for app_name, details in applications.items():
        print(f"\n{app_name}:")
        print(f"  Description: {details['description']}")
        print(f"  Use cases: {', '.join(details['use_cases'])}")
        print(f"  Key advantage: {details['advantage']}")


def main():
    """Run comprehensive demonstration."""
    
    # Show this is more than music generation
    print("\n🧠 SynthNN: Beyond Music Generation")
    print("A Revolutionary AI Architecture Based on Resonance and Wave Physics\n")
    
    # Demonstrate AI architecture capabilities
    demonstrate_ai_architecture()
    
    # Show practical applications
    demonstrate_practical_applications()
    
    # Future research directions
    print("\n" + "="*60)
    print("Future Research Directions")
    print("="*60)
    print("""
    1. NEUROMORPHIC HARDWARE: Implement on oscillator-based chips
    2. QUANTUM RESONANCE: Explore quantum mechanical extensions
    3. SWARM INTELLIGENCE: Distributed AI with resonant consensus
    4. CONSCIOUSNESS STUDIES: Model awareness through global synchronization
    5. HYBRID ARCHITECTURES: Combine with transformers/CNNs
    6. EDGE COMPUTING: Low-power AI for IoT devices
    7. EXPLAINABLE AI: Frequency analysis for interpretability
    8. CONTINUAL LEARNING: Life-long learning through resonance
    """)
    
    print("\n✨ SynthNN represents a paradigm shift in how we think about")
    print("   artificial intelligence - from discrete computation to continuous")
    print("   resonance, from isolated processing to emergent intelligence.")


if __name__ == "__main__":
    main() 