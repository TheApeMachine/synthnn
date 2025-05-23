import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Any
import sys # For sys.exit

# Attempt to import from synthnn.core and synthnn.performance
# These imports will be placeholders if the exact paths or class names need adjustment
# based on the existing synthnn structure.
try:
    from synthnn.performance import AcceleratedResonantNetwork
    from synthnn.core import ResonantNode #, UniversalPatternCodec # Codec might be used later
except ImportError as e:
    print("Error: Failed to import required synthnn components (AcceleratedResonantNetwork, ResonantNode).")
    print(f"Details: {e}")
    print("Please ensure that the synthnn library is correctly installed and accessible in your Python path.")
    print("The SynthNNAdapter relies on these true implementations and cannot use placeholders.")
    sys.exit(1)


class SynthNNAdapter(nn.Module):
    """
    SynthNNAdapter for integrating a Resonant Network with Large Language Models.

    This adapter takes a hidden state from an LLM, uses it to modulate a
    SynthNN resonant network, and then feeds the network's resulting state
    back to the LLM, typically added as a residual connection.

    Initially, the internal parameters of the resonant network (e.g., base
    frequencies, connection weights) are fixed. Only the projection layers
    that interface with the LLM are learnable.
    """
    def __init__(self,
                 llm_hidden_size: int,
                 num_resonant_nodes: int = 32,
                 synthnn_dt: float = 0.01,
                 synthnn_steps: int = 10,
                 node_base_frequency: float = 1.0,
                 connection_weight: float = 0.5):
        super().__init__()

        self.llm_hidden_size = llm_hidden_size
        self.num_resonant_nodes = num_resonant_nodes
        self.synthnn_dt = synthnn_dt
        self.synthnn_steps = synthnn_steps
        self.node_base_frequency = node_base_frequency
        self.connection_weight = connection_weight

        # Initialize the AcceleratedResonantNetwork
        # The parameters of this network are currently fixed.
        # Future work could involve making parts of this network learnable.
        self.resonant_network = self._create_resonant_network()
        self._node_ids_sorted = sorted(list(self.resonant_network.nodes.keys()))
        
        # Store initial state for resetting
        self._store_initial_resonant_network_state()


        # Projection layer: LLM hidden state -> SynthNN external inputs
        # Output size is num_resonant_nodes, as we'll generate one signal per node.
        self.llm_to_synthnn_projection = nn.Linear(llm_hidden_size, num_resonant_nodes)

        # Projection layer: SynthNN state (phases + amplitudes) -> LLM hidden state dimension
        # Input size is num_resonant_nodes * 2 (concatenation of phases and amplitudes)
        self.synthnn_to_llm_projection = nn.Linear(num_resonant_nodes * 2, llm_hidden_size)
        
        # Activation for LLM to SynthNN projection to keep signals in a reasonable range for tanh modulation
        self.input_activation = nn.Tanh()


    def _create_resonant_network(self) -> AcceleratedResonantNetwork:
        """
        Helper function to create and configure the internal resonant network.
        The parameters here are fixed for the initial adapter version.
        """
        network = AcceleratedResonantNetwork(name="llm_adapter_synthnn")

        # Add nodes
        for i in range(self.num_resonant_nodes):
            # Vary frequencies slightly for diversity, could be made more sophisticated
            freq = self.node_base_frequency * (1 + (i / self.num_resonant_nodes) * 0.5)
            node = ResonantNode(
                node_id=f"adapter_node_{i}",
                frequency=freq,
                amplitude=0.1, # Start with low amplitude
                phase=np.random.uniform(0, 2 * np.pi) # Random initial phase
            )
            network.add_node(node)

        # Define connections (e.g., simple chain or local connections)
        if self.num_resonant_nodes > 1:
            for i in range(self.num_resonant_nodes):
                # Connect to next node
                if i < self.num_resonant_nodes - 1:
                    network.connect(f"adapter_node_{i}", f"adapter_node_{i+1}", weight=self.connection_weight)
                # Connect to previous node
                if i > 0:
                    network.connect(f"adapter_node_{i}", f"adapter_node_{i-1}", weight=self.connection_weight)
        
        # For AcceleratedResonantNetwork, ensure internal device states are synced after setup.
        # The .add_node() and .connect() methods in the provided search results for
        # AcceleratedResonantNetwork seem to call _sync_to_device() internally.
        # If not, a manual call like `network._sync_to_device()` would be needed here.

        return network

    def _store_initial_resonant_network_state(self):
        """Stores the initial phases and amplitudes of the resonant network's nodes."""
        self.initial_phases_map = {
            nid: self.resonant_network.nodes[nid].phase 
            for nid in self._node_ids_sorted
        }
        self.initial_amplitudes_map = {
            nid: self.resonant_network.nodes[nid].amplitude
            for nid in self._node_ids_sorted
        }
        # Also store initial time if relevant, though resonant_network.time is reset anyway.
        self.initial_time = self.resonant_network.time


    def reset_resonant_network_state(self):
        """Resets the resonant network to its stored initial state."""
        if not hasattr(self, 'initial_phases_map') or not hasattr(self, 'initial_amplitudes_map'):
            # This case should ideally not be hit if _store_initial_resonant_network_state is called in __init__
            self._store_initial_resonant_network_state()

        for node_id in self._node_ids_sorted:
            if node_id in self.resonant_network.nodes: # Check if node still exists
                self.resonant_network.nodes[node_id].phase = self.initial_phases_map[node_id]
                self.resonant_network.nodes[node_id].amplitude = self.initial_amplitudes_map[node_id]
        
        self.resonant_network.time = self.initial_time # Reset network time

        # If using AcceleratedResonantNetwork, its internal device arrays might need to be
        # re-synchronized from the host ResonantNode objects after their states are reset.
        # The _sync_to_device method typically handles this.
        if hasattr(self.resonant_network, '_sync_to_device') and callable(getattr(self.resonant_network, '_sync_to_device')):
            self.resonant_network._sync_to_device()


    def _get_synthnn_state_vector(self, device: torch.device) -> torch.Tensor:
        """
        Extracts a state vector (phases and amplitudes) from the resonant network.
        Converts numpy arrays from network nodes to PyTorch tensors.
        """
        phases = np.array([self.resonant_network.nodes[nid].phase for nid in self._node_ids_sorted], dtype=np.float32)
        amplitudes = np.array([self.resonant_network.nodes[nid].amplitude for nid in self._node_ids_sorted], dtype=np.float32)

        # Convert to PyTorch tensors and move to the specified device
        phase_tensor = torch.from_numpy(phases).to(device)
        amplitude_tensor = torch.from_numpy(amplitudes).to(device)

        return torch.cat([phase_tensor, amplitude_tensor], dim=0)

    def forward(self, llm_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SynthNNAdapter.

        Args:
            llm_hidden_state: torch.Tensor of shape [batch_size, sequence_length, llm_hidden_size]
                              or [batch_size, llm_hidden_size].
                              For simplicity, this example assumes the input might be
                              already reduced along the sequence dimension for some use cases,
                              e.g. [batch_size, llm_hidden_size].
                              If sequence is present, we might average or take the last token.

        Returns:
            torch.Tensor: The adapted LLM hidden state, same shape as input.
        """
        original_shape = llm_hidden_state.shape
        if not (llm_hidden_state.dim() == 3 and original_shape[-1] == self.llm_hidden_size):
             raise ValueError(
                f"Expected llm_hidden_state shape [batch_size, sequence_length, llm_hidden_size], "
                f"but got {original_shape}. llm_hidden_size is {self.llm_hidden_size}"
            )

        batch_size, seq_len, hidden_size = original_shape

        # Reshape to [B*S, H] for adapter processing, so each token's hidden state is an item.
        processed_llm_state = llm_hidden_state.reshape(-1, hidden_size)
        
        num_total_items = processed_llm_state.shape[0] # This is B*S
        output_adapter_effects = []

        # Reset resonant network state at the beginning of each forward pass.
        # This ensures that the adapter starts from a consistent initial state for each
        # invocation of forward() (e.g., when called by an LLM layer).
        # The resonant network's internal state WILL EVOLVE AND CARRY OVER
        # as it processes items sequentially within the loop below for this single forward pass.
        self.reset_resonant_network_state()

        # --- Batch (Item) Processing Note ---
        # The loop iterates B*S times if the input llm_hidden_state was [B,S,H].
        # The resonant_network.step() is called for self.synthnn_steps for each item.
        # Since reset_resonant_network_state() is called *before* this loop,
        # the internal state of the resonant network (phases, amplitudes)
        # *will carry over* from the processing of item `i` to item `i+1`.
        # This allows the adapter to capture short-term temporal dynamics across
        # the items processed within this forward pass.

        for i in range(num_total_items):
            single_item_state = processed_llm_state[i] # [hidden_size]

            # 1. Project LLM state to control SynthNN external inputs
            synthnn_external_signals_unactivated = self.llm_to_synthnn_projection(single_item_state) # [num_resonant_nodes]
            synthnn_external_signals = self.input_activation(synthnn_external_signals_unactivated)


            # Convert tensor signals to dictionary for resonant_network.step()
            external_input_dict = {
                self._node_ids_sorted[j]: synthnn_external_signals[j].item()
                for j in range(self.num_resonant_nodes)
            }

            # 2. Evolve the resonant network
            # The resonant network's state (phases, amplitudes) evolves here.
            # The `external_inputs` modulate amplitudes as per `amp *= (1 + tanh(signal))`.
            for _ in range(self.synthnn_steps):
                self.resonant_network.step(dt=self.synthnn_dt, external_inputs=external_input_dict)
                # Note: The external_input_dict is applied at each step.
                # This means a sustained input. If a pulse-like input is desired,
                # it should be applied only at the first step or decay over steps.

            # 3. Get the resulting state from SynthNN
            synthnn_output_state_vector = self._get_synthnn_state_vector(device=single_item_state.device) # [num_nodes*2]

            # 4. Project SynthNN state back to LLM hidden dimension
            adapter_effect = self.synthnn_to_llm_projection(synthnn_output_state_vector) # [llm_hidden_size]
            output_adapter_effects.append(adapter_effect)

        # Stack adapter outputs for all items (tokens)
        batched_adapter_effect = torch.stack(output_adapter_effects, dim=0) # [B*S, hidden_size]

        # Reshape back to [B, S, H]
        batched_adapter_effect_reshaped = batched_adapter_effect.reshape(batch_size, seq_len, hidden_size)

        # Add to the original LLM hidden state (residual connection)
        # The adapter's effect is added to each token's hidden state.
        adapted_llm_hidden_state = llm_hidden_state + batched_adapter_effect_reshaped
            
        return adapted_llm_hidden_state


if __name__ == '__main__':
    # Example Usage (requires synthnn to be installed or placeholders to be active)
    print("Running SynthNNAdapter Example...")

    # Configuration
    batch_sz = 4
    seq_len = 10 # Optional sequence length
    llm_hidden_sz = 768 # Example LLM hidden size (like BERT base)
    # For Llama-3.2-1B, hidden_size is typically 2048.
    # We'll use llm_hidden_sz for this generic example.
    
    num_nodes = 16
    s_dt = 0.01
    s_steps = 5

    # Instantiate adapter
    adapter = SynthNNAdapter(
        llm_hidden_size=llm_hidden_sz,
        num_resonant_nodes=num_nodes,
        synthnn_dt=s_dt,
        synthnn_steps=s_steps
    )

    # Create dummy LLM hidden state
    # Option 1: Sequence input
    dummy_llm_input_seq = torch.randn(batch_sz, seq_len, llm_hidden_sz)
    # Option 2: Already processed input (e.g., [CLS] token output) - this adapter now expects 3D input
    # dummy_llm_input_proc = torch.randn(batch_sz, llm_hidden_sz) 

    print(f"Adapter resonant network: {adapter.resonant_network.name} with {len(adapter.resonant_network.nodes)} nodes.")
    # Store initial state to compare after a forward pass
    initial_node_states_before_run = {
        nid: (adapter.resonant_network.nodes[nid].phase, adapter.resonant_network.nodes[nid].amplitude)
        for nid in adapter._node_ids_sorted
    }
    initial_time_before_run = adapter.resonant_network.time


    print("\n--- Testing with sequence input (now expected) ---")
    # Test with sequence input
    try:
        output_seq = adapter(dummy_llm_input_seq)
        print(f"Input shape (sequence): {dummy_llm_input_seq.shape}")
        print(f"Output shape (sequence): {output_seq.shape}")
        assert output_seq.shape == dummy_llm_input_seq.shape
        print("Sequence input test successful.")

        # Check if network state was reset after the forward pass for the next call
        # (by comparing current state to the class-level stored initial state)
        # This test assumes reset_resonant_network_state() correctly resets to the state captured at __init__.
        # Note: The forward pass itself calls reset_resonant_network_state() at its start.
        # So, after a forward pass, the network *is* in its initial state *before* the loop.
        # The state *during* the loop for each token is dynamic.
        # To test the reset function more directly, we can call it and then check.
        
        adapter.reset_resonant_network_state() # Explicitly reset
        
        for nid in adapter._node_ids_sorted:
            assert adapter.resonant_network.nodes[nid].phase == adapter.initial_phases_map[nid], f"Node {nid} phase not reset"
            assert adapter.resonant_network.nodes[nid].amplitude == adapter.initial_amplitudes_map[nid], f"Node {nid} amplitude not reset"
        assert adapter.resonant_network.time == adapter.initial_time, "Network time not reset"
        print("Resonant network state reset mechanism appears to work correctly.")

    except Exception as e:
        print(f"Error during sequence input test: {e}")
        import traceback
        traceback.print_exc()

    # Check if learnable parameters exist
    learnable_params = [p for p in adapter.parameters() if p.requires_grad]
    if learnable_params:
        print(f"\nAdapter has {len(learnable_params)} learnable parameter groups.")
        for name, param in adapter.named_parameters():
            if param.requires_grad:
                print(f"  Learnable: {name} with shape {param.shape}")
    else:
        print("\nAdapter has no learnable parameters (this is unexpected).")

    print("\nSynthNNAdapter Example Finished.") 