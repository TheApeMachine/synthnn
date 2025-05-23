import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add the parent directory of 'synthnn' to the Python path
# This is to ensure that the synthnn module can be found when running this script directly
# Assumes this script is in examples/ and synthnn is in the parent directory (e.g., project_root/synthnn)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # This should be the 'synthnn' project root
sys.path.insert(0, project_root)

try:
    from synthnn.adapters.llm_adapter import SynthNNAdapter
except ModuleNotFoundError:
    print("Error: SynthNNAdapter not found. ")
    print("Ensure that the script is in the 'examples' directory and 'synthnn' is in the project root.")
    print(f"Python path: {sys.path}")
    print(f"Project root calculated as: {project_root}")
    sys.exit(1)
except ImportError as e:
    print(f"Error importing SynthNNAdapter: {e}")
    print("Ensure that synthnn core components are available or placeholders are active in llm_adapter.py.")
    sys.exit(1)

def attach_synthnn_adapter_to_llama_layer(llama_layer, synthnn_adapter_instance):
    """
    Attaches a SynthNNAdapter instance after the MLP block of a LlamaDecoderLayer.

    Args:
        llama_layer: An instance of a LlamaDecoderLayer (or similar).
        synthnn_adapter_instance: An instance of SynthNNAdapter.
    """
    # The specific path to the MLP block might vary slightly based on the Llama model version
    # For Llama models in Hugging Face, it's typically `mlp`.
    if not hasattr(llama_layer, 'mlp'):
        print(f"Warning: Llama layer {type(llama_layer)} does not have an 'mlp' attribute. Cannot attach adapter.")
        return

    original_mlp_forward = llama_layer.mlp.forward

    # Define the new forward pass for the MLP block
    def new_mlp_forward(hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Call the original MLP forward pass
        mlp_output = original_mlp_forward(hidden_states, *args, **kwargs)
        
        # Pass the MLP output through the SynthNNAdapter
        # The adapter expects [batch, seq_len, hidden_size]
        # mlp_output is already in this shape typically from LlamaMLP
        adapted_output = synthnn_adapter_instance(mlp_output)
        
        return adapted_output

    # Replace the forward method of the mlp block in this specific layer
    llama_layer.mlp.forward = new_mlp_forward
    print(f"SynthNNAdapter attached to MLP of layer: {llama_layer}")


if __name__ == '__main__':
    # model_name = "meta-llama/Llama-3.2-1B"
    # model_name = "NousResearch/Llama-2-7b-chat-hf" # Example for another model
    model_name = "gpt2" # For a much smaller model for quick testing if Llama access is an issue

    print(f"Loading tokenizer and model for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load with torch_dtype=torch.float16 for smaller memory footprint if compatible
        # For robust testing, float32 is safer if hardware allows.
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32) 
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("Please ensure you have accepted the terms for Llama models on Hugging Face Hub if applicable,")
        print("and that you are logged in via `huggingface-cli login`.")
        sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer pad_token was None, set to eos_token: {tokenizer.eos_token}")

    # Get LLM's hidden size
    llm_hidden_size = model.config.hidden_size
    print(f"LLM hidden size: {llm_hidden_size}")

    # --- SynthNNAdapter Configuration ---
    # These should match or be compatible with your adapter's capabilities
    num_resonant_nodes_adapter = 32 
    synthnn_dt_adapter = 0.01
    synthnn_steps_adapter = 10

    # Create adapter instances - one per Llama layer you want to modify
    # Here, we create a list of adapters. You could share one or use distinct ones.
    num_llama_layers = model.config.num_hidden_layers
    print(f"Llama model has {num_llama_layers} decoder layers.")
    
    synthnn_adapters = []
    for i in range(num_llama_layers):
        adapter = SynthNNAdapter(
            llm_hidden_size=llm_hidden_size,
            num_resonant_nodes=num_resonant_nodes_adapter,
            synthnn_dt=synthnn_dt_adapter,
            synthnn_steps=synthnn_steps_adapter
        )
        synthnn_adapters.append(adapter)
        print(f"Created SynthNNAdapter for layer {i}")

    # Freeze all parameters in the base LLM
    print("\nFreezing all parameters in the base LLM...")
    for param in model.parameters():
        param.requires_grad = False

    # Attach adapters and ensure only their parameters are trainable
    print("\nAttaching SynthNNAdapters to Llama layers...")
    # The path to layers might be model.model.layers for Llama
    llama_decoder_layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        llama_decoder_layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'): # For GPT-2 style models
         llama_decoder_layers = model.transformer.h
    else:
        print("Could not find decoder layers in the model structure. Exiting.")
        sys.exit(1)

    for i, layer in enumerate(llama_decoder_layers):
        if i < len(synthnn_adapters):
            # Move adapter to the same device as the model layer
            synthnn_adapters[i].to(model.device) 
            attach_synthnn_adapter_to_llama_layer(layer, synthnn_adapters[i])
            # Ensure adapter parameters are trainable
            for param in synthnn_adapters[i].parameters():
                param.requires_grad = True
        else:
            print(f"No adapter created for layer {i}, skipping attachment.")

    # Verify trainable parameters
    print("\nTrainable parameters after attaching adapters:")
    total_trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  Trainable: {name} (shape: {param.shape})")
            total_trainable_params += param.numel()
    
    # Also count parameters from adapters if they are not part of model.named_parameters() directly
    # (which they are not, as they are attached via method override)
    # So, we sum them directly from the adapter list
    total_adapter_params = 0
    for i, adapter in enumerate(synthnn_adapters):
        adapter_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        print(f"  SynthNNAdapter for layer {i} has {adapter_params} trainable parameters.")
        total_trainable_params += adapter_params

    print(f"Total trainable parameters in the model (including adapters): {total_trainable_params}")
    if total_trainable_params == 0:
        print("Warning: No trainable parameters found! Check adapter attachment and requires_grad settings.")

    # --- Test forward pass ---
    print("\nTesting forward pass with adapted model...")
    # Prepare some dummy input
    input_text = "Hello, SynthNN world! This is a test."
    print(f"Input text: {input_text}")
    
    try:
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
        print(f"Tokenized input shape: {inputs.input_ids.shape}")

        # Perform a forward pass
        with torch.no_grad(): # No need to compute gradients for this test
            outputs = model(**inputs)
        
        # Check output shape (logits)
        # For CausalLM, output is a CausalLMOutputWithPast object, logits are the first element
        logits = outputs.logits
        print(f"Output logits shape: {logits.shape}") # Expected: [batch_size, seq_len, vocab_size]
        assert logits.shape[0] == inputs.input_ids.shape[0]
        assert logits.shape[1] == inputs.input_ids.shape[1]
        assert logits.shape[2] == model.config.vocab_size
        print("Forward pass successful with adapted Llama model!")

    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

    print("\nIntegration script finished.") 