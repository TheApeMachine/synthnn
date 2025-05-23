import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AdamW, 
    get_linear_schedule_with_warmup
)
import sys
import os
import json # For saving adapter configs

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) 
sys.path.insert(0, project_root)

try:
    from synthnn.adapters.llm_adapter import SynthNNAdapter
except ModuleNotFoundError:
    print("Error: SynthNNAdapter not found. Ensure a) this script is in 'examples/', b) 'synthnn' is in project root, c) __init__.py files are present.")
    print(f"Python path: {sys.path}")
    print(f"Calculated project root: {project_root}")
    sys.exit(1)
except ImportError as e:
    print(f"Error importing SynthNNAdapter: {e}. Ensure synthnn components are correctly installed/available.")
    sys.exit(1)

# --- Configuration ---
MODEL_NAME = "meta-llama/Llama-3.2-1B" # Replace with desired model
# MODEL_NAME = "gpt2" # Smaller model for faster testing without Llama access
ADAPTER_NUM_RESONANT_NODES = 32
ADAPTER_SYNTHNN_DT = 0.01
ADAPTER_SYNTHNN_STEPS = 10

LEARNING_RATE = 5e-5
BATCH_SIZE = 2 # Keep small for memory, adjust based on your hardware
NUM_EPOCHS = 1 # Start with 1 epoch for testing
MAX_LENGTH = 128 # Max sequence length for tokenization
WARMUP_STEPS = 0
OUTPUT_DIR = "./trained_adapters"

# --- Helper Functions ---

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif torch.backends.mps.is_available(): # For Apple Silicon, if you want to test
    #     return torch.device("mps")
    return torch.device("cpu")

def load_model_and_tokenizer(model_name_or_path: str, device: torch.device):
    print(f"Loading tokenizer and model for {model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32) # Use float32 for stability
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        if "meta-llama" in model_name_or_path:
            print("Please ensure you have accepted Llama terms and are logged in via `huggingface-cli login`.")
        sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to eos_token: {tokenizer.eos_token}")
    
    model.to(device)
    return model, tokenizer

def attach_synthnn_adapter_to_layer(decoder_layer, adapter_instance):
    if not hasattr(decoder_layer, 'mlp'):
        print(f"Warning: Layer {type(decoder_layer)} lacks 'mlp' attribute. Adapter not attached.")
        return False
    
    original_mlp_forward = decoder_layer.mlp.forward
    def new_mlp_forward(hidden_states: torch.Tensor, *args, **kwargs):
        mlp_output = original_mlp_forward(hidden_states, *args, **kwargs)
        return adapter_instance(mlp_output)
    
    decoder_layer.mlp.forward = new_mlp_forward
    # print(f"SynthNNAdapter attached to MLP of layer: {decoder_layer}")
    return True

def create_and_attach_adapters(model, llm_hidden_size: int, num_layers_to_adapt: int, adapter_config: dict, device: torch.device):
    print("\nFreezing base model parameters...")
    for param in model.parameters():
        param.requires_grad = False

    adapters_list = []
    actual_layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'): # Llama-style
        actual_layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'): # GPT-2 style
        actual_layers = model.transformer.h
    else:
        raise ValueError("Cannot find decoder layers in the model structure.")
    
    if num_layers_to_adapt > len(actual_layers):
        print(f"Warning: Requested to adapt {num_layers_to_adapt} layers, but model only has {len(actual_layers)}. Adapting all layers.")
        num_layers_to_adapt = len(actual_layers)

    print(f"\nCreating and attaching SynthNNAdapters to {num_layers_to_adapt} layers...")
    for i in range(num_layers_to_adapt):
        adapter = SynthNNAdapter(
            llm_hidden_size=llm_hidden_size,
            num_resonant_nodes=adapter_config['num_resonant_nodes'],
            synthnn_dt=adapter_config['synthnn_dt'],
            synthnn_steps=adapter_config['synthnn_steps']
        ).to(device)
        
        if attach_synthnn_adapter_to_layer(actual_layers[i], adapter):
            for param in adapter.parameters(): # Ensure adapter params are trainable
                param.requires_grad = True
            adapters_list.append({"layer_index": i, "adapter": adapter, "config": adapter_config})
            print(f"  Attached adapter to layer {i}")
        else:
            print(f"  Failed to attach adapter to layer {i}")
            
    return adapters_list

class SimpleTextDataset(Dataset):
    def __init__(self, texts: list, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # For Causal LM, input_ids and labels are often the same, with loss calculation handling shifts.
        # Or, labels are input_ids shifted, and padding tokens in labels are ignored (-100).
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        input_ids = encoding.input_ids.squeeze(0) # Remove batch dim from tokenizer output
        attention_mask = encoding.attention_mask.squeeze(0)
        
        # Create labels: shift input_ids to the right. Pad tokens in labels should be -100.
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100 # Last token has no target for prediction in this simple shift
        # Mask padding tokens in labels
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def prepare_dataloader(texts: list, tokenizer, batch_size: int, max_length: int):
    dataset = SimpleTextDataset(texts, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def prepare_optimizer_and_scheduler(adapters_params_list: list, learning_rate: float, num_training_steps: int, warmup_steps: int):
    optimizer = AdamW(adapters_params_list, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    return optimizer, scheduler

def save_adapters(adapters_list_with_meta: list, save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for item in adapters_list_with_meta:
        adapter = item["adapter"]
        layer_idx = item["layer_index"]
        adapter_config = item["config"]
        
        adapter_save_path = os.path.join(save_dir, f"synthnn_adapter_layer_{layer_idx}.pt")
        config_save_path = os.path.join(save_dir, f"synthnn_adapter_layer_{layer_idx}_config.json")
        
        torch.save(adapter.state_dict(), adapter_save_path)
        with open(config_save_path, 'w') as f:
            json.dump(adapter_config, f, indent=4)
        print(f"Saved adapter for layer {layer_idx} to {adapter_save_path} and config to {config_save_path}")

# --- Main Training Script ---
if __name__ == '__main__':
    device = get_device()
    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device)
    llm_hidden_size = model.config.hidden_size

    adapter_hyperparams = {
        'num_resonant_nodes': ADAPTER_NUM_RESONANT_NODES,
        'synthnn_dt': ADAPTER_SYNTHNN_DT,
        'synthnn_steps': ADAPTER_SYNTHNN_STEPS
    }
    
    # Adapting all layers for this example
    num_layers_to_adapt = model.config.num_hidden_layers 
    attached_adapters_with_meta = create_and_attach_adapters(model, llm_hidden_size, num_layers_to_adapt, adapter_hyperparams, device)

    if not attached_adapters_with_meta:
        print("No adapters were attached. Exiting training.")
        sys.exit(1)

    # Prepare data (simple example)
    # Replace with your actual dataset loading and preprocessing
    sample_texts = [
        "def hello_world():\n    print(\"Hello, SynthNN!\")",
        "import torch\nclass SimpleNet(torch.nn.Module):\n    def __init__(self): super().__init__()",
        "for i in range(10):\n    print(f\"Iteration {i}\")",
        "The resonant frequencies interact harmonically.",
        "Adapting LLMs with novel architectures is key.",
        "SynthNN provides a unique approach to temporal modeling.",
        "This training loop fine-tunes only the adapter parameters.",
        "Causal language modeling predicts the next token."
    ] * 10 # Multiply to make the dataset a bit larger for demonstration
    
    print(f"\nPreparing DataLoader with {len(sample_texts)} samples...")
    train_dataloader = prepare_dataloader(sample_texts, tokenizer, BATCH_SIZE, MAX_LENGTH)
    num_training_steps = len(train_dataloader) * NUM_EPOCHS

    # Prepare optimizer and scheduler
    adapter_params_to_optimize = []
    for item in attached_adapters_with_meta:
        adapter_params_to_optimize.extend(list(item["adapter"].parameters()))
    
    if not adapter_params_to_optimize:
        print("No trainable adapter parameters found. Exiting.")
        sys.exit(1)
        
    print(f"\nOptimizing {sum(p.numel() for p in adapter_params_to_optimize)} parameters from adapters.")
    optimizer, scheduler = prepare_optimizer_and_scheduler(adapter_params_to_optimize, LEARNING_RATE, num_training_steps, WARMUP_STEPS)

    # Training loop
    print("\nStarting training...")
    model.train() # Set model to training mode (affects dropout, etc., though base is frozen)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            if loss is None:
                print("Warning: Loss is None. Check model output and label preparation.")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter_params_to_optimize, max_norm=1.0) # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_dataloader):
                print(f"  Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}, Avg Loss: {total_loss/(batch_idx+1):.4f}")
        
        avg_epoch_loss = total_loss / len(train_dataloader)
        print(f"End of Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f}")

    print("\nTraining finished.")

    # Save adapters
    print("\nSaving trained adapters...")
    save_adapters(attached_adapters_with_meta, OUTPUT_DIR)

    print(f"\nAdapters saved to {OUTPUT_DIR}. Training script complete.") 