from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoModelForSequenceClassification,
    AutoConfig,
    LlamaConfig
)
from models.modeling_llama import LlamaForSequenceClassification
from models.modeling_utils import BackwardSupportedArguments
from transformers.modeling_outputs import SequenceClassifierOutput
import os

class ModelLoader:
    def __init__(self) -> None:
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit = True, # Activate 4-bit precision base model loading
            bnb_4bit_use_double_quant = True, # Activate nested quantization for 4-bit base models (double quantization)
            bnb_4bit_quant_type = "nf4",# Quantization type (fp4 or nf4)
            bnb_4bit_compute_dtype = torch.bfloat16, # Compute data type for 4-bit base models
            )


    def load_model_from_path(self, model_path: str, device_map='auto', 
                             labels=None, label2id=None, id2label=None, 
                             emb_type=None, input_type=None, use_custom_llama=False,
                             **model_args):
        """
        Load model from path with support for custom Llama models
        
        Args:
            model_path: Path to the model
            device_map: Device mapping configuration
            labels: List of labels
            label2id: Label to ID mapping
            id2label: ID to label mapping
            emb_type: Embedding type for specialized models
            input_type: Input type configuration
            use_custom_llama: Whether to use custom Llama implementation
            **model_args: Additional model arguments for custom Llama
        """

        print('Loading model from...', model_path)	

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        if use_custom_llama and ('llama' in model_path.lower() or 'llama' in str(model_path).lower()):
            # Use custom Llama model
            print("Using custom Llama model implementation...")
            
            try:
                # Load configuration
                print("Loading configuration...")
                config = LlamaConfig.from_pretrained(model_path)
                
                # Update config with backward supported arguments
                print("Updating config with backward supported arguments...")
                backward_args = BackwardSupportedArguments(**model_args)
                for key, value in backward_args.to_dict().items():
                    setattr(config, key, value)
                
                # Set classification specific config
                config.num_labels = len(labels) if labels else 2
                config.id2label = id2label
                config.label2id = label2id
                config.pad_token_id = tokenizer.pad_token_id
                
                print(f"Config setup complete. Num labels: {config.num_labels}")
                
                # Load the custom model
                print("Initializing custom Llama model...")
                model = LlamaForSequenceClassification(config)
                print("Custom model initialized successfully")
                
                # Load pre-trained weights if available
                if os.path.exists(model_path):
                    try:
                        print("Looking for pre-trained weights...")
                        # Try to load state dict (handle potential key mismatches)
                        model_file = os.path.join(model_path, "pytorch_model.bin")
                        if os.path.exists(model_file):
                            print("Loading pytorch_model.bin...")
                            state_dict = torch.load(model_file, map_location="cpu")
                            model.load_state_dict(state_dict, strict=False)
                            print("Loaded pre-trained weights successfully")
                        else:
                            print("No pytorch_model.bin found, trying safetensors...")
                            # Try to load safetensors format
                            safetensors_file = os.path.join(model_path, "model.safetensors")
                            if os.path.exists(safetensors_file):
                                try:
                                    from safetensors.torch import load_file
                                    print("Loading safetensors...")
                                    state_dict = load_file(safetensors_file)
                                    model.load_state_dict(state_dict, strict=False)
                                    print("Loaded safetensors weights successfully")
                                except ImportError:
                                    print("safetensors not available, using randomly initialized model...")
                                except Exception as e3:
                                    print(f"Could not load safetensors: {e3}")
                                    print("Using randomly initialized model...")
                            else:
                                print("No model weights found, using randomly initialized model...")
                    except Exception as e:
                        print(f"Could not load pre-trained weights: {e}")
                        print("Initializing with random weights...")
                        
            except Exception as e:
                print(f"Error during custom model loading: {e}")
                print("Falling back to standard model loading...")
                raise e
        
        else:
            # Use standard AutoModel approach
            print("Using standard AutoModelForSequenceClassification...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                quantization_config=self.bnb_config,
                device_map=device_map, 
                num_labels=len(labels) if labels else 2,
            ) 
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.id2label = id2label
            model.config.label2id = label2id

        # Handle Mistral specific configuration
        if 'Mistral' in model_path:
            model.config.sliding_window = 4096
            
        return model, tokenizer
    
