from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoModelForSequenceClassification,
    AutoConfig,
    LlamaConfig
)
from custom_models.modeling_llama import LlamaForSequenceClassification
from custom_models.modeling_utils import BackwardSupportedArguments, get_custom_model
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
                # Add required attributes for get_custom_model
                setattr(backward_args, 'model_name_or_path', model_path)
                setattr(backward_args, 'torch_dtype', getattr(backward_args, 'torch_dtype', 'auto'))
                setattr(backward_args, 'trust_remote_code', getattr(backward_args, 'trust_remote_code', False))
                setattr(backward_args, 'cache_dir', getattr(backward_args, 'cache_dir', None))
                setattr(backward_args, 'model_revision', getattr(backward_args, 'model_revision', None))
                setattr(backward_args, 'token', getattr(backward_args, 'token', None))
                
                for key, value in backward_args.to_dict().items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                
                # Set classification specific config
                config.num_labels = len(labels) if labels else 2
                config.id2label = id2label
                config.label2id = label2id
                config.pad_token_id = tokenizer.pad_token_id
                
                print(f"Config setup complete. Num labels: {config.num_labels}")
                
                # Load the custom model using the designated function
                print("Initializing custom Llama model with get_custom_model function...")
                model = get_custom_model(backward_args, config, LlamaForSequenceClassification)
                print("Custom model loaded successfully")
                        
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
        setattr(model, 'accepts_loss_kwargs', False)  # Ensure Trainer does not pass unexpected kwargs
        return model, tokenizer
    
