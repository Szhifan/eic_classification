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

# Register our custom Llama model for sequence classification
# This allows AutoModelForSequenceClassification to find our custom implementation
try:
    from transformers import AutoModel
    # Register the custom model class so AutoModelForSequenceClassification can use it
    AutoModelForSequenceClassification.register(LlamaConfig, LlamaForSequenceClassification)
    print("Custom LlamaForSequenceClassification registered successfully")
except Exception as e:
    print(f"Warning: Failed to register custom Llama model: {e}")
    print("Will fall back to standard model loading if needed")


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
                             emb_type=None, input_type=None, **model_args):
        """
        Load model from path. Automatically uses custom Llama implementation for Llama models.
        
        Args:
            model_path: Path to the model
            device_map: Device mapping configuration
            labels: List of labels
            label2id: Label to ID mapping
            id2label: ID to label mapping
            emb_type: Embedding type for specialized models
            input_type: Input type configuration
            **model_args: Additional model arguments for custom models
        """

        print('Loading model from...', model_path)	

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = 'right'
        
        # Prepare configuration - add custom Llama config if it's a Llama model
        config = None
        if 'llama' in model_path.lower() or 'llama' in str(model_path).lower():
            print("Detected Llama model - preparing custom configuration...")
            config = self._prepare_custom_llama_config(model_path, labels, label2id, id2label, tokenizer, **model_args)
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        config.num_labels = len(labels) if labels else 2
        # Always use AutoModelForSequenceClassification for sequence classification tasks
        print("Loading with AutoModelForSequenceClassification...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,  # Pass custom config if it's a Llama model
            quantization_config=self.bnb_config if torch.cuda.is_available() else None,
            device_map=device_map, 
        ) 

        
        model.config.id2label = id2label
        model.config.label2id = label2id
        print("PAD token ID:", model.config.pad_token_id)
        # Handle Mistral specific configuration
        if 'Mistral' in model_path:
            model.config.sliding_window = 4096
        setattr(model, 'accepts_loss_kwargs', False)  # Ensure Trainer does not pass unexpected kwargs
        return model, tokenizer
    
    def _prepare_custom_llama_config(self, model_path: str, labels=None, label2id=None, 
                                    id2label=None, tokenizer=None, **model_args):
        """
        Prepare custom Llama configuration with backward supported arguments for AutoModelForSequenceClassification
        """
        try:
            # Load configuration
            print("Loading LlamaConfig...")
            config = LlamaConfig.from_pretrained(model_path)
            
            # Set up quantization parameters
            print("Setting up model arguments...")
            # Create backward supported arguments
            backward_args = BackwardSupportedArguments(**model_args)
            
            # Update config with backward arguments
            for key, value in backward_args.to_dict().items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Set classification specific config
            config.num_labels = len(labels) if labels else 2
            config.id2label = id2label
            config.label2id = label2id
            config.pad_token_id = tokenizer.pad_token_id if tokenizer else None
            
            print(f"Custom Llama config prepared. Num labels: {config.num_labels}")
            print(f"Architecture: {getattr(config, 'architecture', 'NONE')}")
            
            return config
            
        except Exception as e:
            print(f"Error preparing custom Llama config: {e}")
            print("Falling back to standard config loading...")
            return None
    
