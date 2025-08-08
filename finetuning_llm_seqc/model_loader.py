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
            llm_int8_skip_modules = ["classifier", "latent_attention", "pooler"], # Skip quantization for these modules
            )

    def get_quantization_config(self, skip_custom_layers=True):
        """
        Get quantization config with optional skipping of custom layers
        
        Args:
            skip_custom_layers: Whether to skip quantization for custom layers
        """
        if not torch.cuda.is_available():
            return None
            
        config = {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        }
        
        if skip_custom_layers:
            config["llm_int8_skip_modules"] = ["classifier", "latent_attention", "pooler", "score"]
            
        return BitsAndBytesConfig(**config)


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
        config = AutoConfig.from_pretrained(model_path)
        if 'llama' in str(model_path).lower():
            print("Detected Llama model - preparing custom configuration...")
            config = self._prepare_custom_llama_config(model_path, labels, label2id, id2label, tokenizer, **model_args)
            tokenizer.pad_token = tokenizer.eos_token
            
            # Use the enhanced quantization config for Llama models
            quant_config = self.get_quantization_config(skip_custom_layers=True) if torch.cuda.is_available() else None
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=config,  # Use custom config for Llama
                quantization_config=quant_config,
                device_map=device_map, 
            )
            model.config.pad_token_id = tokenizer.pad_token_id
        
        else:
            print("Loading with AutoModelForSequenceClassification...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=config,  # Pass custom config if it's a Llama model
                quantization_config=self.bnb_config if torch.cuda.is_available() else None,
                device_map=device_map, 
            ) 

        config.num_labels = len(labels) if labels else 2
        model.config.id2label = id2label
        model.config.label2id = label2id
        
        # Final safety check: ensure custom layers are properly configured
        if 'llama' in str(model_path).lower():
            self._ensure_custom_layers_precision(model)
            
        return model, tokenizer
    
    def _ensure_custom_layers_precision(self, model):
        """
        Final safety check to ensure custom layers use float32 precision
        """
        custom_layer_names = ["classifier", "latent_attention", "pooler"]
        
        for name in custom_layer_names:
            if hasattr(model, name):
                layer = getattr(model, name)
                if layer is not None:
                    layer = layer.float()
                    setattr(model, name, layer)
                    print(f"Ensured {name} layer uses float32 precision")
        
        # Check if model has _ensure_float32_precision method and call it
        if hasattr(model, '_ensure_float32_precision'):
            model._ensure_float32_precision()
            print("Called model's _ensure_float32_precision method")
    
    def _prepare_custom_llama_config(self, model_path: str, labels=None, label2id=None, 
                                    id2label=None, tokenizer=None, **model_args):
        """
        Prepare custom Llama configuration with backward supported arguments for AutoModelForSequenceClassification
        """

        # Load configuration
        print("Loading LlamaConfig...")
        config = LlamaConfig.from_pretrained(model_path)
        
        # Set up quantization parameters
        print("Setting up model arguments...")
        # Create backward supported arguments
        backward_args = BackwardSupportedArguments(**model_args)
        
        # Update config with backward arguments
        for key, value in backward_args.to_dict().items():
            setattr(config, key, value)
        
        # Set classification specific config
        config.num_labels = len(labels) if labels else 2
        config.id2label = id2label
        config.label2id = label2id
        config.pad_token_id = tokenizer.pad_token_id if tokenizer else None
        
        print(f"Custom Llama config prepared. Num labels: {config.num_labels}")
        print(f"Architecture: {getattr(config, 'architecture', 'NONE')}")
        
        return config
 