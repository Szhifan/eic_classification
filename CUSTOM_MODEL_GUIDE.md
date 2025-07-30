# Custom Llama Model Integration Guide

This guide explains how to use the customized Llama models from the `models/` directory with the existing experiment scripts.

## Overview

The project now supports both standard Hugging Face models and the custom Llama models that include advanced attention mechanisms (unsink attention, bidirectional attention, etc.) adapted from research papers.

## Files Modified/Created

1. **`models/modeling_llama_classification.py`** - New classification wrapper for the custom Llama model
2. **`finetuning_llm_seqc/model_loader.py`** - Updated to support custom models
3. **`finetune_EIC SeqC.py`** - Updated with new configuration options
4. **`test_custom_model.py`** - Test script to verify integration

## Configuration Options

### Basic Settings

```python
# In finetune_EIC SeqC.py
model_path = 'path/to/your/llama/model'  # Path to your Llama model
use_custom_llama = True  # Set to True to use custom implementation
```

### Custom Model Arguments

The custom Llama model supports several advanced configurations:

```python
model_args = {
    'architecture': 'INPLACE',  # Options: NONE, INPLACE, EXTEND, INTER, EXTRA
    'mask_type': 'MASK0',       # Options: MASK0, BACK
    'num_unsink_layers': 0,     # Number of layers with unsink attention
    'num_bidir_layers': 0,      # Number of layers with bidirectional attention  
    'unsink_layers': None,      # Specific layers for unsink attention (list)
    'bidir_layers': None,       # Specific layers for bidirectional attention (list)
    'res_connect': 3,           # Residual connection interval
    'freeze_type': None,        # Freezing strategy: 'all', 'backbone', 'default', None
    'num_unfreeze_layers': 0,   # Number of layers to keep unfrozen
    'model_init': True,         # Initialize extended layers with forward params
    'num_classifier_layers': 1, # Number of classification head layers
}
```

## Architecture Types

- **NONE**: Standard Llama model without modifications
- **INPLACE**: Modify existing layers in place
- **EXTEND**: Add additional layers to the model
- **INTER**: Intermediate processing architecture
- **EXTRA**: Extra layer processing architecture

## Attention Mechanisms

- **Unsink Attention**: Allows certain layers to attend to all previous tokens without causality constraints
- **Bidirectional Attention**: Enables bidirectional attention in specified layers
- **MASK0/BACK**: Different masking strategies for attention

## Usage Examples

### 1. Standard Usage (Existing Behavior)
```python
use_custom_llama = False
# Uses AutoModelForSequenceClassification
```

### 2. Custom Llama with Default Settings
```python
use_custom_llama = True
model_args = {
    'architecture': 'INPLACE',
    'num_classifier_layers': 1,
}
```

### 3. Advanced Configuration
```python
use_custom_llama = True
model_args = {
    'architecture': 'EXTEND',
    'mask_type': 'BACK',
    'num_unsink_layers': 2,
    'num_bidir_layers': 1,
    'freeze_type': 'backbone',
    'num_unfreeze_layers': 2,
    'num_classifier_layers': 2,
}
```

## Testing

Run the test script to verify the integration:

```bash
python test_custom_model.py
```

This will test:
- Custom model creation
- Configuration parsing
- Model loader integration
- Forward pass functionality

## Model Loading Process

1. **Configuration Loading**: Loads base Llama config from the model path
2. **Custom Arguments Application**: Applies BackwardSupportedArguments to the config
3. **Model Creation**: Creates LlamaForSequenceClassification with custom base model
4. **Weight Loading**: Attempts to load pre-trained weights (with fallbacks)
5. **Classification Head Setup**: Configures the classification head based on label configuration

## Error Handling

The model loader includes several fallback mechanisms:
- If custom weights can't be loaded, falls back to random initialization
- If custom model creation fails, can fall back to standard AutoModel
- Supports both local model files and Hugging Face Hub models

## Performance Considerations

- Custom models may have different memory requirements depending on architecture
- Extended architectures (EXTEND) will have more parameters
- Freezing strategies can significantly reduce trainable parameters
- Use gradient checkpointing for memory efficiency during training

## Troubleshooting

1. **Import Errors**: Ensure the project root is in Python path
2. **Config Errors**: Verify all required config parameters are set
3. **Weight Loading Issues**: Check model path and file permissions
4. **Memory Issues**: Try reducing batch size or using gradient checkpointing

## Future Extensions

The framework is designed to support additional custom architectures:
- Add new attention mechanisms to `modeling_llama.py`
- Extend `BackwardSupportedArguments` for new configuration options
- Add support for other base model types beyond Llama
