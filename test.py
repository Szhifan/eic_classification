#!/usr/bin/env python3
"""
Example of loading the customized Llama model with backward-supported arguments.
This demonstrates how to use the enhanced Llama model with different architectures,
attention masks, and configuration options.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from models.modeling_utils import BackwardSupportedArguments, get_model
import models  # This registers the custom models

def example_basic_loading():
    """Basic example of loading the customized Llama model"""
    print("=== Basic Loading Example ===")
    
    # Model path - replace with your actual model path
    model_path = "meta-llama/Llama-3.1-8B"  # or your local path
    
    # Create backward supported arguments
    model_args = BackwardSupportedArguments(
        architecture="INPLACE",
        mask_type="MASK0",
        num_unsink_layers=2,
        num_bidir_layers=1,
        freeze_type=None,
        num_classifier_layers=1
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load config and add custom parameters
    config = AutoConfig.from_pretrained(model_path)
    
    # Add custom configuration parameters
    for key, value in model_args.to_dict().items():
        setattr(config, key, value)
    
    # Set classification parameters
    config.num_labels = 3  # Example: 3 classes
    config.problem_type = "single_label_classification"
    
    # Load the customized model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Model loaded: {type(model)}")
    print(f"Architecture: {config.architecture}")
    print(f"Num unsink layers: {config.num_unsink_layers}")
    print(f"Num bidir layers: {config.num_bidir_layers}")
    
    return model, tokenizer

def example_advanced_loading():
    """Advanced example with different architecture configurations"""
    print("\n=== Advanced Loading Example ===")
    
    model_path = "meta-llama/Llama-2-7b-hf"
    
    # Example with EXTEND architecture
    model_args = BackwardSupportedArguments(
        architecture="EXTEND",
        mask_type="BACK",
        num_unsink_layers=4,
        num_bidir_layers=2,
        res_connect=3,
        freeze_type="backbone",
        num_unfreeze_layers=2,
        num_classifier_layers=2,
        model_init=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and configure
    config = AutoConfig.from_pretrained(model_path)
    
    # Add custom parameters
    for key, value in model_args.to_dict().items():
        setattr(config, key, value)
    
    config.num_labels = 5  # Example: 5 classes
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Model architecture: {config.architecture}")
    print(f"Mask type: {config.mask_type}")
    print(f"ResConnect layers: {config.res_connect}")
    print(f"Freeze type: {config.freeze_type}")
    print(f"Classifier layers: {config.num_classifier_layers}")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer

def example_with_specific_layers():
    """Example using specific layer indices instead of number of layers"""
    print("\n=== Specific Layers Example ===")
    
    model_path = "meta-llama/Llama-2-7b-hf"
    
    # Specify exact layers to modify
    model_args = BackwardSupportedArguments(
        architecture="INPLACE",
        mask_type="MASK0",
        unsink_layers="30,31",  # Last 2 layers as unsink
        bidir_layers="28,29",   # 2 layers before as bidirectional
        freeze_type="default",
        num_classifier_layers=1
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(model_path)
    
    # Add custom parameters
    for key, value in model_args.to_dict().items():
        setattr(config, key, value)
    
    config.num_labels = 2  # Binary classification
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Unsink layers: {config.unsink_layers}")
    print(f"Bidir layers: {config.bidir_layers}")
    
    return model, tokenizer

def example_inference():
    """Example of running inference with the loaded model"""
    print("\n=== Inference Example ===")
    
    # Load model (using basic configuration for this example)
    model, tokenizer = example_basic_loading()
    
    # Example text for classification
    texts = [
        "This is a positive example.",
        "This is a negative example.",
        "This is a neutral example."
    ]
    
    # Tokenize
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    print("Predictions:")
    for i, (text, pred) in enumerate(zip(texts, predictions)):
        print(f"Text: {text}")
        print(f"Probabilities: {pred.cpu().numpy()}")
        print(f"Predicted class: {pred.argmax().item()}")
        print()

def example_model_info():
    """Display detailed model information"""
    print("\n=== Model Information Example ===")
    
    model_args = BackwardSupportedArguments(
        architecture="INTER",
        mask_type="BACK",
        num_unsink_layers=3,
        num_bidir_layers=2,
        res_connect=2,
        num_classifier_layers=2
    )
    
    # Display suffix that would be used for model naming
    suffix = model_args.get_suffix()
    print(f"Model suffix: {suffix}")
    
    # Display all arguments
    print("Model arguments:")
    for key, value in model_args.to_dict().items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    print("Customized Llama Model Loading Examples")
    print("=" * 50)
    
    # Run examples

    # Basic loading
    model1, tokenizer1 = example_basic_loading()
    
    # Advanced loading
    model2, tokenizer2 = example_advanced_loading()
    
    # Specific layers
    model3, tokenizer3 = example_with_specific_layers()
    
    # Model info
    example_model_info()
    
    # Inference example (comment out if you don't want to run inference)
    # example_inference()

