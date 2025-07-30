#!/usr/bin/env python3
"""
Test script to verify that custom Llama model loading works without infinite loops
"""

import torch
import os
import sys
from finetuning_llm_seqc.model_loader import ModelLoader

def test_custom_model_loading():
    print("Testing custom Llama model loading...")
    
    # Create a ModelLoader instance
    loader = ModelLoader()
    
    # Test parameters
    model_path = "meta-llama/Llama-3.1-8B"  # You can change this to your model path
    labels = ["label1", "label2"]  # Sample labels
    label2id = {"label1": 0, "label2": 1}
    id2label = {0: "label1", 1: "label2"}
    
    try:
        print(f"Attempting to load model from: {model_path}")
        print("Using custom Llama implementation...")
        
        # Test with custom Llama model
        model, tokenizer = loader.load_model_from_path(
            model_path=model_path,
            device_map='cpu',  # Use CPU to avoid GPU memory issues during testing
            labels=labels,
            label2id=label2id,
            id2label=id2label,
            use_custom_llama=True,
            architecture="NONE",  # Use NONE to avoid complex configurations
            mask_type="MASK0",
            num_unsink_layers=0,
            num_bidir_layers=0
        )
        
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Tokenizer type: {type(tokenizer)}")
        print(f"Model config: {model.config}")
        
        # Test a simple forward pass
        print("\nTesting forward pass...")
        test_input = "This is a test sentence."
        inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)
        
        print("Running inference...")
        with torch.no_grad():
            outputs = model(**inputs)
            
        print("✅ Forward pass successful!")
        print(f"Output logits shape: {outputs.logits.shape}")
        print(f"Predicted probabilities: {torch.softmax(outputs.logits, dim=-1)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_custom_model_loading()
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)
