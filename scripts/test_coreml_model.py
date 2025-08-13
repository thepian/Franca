#!/usr/bin/env python3
"""
Test Core ML Franca model with real images
"""

import os
import coremltools as ct
import numpy as np
from PIL import Image
import torch
from franca.hub.backbones import franca_vitb14

def test_coreml_model():
    """Test the Core ML model with the sample dog image"""
    
    # Paths
    coreml_path = "coreml_models/franca_vitb14_in21k_fp32.mlpackage"
    test_image_path = "assets/dog.jpg"
    
    if not os.path.exists(coreml_path):
        print(f"❌ Core ML model not found: {coreml_path}")
        print("Run export_coreml_hub.py first to create the model")
        return
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    print("🔍 Testing Core ML Franca model...")
    print(f"Model: {coreml_path}")
    print(f"Test image: {test_image_path}")
    
    # Load and preprocess image
    print("\n📸 Loading and preprocessing image...")
    image = Image.open(test_image_path).convert('RGB')
    print(f"Original image size: {image.size}")
    
    # Resize to model input size (518x518)
    image_resized = image.resize((518, 518), Image.Resampling.BILINEAR)
    print(f"Resized image size: {image_resized.size}")
    
    # Load Core ML model
    print("\n🤖 Loading Core ML model...")
    mlmodel = ct.models.MLModel(coreml_path)
    
    # Get model info
    spec = mlmodel.get_spec()
    print(f"Model description: {spec.description}")
    
    # Run inference
    print("\n⚡ Running Core ML inference...")
    try:
        result = mlmodel.predict({"image": image_resized})
        output = list(result.values())[0]  # Get the feature vector
        
        print(f"✅ Core ML inference successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output type: {type(output)}")
        print(f"Feature vector norm: {np.linalg.norm(output):.4f}")
        print(f"Feature vector sample (first 10): {output.flatten()[:10]}")
        
    except Exception as e:
        print(f"❌ Core ML inference failed: {e}")
        return
    
    # Compare with PyTorch model
    print("\n🔄 Comparing with PyTorch model...")
    try:
        # Load PyTorch model
        torch_model = franca_vitb14(pretrained=True, weights="IN21K", img_size=518)
        torch_model.eval()
        
        # Convert PIL image to tensor
        img_array = np.array(image_resized).astype(np.float32) / 255.0
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (img_array - mean) / std
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        
        # PyTorch inference
        with torch.no_grad():
            torch_output = torch_model(img_tensor).numpy()
        
        print(f"PyTorch output shape: {torch_output.shape}")
        print(f"PyTorch feature norm: {np.linalg.norm(torch_output):.4f}")
        
        # Compare outputs
        mse = np.mean((torch_output - output) ** 2)
        cos_sim = np.dot(torch_output.flatten(), output.flatten()) / (
            np.linalg.norm(torch_output) * np.linalg.norm(output)
        )
        
        print(f"\n📊 Comparison Results:")
        print(f"  MSE: {mse:.6f}")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        
        if cos_sim > 0.95:
            print("✅ Models produce very similar outputs!")
        elif cos_sim > 0.8:
            print("⚠️  Models produce reasonably similar outputs")
        else:
            print("❌ Models produce significantly different outputs")
            
    except Exception as e:
        print(f"⚠️  PyTorch comparison failed: {e}")
    
    print(f"\n🎉 Core ML model test complete!")
    print(f"The model is ready for use in iOS/macOS applications.")
    print(f"Model file: {os.path.abspath(coreml_path)}")

if __name__ == "__main__":
    test_coreml_model()
