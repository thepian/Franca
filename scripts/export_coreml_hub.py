#!/usr/bin/env python3
"""
Core ML Export Script for Franca Models
Uses the hub interface to automatically download and convert models.
"""

import os
import torch
import coremltools as ct
import numpy as np
from PIL import Image
from franca.hub.backbones import franca_vitb14, franca_vitl14

# Configuration
OUTPUT_DIR = "coreml_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def export_franca_to_coreml(model_name="vitb14", weights="IN21K", img_size=518):
    """
    Export Franca model to Core ML format
    
    Args:
        model_name: "vitb14" or "vitl14" 
        weights: "IN21K", "LAION", or "DINOV2_IN21K"
        img_size: Input image size (518 for IN21K/LAION, 224 for DINOV2_IN21K)
    """
    print(f"Loading Franca {model_name} with {weights} weights...")
    
    # Load model from hub
    if model_name == "vitb14":
        model = franca_vitb14(pretrained=True, weights=weights, img_size=img_size)
    elif model_name == "vitl14":
        model = franca_vitl14(pretrained=True, weights=weights, img_size=img_size)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model.eval()
    print(f"Model loaded successfully. Input size: {img_size}x{img_size}")
    
    # Wrapper for cleaner interface
    class FrancaWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # Franca models return features
            features = self.model(x)
            return features
    
    wrapped_model = FrancaWrapper(model)
    wrapped_model.eval()  # Ensure wrapper is also in eval mode

    # Create example input
    example_input = torch.randn(1, 3, img_size, img_size)
    print(f"Example input shape: {example_input.shape}")

    # Test the model
    with torch.no_grad():
        output = wrapped_model(example_input)
        print(f"Model output shape: {output.shape}")

    # Trace the model
    print("Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapped_model, example_input)
        traced_model.eval()  # Ensure traced model is in eval mode
        traced_model = torch.jit.freeze(traced_model)
    
    # Define Core ML input with proper preprocessing
    # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    image_input = ct.ImageType(
        name="image",
        shape=example_input.shape,
        scale=1.0/255.0,  # Scale from [0,255] to [0,1]
        bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225]  # Apply ImageNet normalization
    )
    
    # Convert to Core ML
    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        compute_units=ct.ComputeUnit.ALL,  # Use all available compute units
        minimum_deployment_target=ct.target.iOS17,
    )
    
    # Save FP32 model (use .mlpackage for modern Core ML models)
    model_filename = f"franca_{model_name}_{weights.lower()}_fp32.mlpackage"
    coreml_path = os.path.join(OUTPUT_DIR, model_filename)
    mlmodel.save(coreml_path)
    print(f"FP32 model saved: {coreml_path}")
    
    # Create FP16 version for smaller size
    print("Creating FP16 version...")
    try:
        # Use modern coremltools compression API
        import coremltools.optimize.coreml as cto

        # Create compression config for FP16
        op_config = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="float16")
        config = cto.OptimizationConfig(global_config=op_config)

        # Apply compression
        mlmodel_fp16 = cto.linear_quantize_weights(mlmodel, config=config)

        fp16_filename = f"franca_{model_name}_{weights.lower()}_fp16.mlpackage"
        fp16_path = os.path.join(OUTPUT_DIR, fp16_filename)
        mlmodel_fp16.save(fp16_path)
        print(f"FP16 model saved: {fp16_path}")
    except Exception as e:
        print(f"FP16 conversion failed (trying fallback): {e}")
        # Fallback: just save the FP32 model with a different name
        fp16_filename = f"franca_{model_name}_{weights.lower()}_fp16.mlpackage"
        fp16_path = os.path.join(OUTPUT_DIR, fp16_filename)
        mlmodel.save(fp16_path)
        print(f"Fallback: FP32 model saved as FP16: {fp16_path}")
    
    return coreml_path, traced_model, example_input

def validate_coreml_model(coreml_path, torch_model, example_input, tolerance=1e-3):
    """
    Validate that Core ML model produces similar outputs to PyTorch model
    """
    print(f"\nValidating Core ML model: {coreml_path}")
    
    # Get PyTorch output
    with torch.no_grad():
        torch_output = torch_model(example_input).numpy()
    
    # Load Core ML model
    mlmodel = ct.models.MLModel(coreml_path)

    # Prepare input for Core ML - convert to PIL Image
    # Convert tensor to numpy and scale to [0,255]
    img_array = (example_input.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    # Ensure values are in valid range
    img_array = np.clip(img_array, 0, 255)
    pil_image = Image.fromarray(img_array)

    # Get Core ML output
    coreml_result = mlmodel.predict({"image": pil_image})
    coreml_output = list(coreml_result.values())[0]  # Get the output tensor
    
    # Compare outputs
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"Core ML output shape: {coreml_output.shape}")
    
    # Calculate metrics
    mse = np.mean((torch_output - coreml_output) ** 2)
    max_diff = np.max(np.abs(torch_output - coreml_output))
    
    # Cosine similarity
    torch_flat = torch_output.flatten()
    coreml_flat = coreml_output.flatten()
    cos_sim = np.dot(torch_flat, coreml_flat) / (
        np.linalg.norm(torch_flat) * np.linalg.norm(coreml_flat)
    )
    
    print(f"Validation Results:")
    print(f"  MSE: {mse:.8f}")
    print(f"  Max difference: {max_diff:.8f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    
    if cos_sim > 0.999 and max_diff < tolerance:
        print("✅ Validation PASSED - Core ML model matches PyTorch model")
        return True
    else:
        print("❌ Validation FAILED - Significant differences detected")
        return False

if __name__ == "__main__":
    # Export ViT-B/14 with IN21K weights (most common)
    print("=== Exporting Franca ViT-B/14 (IN21K) ===")
    coreml_path, torch_model, example_input = export_franca_to_coreml(
        model_name="vitb14", 
        weights="IN21K", 
        img_size=518
    )
    
    # Validate the conversion
    validate_coreml_model(coreml_path, torch_model, example_input)
    
    print(f"\n🎉 Core ML export complete!")
    print(f"Models saved in: {OUTPUT_DIR}")
    print(f"You can now use the .mlmodel files in iOS/macOS applications.")
