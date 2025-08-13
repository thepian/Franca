#!/usr/bin/env python3
"""
Quick evaluation script for Franca Core ML models
Provides essential metrics without heavy dependencies
"""

import os
import time
import numpy as np
from PIL import Image
import coremltools as ct

def quick_evaluate(coreml_path="coreml_models/franca_vitb14_in21k_fp32.mlpackage", 
                  test_image="assets/dog.jpg"):
    """Quick evaluation of Core ML model"""
    
    print("🚀 Franca Core ML Quick Evaluation")
    print("=" * 50)
    
    # Check files
    if not os.path.exists(coreml_path):
        print(f"❌ Model not found: {coreml_path}")
        return
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    # Load model
    print(f"📱 Loading Core ML model...")
    model = ct.models.MLModel(coreml_path)
    
    # Model info
    spec = model.get_spec()
    input_desc = spec.description.input[0]
    output_desc = spec.description.output[0]
    
    print(f"✅ Model loaded successfully")
    print(f"   Input: {input_desc.name} - {input_desc.type}")
    print(f"   Output: {output_desc.name} - {output_desc.type}")
    
    # Load and preprocess image
    print(f"\n📸 Loading test image: {test_image}")
    image = Image.open(test_image).convert('RGB')
    print(f"   Original size: {image.size}")
    
    # Resize to model input size
    image_resized = image.resize((518, 518), Image.Resampling.BILINEAR)
    print(f"   Resized to: {image_resized.size}")
    
    # Performance test
    print(f"\n⚡ Performance Testing...")
    
    # Warmup
    for _ in range(3):
        _ = model.predict({"image": image_resized})
    
    # Benchmark
    times = []
    for i in range(10):
        start = time.time()
        result = model.predict({"image": image_resized})
        times.append(time.time() - start)
        if i == 0:  # Save first result for analysis
            features = list(result.values())[0]
    
    # Results
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    
    print(f"✅ Inference Results:")
    print(f"   Average time: {avg_time:.1f} ± {std_time:.1f} ms")
    print(f"   Min time: {np.min(times)*1000:.1f} ms")
    print(f"   Max time: {np.max(times)*1000:.1f} ms")
    
    # Feature analysis
    print(f"\n🧠 Feature Analysis:")
    print(f"   Output shape: {features.shape}")
    print(f"   Output type: {features.dtype}")
    print(f"   Feature norm: {np.linalg.norm(features):.4f}")
    print(f"   Mean: {features.mean():.4f}")
    print(f"   Std: {features.std():.4f}")
    print(f"   Min: {features.min():.4f}")
    print(f"   Max: {features.max():.4f}")
    
    # Model size
    if os.path.isdir(coreml_path):
        total_size = 0
        for root, dirs, files in os.walk(coreml_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
    else:
        total_size = os.path.getsize(coreml_path)
    
    size_mb = total_size / (1024 * 1024)
    
    print(f"\n📁 Model Info:")
    print(f"   File size: {size_mb:.1f} MB")
    print(f"   Memory usage: ~{size_mb + 518*518*3*4/(1024*1024):.1f} MB")
    
    # Summary
    print(f"\n🎯 Summary:")
    print(f"   ✅ Model loads and runs successfully")
    print(f"   ⚡ Average inference: {avg_time:.1f} ms")
    print(f"   🧠 Output: {features.shape} features")
    print(f"   📱 Ready for iOS/macOS deployment")
    
    print(f"\n🎉 Quick evaluation complete!")
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'features_shape': features.shape,
        'features_norm': float(np.linalg.norm(features)),
        'model_size_mb': size_mb
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Franca Core ML evaluation")
    parser.add_argument("--model", default="coreml_models/franca_vitb14_in21k_fp32.mlpackage",
                       help="Path to Core ML model")
    parser.add_argument("--image", default="assets/dog.jpg",
                       help="Test image path")
    
    args = parser.parse_args()
    
    quick_evaluate(args.model, args.image)
