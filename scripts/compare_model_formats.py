#!/usr/bin/env python3
"""
Compare different Core ML model formats: FP32 vs FP16 vs .mlprogram
Explores model size, performance, and accuracy trade-offs
"""

import os
import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import coremltools as ct
import torch
from franca.hub.backbones import franca_vitb14

class ModelFormatComparator:
    """Compare different Core ML model formats"""
    
    def __init__(self, base_model_name="franca_vitb14_in21k"):
        self.base_name = base_model_name
        self.models = {}
        self.results = {}
        
    def load_models(self, model_dir="coreml_models"):
        """Load all available model formats"""
        model_dir = Path(model_dir)
        
        # Look for different formats
        formats = {
            'fp32': f"{self.base_name}_fp32.mlpackage",
            'fp16': f"{self.base_name}_fp16.mlpackage",
            'mlprogram_fp32': f"{self.base_name}_mlprogram_fp32.mlpackage",
            'compressed_fp16': f"{self.base_name}_compressed_fp16.mlpackage",
            'palettized': f"{self.base_name}_palettized.mlpackage",
            'neuralnetwork': f"{self.base_name}_neuralnetwork.mlpackage"
        }
        
        for format_name, filename in formats.items():
            model_path = model_dir / filename
            if model_path.exists():
                try:
                    model = ct.models.MLModel(str(model_path))
                    self.models[format_name] = {
                        'model': model,
                        'path': model_path,
                        'size_mb': self._get_model_size(model_path)
                    }
                    print(f"✅ Loaded {format_name}: {filename} ({self.models[format_name]['size_mb']:.1f} MB)")
                except Exception as e:
                    print(f"❌ Failed to load {format_name}: {e}")
            else:
                print(f"⚠️  {format_name} not found: {filename}")
    
    def _get_model_size(self, model_path):
        """Calculate model size in MB"""
        if model_path.is_dir():
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        else:
            total_size = model_path.stat().st_size
        return total_size / (1024 * 1024)
    
    def benchmark_performance(self, test_image_path="assets/dog.jpg", num_runs=10):
        """Benchmark performance across all model formats"""
        if not os.path.exists(test_image_path):
            # Create dummy image
            test_image = Image.new('RGB', (518, 518), color='red')
        else:
            test_image = Image.open(test_image_path).convert('RGB')
            test_image = test_image.resize((518, 518), Image.Resampling.BILINEAR)
        
        print(f"\n⚡ Benchmarking performance ({num_runs} runs)...")
        
        for format_name, model_info in self.models.items():
            model = model_info['model']
            
            # Warmup
            for _ in range(3):
                _ = model.predict({"image": test_image})
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.time()
                result = model.predict({"image": test_image})
                times.append(time.time() - start)
            
            # Store results
            self.results[format_name] = {
                'times': times,
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'size_mb': model_info['size_mb'],
                'last_output': list(result.values())[0]
            }
            
            print(f"  {format_name:15}: {np.mean(times)*1000:6.1f} ± {np.std(times)*1000:4.1f} ms")
    
    def compare_accuracy(self):
        """Compare accuracy between different formats"""
        if len(self.results) < 2:
            print("⚠️  Need at least 2 models to compare accuracy")
            return
        
        print(f"\n🎯 Accuracy Comparison:")
        
        # Use FP32 as reference if available, otherwise first model
        reference_key = 'fp32' if 'fp32' in self.results else list(self.results.keys())[0]
        reference_output = self.results[reference_key]['last_output'].flatten()
        
        print(f"  Reference: {reference_key}")
        
        for format_name, result in self.results.items():
            if format_name == reference_key:
                continue
                
            output = result['last_output'].flatten()
            
            # Calculate metrics
            mse = np.mean((reference_output - output) ** 2)
            cos_sim = np.dot(reference_output, output) / (
                np.linalg.norm(reference_output) * np.linalg.norm(output)
            )
            max_diff = np.max(np.abs(reference_output - output))
            
            print(f"  {format_name:15}: cos_sim={cos_sim:.6f}, mse={mse:.6f}, max_diff={max_diff:.6f}")
    
    def generate_summary(self):
        """Generate comprehensive comparison summary"""
        if not self.results:
            print("❌ No results to summarize")
            return
        
        print(f"\n📊 Model Format Comparison Summary:")
        print(f"{'Format':<15} {'Size (MB)':<10} {'Time (ms)':<12} {'Speedup':<8} {'Efficiency':<12}")
        print("-" * 70)
        
        # Calculate speedup relative to slowest model
        slowest_time = max(r['mean_time'] for r in self.results.values())
        
        for format_name, result in self.results.items():
            speedup = slowest_time / result['mean_time']
            efficiency = speedup / result['size_mb']  # Performance per MB
            
            print(f"{format_name:<15} {result['size_mb']:<10.1f} "
                  f"{result['mean_time']*1000:<12.1f} {speedup:<8.2f} {efficiency:<12.4f}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        # Best performance
        fastest = min(self.results.items(), key=lambda x: x[1]['mean_time'])
        print(f"  🚀 Fastest: {fastest[0]} ({fastest[1]['mean_time']*1000:.1f} ms)")
        
        # Smallest size
        smallest = min(self.results.items(), key=lambda x: x[1]['size_mb'])
        print(f"  📱 Smallest: {smallest[0]} ({smallest[1]['size_mb']:.1f} MB)")
        
        # Best efficiency (performance per MB)
        if len(self.results) > 1:
            best_efficiency = max(self.results.items(), 
                                key=lambda x: (slowest_time / x[1]['mean_time']) / x[1]['size_mb'])
            efficiency_score = (slowest_time / best_efficiency[1]['mean_time']) / best_efficiency[1]['size_mb']
            print(f"  ⚖️  Most Efficient: {best_efficiency[0]} (score: {efficiency_score:.4f})")

def create_mlprogram_models(input_model_path="coreml_models/franca_vitb14_in21k_fp32.mlpackage",
                           output_dir="coreml_models"):
    """Create ML Program versions and proper FP16 models"""
    print(f"\n🔧 Creating ML Program models and proper FP16...")

    if not os.path.exists(input_model_path):
        print(f"❌ Input model not found: {input_model_path}")
        return

    # Load the original model
    model = ct.models.MLModel(input_model_path)

    # Create ML Program FP32 (.mlpackage with ML Program format)
    fp32_mlprogram_path = os.path.join(output_dir, "franca_vitb14_in21k_mlprogram_fp32.mlpackage")
    try:
        # Convert to ML Program format (still uses .mlpackage extension)
        model.save(fp32_mlprogram_path)
        print(f"✅ Created ML Program FP32: {fp32_mlprogram_path}")
    except Exception as e:
        print(f"❌ Failed to create ML Program FP32: {e}")

    # Create proper FP16 model with compression
    fp16_compressed_path = os.path.join(output_dir, "franca_vitb14_in21k_compressed_fp16.mlpackage")
    try:
        # Use modern compression API with correct dtype
        import coremltools.optimize.coreml as cto

        # Create compression config with proper numpy dtype
        op_config = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype=np.float16)
        config = cto.OptimizationConfig(global_config=op_config)

        # Apply compression
        compressed_model = cto.linear_quantize_weights(model, config=config)
        compressed_model.save(fp16_compressed_path)
        print(f"✅ Created compressed FP16: {fp16_compressed_path}")

    except Exception as e:
        print(f"❌ Failed to create compressed FP16: {e}")

        # Alternative: Try palettization for size reduction
        try:
            import coremltools.optimize.coreml as cto

            # Use palettization as alternative compression
            palettize_config = cto.OptimizationConfig(
                global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=4)
            )

            palettized_model = cto.palettize_weights(model, config=palettize_config)
            palettized_path = os.path.join(output_dir, "franca_vitb14_in21k_palettized.mlpackage")
            palettized_model.save(palettized_path)
            print(f"✅ Created palettized model: {palettized_path}")

        except Exception as e2:
            print(f"❌ Palettization also failed: {e2}")

    # Create Neural Network format model (legacy format)
    try:
        # Convert to Neural Network format for comparison
        nn_path = os.path.join(output_dir, "franca_vitb14_in21k_neuralnetwork.mlpackage")

        # Note: This would require re-conversion from PyTorch with different target
        print(f"ℹ️  Neural Network format requires re-conversion from PyTorch")

    except Exception as e:
        print(f"❌ Neural Network format creation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Compare Core ML model formats")
    parser.add_argument("--create-mlprogram", action="store_true", 
                       help="Create .mlprogram versions of existing models")
    parser.add_argument("--model-dir", default="coreml_models",
                       help="Directory containing models")
    parser.add_argument("--test-image", default="assets/dog.jpg",
                       help="Test image path")
    parser.add_argument("--runs", type=int, default=10,
                       help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Create .mlprogram models if requested
    if args.create_mlprogram:
        create_mlprogram_models(output_dir=args.model_dir)
        print()
    
    # Compare all available models
    comparator = ModelFormatComparator()
    comparator.load_models(args.model_dir)
    
    if not comparator.models:
        print("❌ No models found to compare")
        return
    
    # Run benchmarks
    comparator.benchmark_performance(args.test_image, args.runs)
    comparator.compare_accuracy()
    comparator.generate_summary()
    
    print(f"\n🎉 Model format comparison complete!")

if __name__ == "__main__":
    main()
