#!/usr/bin/env python3
"""
Comprehensive evaluation script for Franca Core ML models
Can be run from command line or imported in notebooks
"""

import os
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from PIL import Image
import coremltools as ct
from franca.hub.backbones import franca_vitb14, franca_vitl14

class FrancaCoreMLEvaluator:
    """Comprehensive evaluator for Franca Core ML models"""
    
    def __init__(self, coreml_path: str, model_name: str = "vitb14", weights: str = "IN21K"):
        self.coreml_path = coreml_path
        self.model_name = model_name
        self.weights = weights
        self.results = {}
        
        # Load Core ML model
        print(f"Loading Core ML model: {coreml_path}")
        self.coreml_model = ct.models.MLModel(coreml_path)
        
        # Load PyTorch reference model
        print(f"Loading PyTorch reference model: {model_name}")
        if model_name == "vitb14":
            self.torch_model = franca_vitb14(pretrained=True, weights=weights, img_size=518)
        elif model_name == "vitl14":
            self.torch_model = franca_vitl14(pretrained=True, weights=weights, img_size=518)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.torch_model.eval()
        
    def preprocess_image_torch(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for PyTorch model"""
        # Resize to 518x518
        image_resized = image.resize((518, 518), Image.Resampling.BILINEAR)
        
        # Convert to tensor and normalize
        img_array = np.array(image_resized).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (img_array - mean) / std
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
        
        return img_tensor
    
    def preprocess_image_coreml(self, image: Image.Image) -> Image.Image:
        """Preprocess image for Core ML model"""
        return image.resize((518, 518), Image.Resampling.BILINEAR)
    
    def evaluate_single_image(self, image_path: str) -> Dict:
        """Evaluate a single image with both models"""
        print(f"Evaluating: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # PyTorch inference
        torch_input = self.preprocess_image_torch(image)
        torch_start = time.time()
        with torch.no_grad():
            torch_output = self.torch_model(torch_input).numpy()
        torch_time = time.time() - torch_start
        
        # Core ML inference
        coreml_input = self.preprocess_image_coreml(image)
        coreml_start = time.time()
        coreml_result = self.coreml_model.predict({"image": coreml_input})
        coreml_output = list(coreml_result.values())[0]
        coreml_time = time.time() - coreml_start
        
        # Compute metrics
        mse = np.mean((torch_output - coreml_output) ** 2)
        mae = np.mean(np.abs(torch_output - coreml_output))
        max_diff = np.max(np.abs(torch_output - coreml_output))
        
        # Cosine similarity
        torch_flat = torch_output.flatten()
        coreml_flat = coreml_output.flatten()
        cos_sim = np.dot(torch_flat, coreml_flat) / (
            np.linalg.norm(torch_flat) * np.linalg.norm(coreml_flat)
        )
        
        # Feature statistics
        torch_norm = np.linalg.norm(torch_output)
        coreml_norm = np.linalg.norm(coreml_output)
        
        return {
            'image_path': image_path,
            'original_size': original_size,
            'torch_time': torch_time,
            'coreml_time': coreml_time,
            'speedup': torch_time / coreml_time,
            'mse': mse,
            'mae': mae,
            'max_diff': max_diff,
            'cosine_similarity': cos_sim,
            'torch_norm': torch_norm,
            'coreml_norm': coreml_norm,
            'torch_output_shape': torch_output.shape,
            'coreml_output_shape': coreml_output.shape,
        }
    
    def evaluate_directory(self, image_dir: str, max_images: Optional[int] = None) -> Dict:
        """Evaluate all images in a directory"""
        image_dir = Path(image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"Found {len(image_paths)} images in {image_dir}")
        
        results = []
        for i, image_path in enumerate(image_paths):
            try:
                result = self.evaluate_single_image(str(image_path))
                results.append(result)
                print(f"  [{i+1}/{len(image_paths)}] ✅ {image_path.name}")
            except Exception as e:
                print(f"  [{i+1}/{len(image_paths)}] ❌ {image_path.name}: {e}")
        
        return self.compute_summary_stats(results)
    
    def compute_summary_stats(self, results: List[Dict]) -> Dict:
        """Compute summary statistics from individual results"""
        if not results:
            return {"error": "No successful evaluations"}
        
        metrics = ['torch_time', 'coreml_time', 'speedup', 'mse', 'mae', 'max_diff', 
                  'cosine_similarity', 'torch_norm', 'coreml_norm']
        
        summary = {
            'num_images': len(results),
            'individual_results': results
        }
        
        for metric in metrics:
            values = [r[metric] for r in results]
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
            summary[f'{metric}_min'] = np.min(values)
            summary[f'{metric}_max'] = np.max(values)
            summary[f'{metric}_median'] = np.median(values)
        
        return summary
    
    def benchmark_performance(self, num_runs: int = 10) -> Dict:
        """Benchmark inference performance"""
        print(f"Benchmarking performance with {num_runs} runs...")
        
        # Create dummy input
        dummy_image = Image.new('RGB', (518, 518), color='red')
        torch_input = self.preprocess_image_torch(dummy_image).float()
        coreml_input = self.preprocess_image_coreml(dummy_image)
        
        # Warmup
        print("Warming up models...")
        for _ in range(3):
            with torch.no_grad():
                _ = self.torch_model(torch_input)
            _ = self.coreml_model.predict({"image": coreml_input})
        
        # Benchmark PyTorch
        torch_times = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = self.torch_model(torch_input)
            torch_times.append(time.time() - start)
        
        # Benchmark Core ML
        coreml_times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.coreml_model.predict({"image": coreml_input})
            coreml_times.append(time.time() - start)
        
        return {
            'num_runs': num_runs,
            'torch_mean_time': np.mean(torch_times),
            'torch_std_time': np.std(torch_times),
            'coreml_mean_time': np.mean(coreml_times),
            'coreml_std_time': np.std(coreml_times),
            'speedup_mean': np.mean(torch_times) / np.mean(coreml_times),
            'torch_times': torch_times,
            'coreml_times': coreml_times
        }
    
    def generate_report(self, output_path: str = "evaluation_report.json"):
        """Generate comprehensive evaluation report"""
        print("Generating comprehensive evaluation report...")
        
        report = {
            'model_info': {
                'coreml_path': self.coreml_path,
                'model_name': self.model_name,
                'weights': self.weights,
                'coreml_spec': str(self.coreml_model.get_spec())[:500] + "..."
            },
            'performance_benchmark': self.benchmark_performance(),
        }
        
        # Test with sample image if available
        sample_image = "assets/dog.jpg"
        if os.path.exists(sample_image):
            print(f"Testing with sample image: {sample_image}")
            report['sample_image_test'] = self.evaluate_single_image(sample_image)
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to: {output_path}")
        return report

def main():
    parser = argparse.ArgumentParser(description="Evaluate Franca Core ML models")
    parser.add_argument("--model", default="coreml_models/franca_vitb14_in21k_fp32.mlpackage",
                       help="Path to Core ML model")
    parser.add_argument("--model-name", default="vitb14", choices=["vitb14", "vitl14"],
                       help="Model architecture name")
    parser.add_argument("--weights", default="IN21K", choices=["IN21K", "LAION", "DINOV2_IN21K"],
                       help="Model weights")
    parser.add_argument("--image", help="Single image to evaluate")
    parser.add_argument("--image-dir", help="Directory of images to evaluate")
    parser.add_argument("--max-images", type=int, help="Maximum number of images to evaluate")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--output", default="evaluation_report.json", help="Output report path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("Run export_coreml_hub.py first to create the model")
        return
    
    # Create evaluator
    evaluator = FrancaCoreMLEvaluator(args.model, args.model_name, args.weights)
    
    # Run evaluations
    if args.image:
        result = evaluator.evaluate_single_image(args.image)
        print("\n📊 Single Image Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    elif args.image_dir:
        results = evaluator.evaluate_directory(args.image_dir, args.max_images)
        print("\n📊 Directory Evaluation Results:")
        print(f"  Images processed: {results['num_images']}")
        print(f"  Average cosine similarity: {results['cosine_similarity_mean']:.4f} ± {results['cosine_similarity_std']:.4f}")
        print(f"  Average speedup: {results['speedup_mean']:.2f}x ± {results['speedup_std']:.2f}x")
    
    elif args.benchmark:
        results = evaluator.benchmark_performance()
        print("\n⚡ Performance Benchmark Results:")
        print(f"  PyTorch: {results['torch_mean_time']*1000:.1f} ± {results['torch_std_time']*1000:.1f} ms")
        print(f"  Core ML: {results['coreml_mean_time']*1000:.1f} ± {results['coreml_std_time']*1000:.1f} ms")
        print(f"  Speedup: {results['speedup_mean']:.2f}x")
    
    else:
        # Generate comprehensive report
        report = evaluator.generate_report(args.output)
        print("\n📋 Comprehensive Report Generated")
        print(f"  Performance: {report['performance_benchmark']['speedup_mean']:.2f}x speedup")
        if 'sample_image_test' in report:
            print(f"  Sample test: {report['sample_image_test']['cosine_similarity']:.4f} cosine similarity")

if __name__ == "__main__":
    main()
