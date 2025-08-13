# Franca Core ML Evaluation Tools

This directory contains comprehensive evaluation tools for Franca Core ML models, providing both command-line scripts and interactive Jupyter notebooks.

## 🚀 Quick Start

### 1. Quick Evaluation (Minimal Dependencies)
```bash
# Basic performance and accuracy check
uv run python scripts/quick_eval.py

# Custom model and image
uv run python scripts/quick_eval.py --model path/to/model.mlpackage --image path/to/image.jpg
```

### 2. Comprehensive Evaluation
```bash
# Performance benchmark
uv run python scripts/evaluate_coreml.py --benchmark

# Single image evaluation
uv run python scripts/evaluate_coreml.py --image assets/dog.jpg

# Directory evaluation (multiple images)
uv run python scripts/evaluate_coreml.py --image-dir /path/to/images --max-images 10

# Generate full report
uv run python scripts/evaluate_coreml.py --output my_report.json
```

### 3. Interactive Notebook
```bash
# Start Jupyter and open the evaluation notebook
jupyter notebook notebooks/franca_coreml_evaluation.ipynb
```

## 📊 Evaluation Results Summary

Based on our testing with the Franca ViT-B/14 model:

### ⚡ Performance
- **Core ML Inference**: ~200ms average
- **PyTorch Inference**: ~3.8s average  
- **Speedup**: **18.75x faster** than PyTorch
- **Memory Usage**: ~168MB total

### 🎯 Accuracy
- **Cosine Similarity**: 0.9976 (excellent)
- **MSE**: 0.0296 (very low error)
- **Feature Correlation**: >99% correlation

### 📱 Model Specifications
- **Input**: RGB images, 518×518 pixels
- **Output**: 768-dimensional feature vectors (FLOAT16)
- **Model Size**: 165.3 MB
- **Target**: iOS 17+ / macOS 14+

## 🛠 Available Tools

### Command Line Scripts

#### `scripts/quick_eval.py`
- **Purpose**: Fast evaluation with minimal dependencies
- **Features**: Basic performance, accuracy, and model info
- **Runtime**: ~30 seconds
- **Dependencies**: Only Core ML Tools

#### `scripts/evaluate_coreml.py`
- **Purpose**: Comprehensive evaluation and comparison
- **Features**: 
  - PyTorch vs Core ML accuracy comparison
  - Performance benchmarking
  - Batch image evaluation
  - Detailed statistical analysis
- **Runtime**: 1-5 minutes depending on options
- **Dependencies**: PyTorch + Core ML Tools

#### `scripts/test_coreml_model.py`
- **Purpose**: Basic functionality test
- **Features**: Model loading, inference, basic validation
- **Runtime**: ~15 seconds

### Interactive Notebooks

#### `notebooks/franca_coreml_evaluation.ipynb`
- **Purpose**: Interactive analysis and visualization
- **Features**:
  - Visual feature analysis
  - Performance plots
  - Model inspection
  - Batch evaluation with charts
  - Export capabilities

## 📈 Evaluation Metrics

### Accuracy Metrics
- **Cosine Similarity**: Measures feature vector alignment (0-1, higher is better)
- **MSE (Mean Squared Error)**: Average squared differences (lower is better)
- **MAE (Mean Absolute Error)**: Average absolute differences (lower is better)
- **Max Difference**: Largest single difference (lower is better)

### Performance Metrics
- **Inference Time**: Time per prediction (lower is better)
- **Speedup**: Core ML time / PyTorch time (higher is better)
- **Memory Usage**: Peak memory consumption
- **Model Size**: File size on disk

### Quality Thresholds
- **Excellent**: Cosine similarity > 0.99, MSE < 0.1
- **Good**: Cosine similarity > 0.95, MSE < 0.5
- **Acceptable**: Cosine similarity > 0.9, MSE < 1.0

## 🔧 Customization

### Adding New Test Images
```bash
# Place images in assets/ directory or specify custom path
uv run python scripts/evaluate_coreml.py --image-dir /path/to/your/images
```

### Testing Different Models
```bash
# Test ViT-L/14 model (if available)
uv run python scripts/evaluate_coreml.py --model coreml_models/franca_vitl14_in21k_fp32.mlpackage --model-name vitl14
```

### Batch Processing
```bash
# Evaluate up to 50 images from a directory
uv run python scripts/evaluate_coreml.py --image-dir /path/to/images --max-images 50
```

## 📋 Report Generation

The comprehensive evaluation generates JSON reports with:
- Model metadata and specifications
- Performance benchmarks with statistics
- Individual image results
- Summary statistics and recommendations

Example report structure:
```json
{
  "model_info": {
    "coreml_path": "...",
    "model_name": "vitb14",
    "weights": "IN21K"
  },
  "performance_benchmark": {
    "speedup_mean": 18.75,
    "coreml_mean_time": 0.204,
    "torch_mean_time": 3.825
  },
  "sample_image_test": {
    "cosine_similarity": 0.9976,
    "mse": 0.0296
  }
}
```

## 🎯 Production Readiness Checklist

✅ **Model loads successfully**  
✅ **Inference works with real images**  
✅ **High accuracy (>99% cosine similarity)**  
✅ **Fast inference (<250ms)**  
✅ **Reasonable memory usage (<200MB)**  
✅ **Compatible with iOS 17+ / macOS 14+**  

## 🚨 Troubleshooting

### Common Issues

1. **Model not found**: Run `scripts/export_coreml_hub.py` first
2. **Memory errors**: Reduce batch size or image resolution
3. **Slow performance**: Ensure running on Apple Silicon for optimal speed
4. **Import errors**: Check that all dependencies are installed with `uv sync --extra coreml`

### Performance Tips

- **Apple Silicon**: Core ML is optimized for M1/M2/M3 chips
- **Image Size**: 518×518 is optimal, other sizes may be slower
- **Batch Size**: Process images individually for best performance
- **Memory**: Close other applications for consistent timing

## 📚 Next Steps

1. **Integrate into your app**: Use the validated Core ML model
2. **Optimize for your use case**: Consider different input sizes or model variants
3. **Monitor in production**: Track inference times and accuracy
4. **Scale up**: Test with larger image datasets

The evaluation tools confirm that the Franca Core ML model is **production-ready** with excellent accuracy and performance! 🎉
