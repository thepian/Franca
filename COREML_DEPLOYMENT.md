# Franca Core ML Deployment Guide

Complete guide for deploying Franca models on iOS and macOS using Core ML.

## 🚀 Quick Start

### 1. Export Core ML Model
```bash
# Export Franca ViT-B/14 to Core ML
uv run python scripts/export_coreml_hub.py

# Compare different formats
uv run python scripts/compare_model_formats.py
```

### 2. Load in iOS/macOS App
```swift
import CoreML

let modelURL = Bundle.main.url(forResource: "franca_vitb14_in21k_fp16", 
                              withExtension: "mlpackage")!
let model = try await MLModel.load(contentsOf: modelURL)

// Predict
let input = try MLDictionaryFeatureProvider(dictionary: [
    "image": MLFeatureValue(cgImage: yourImage)
])
let output = try await model.prediction(from: input)
let features = output.featureValue(for: "var_625")?.multiArrayValue
```

## 📊 Performance Benchmarks

Based on evaluation with Franca ViT-B/14 on Apple Silicon:

| Format | Size | Inference Time | Speedup | Accuracy |
|--------|------|---------------|---------|----------|
| **FP32** | 165MB | 223.8ms | 1.00x | Baseline |
| **FP16** | 165MB | 188.2ms | **1.19x** | 99.99%+ |
| **ML Program** | 165MB | 190.4ms | 1.18x | 100% |

**Key Finding**: FP16 provides 19% faster inference with virtually no accuracy loss.

## 🎯 Model Formats

### Available Models
- `franca_vitb14_in21k_fp32.mlpackage` - Full precision (development)
- `franca_vitb14_in21k_fp16.mlpackage` - Half precision (production)
- `franca_vitb14_in21k_mlprogram_fp32.mlpackage` - ML Program format

### Format Comparison
- **FP32**: Highest accuracy, baseline performance
- **FP16**: 19% faster, minimal accuracy loss, **recommended for production**
- **ML Program**: Modern format, iOS 15+/macOS 12+, better optimization

## 📱 Integration Examples

### SwiftUI Integration
```swift
import SwiftUI
import CoreML
import Vision

struct FrancaImageClassifier: View {
    @State private var selectedImage: UIImage?
    @State private var features: [Float] = []
    @State private var isProcessing = false
    
    private let model = try! MLModel(contentsOf: Bundle.main.url(
        forResource: "franca_vitb14_in21k_fp16", 
        withExtension: "mlpackage"
    )!)
    
    var body: some View {
        VStack {
            if let image = selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 300)
            }
            
            Button("Select Image") {
                // Image picker logic
            }
            
            Button("Extract Features") {
                extractFeatures()
            }
            .disabled(selectedImage == nil || isProcessing)
            
            if !features.isEmpty {
                Text("Features extracted: \(features.count) dimensions")
                Text("Feature norm: \(sqrt(features.map { $0 * $0 }.reduce(0, +)))")
            }
        }
    }
    
    private func extractFeatures() {
        guard let image = selectedImage else { return }
        
        isProcessing = true
        
        Task {
            do {
                let input = try MLDictionaryFeatureProvider(dictionary: [
                    "image": MLFeatureValue(cgImage: image.cgImage!)
                ])
                
                let output = try await model.prediction(from: input)
                let featureArray = output.featureValue(for: "var_625")?.multiArrayValue
                
                await MainActor.run {
                    if let array = featureArray {
                        features = (0..<array.count).map { 
                            array[[$0]].floatValue 
                        }
                    }
                    isProcessing = false
                }
            } catch {
                print("Prediction error: \(error)")
                await MainActor.run {
                    isProcessing = false
                }
            }
        }
    }
}
```

### UIKit Integration
```swift
import UIKit
import CoreML

class FrancaViewController: UIViewController {
    private var model: MLModel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        loadModel()
    }
    
    private func loadModel() {
        guard let modelURL = Bundle.main.url(
            forResource: "franca_vitb14_in21k_fp16", 
            withExtension: "mlpackage"
        ) else {
            fatalError("Model not found")
        }
        
        Task {
            do {
                let config = MLModelConfiguration()
                config.computeUnits = .all
                self.model = try await MLModel.load(
                    contentsOf: modelURL, 
                    configuration: config
                )
            } catch {
                print("Failed to load model: \(error)")
            }
        }
    }
    
    func processImage(_ image: UIImage) async -> [Float]? {
        guard let model = model,
              let cgImage = image.cgImage else { return nil }
        
        do {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(cgImage: cgImage)
            ])
            
            let output = try await model.prediction(from: input)
            let features = output.featureValue(for: "var_625")?.multiArrayValue
            
            return features?.toFloatArray()
        } catch {
            print("Prediction failed: \(error)")
            return nil
        }
    }
}

extension MLMultiArray {
    func toFloatArray() -> [Float] {
        return (0..<count).map { self[[$0]].floatValue }
    }
}
```

## 🔧 Advanced Usage

### Batch Processing
```swift
class FrancaBatchProcessor {
    private let model: MLModel
    private let queue = DispatchQueue(label: "franca.processing", qos: .userInitiated)
    
    init() async throws {
        let modelURL = Bundle.main.url(
            forResource: "franca_vitb14_in21k_fp16", 
            withExtension: "mlpackage"
        )!
        
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try await MLModel.load(contentsOf: modelURL, configuration: config)
    }
    
    func processImages(_ images: [UIImage]) async -> [[Float]] {
        return await withTaskGroup(of: (Int, [Float]?).self) { group in
            for (index, image) in images.enumerated() {
                group.addTask {
                    let features = await self.extractFeatures(from: image)
                    return (index, features)
                }
            }
            
            var results: [[Float]] = Array(repeating: [], count: images.count)
            for await (index, features) in group {
                if let features = features {
                    results[index] = features
                }
            }
            return results
        }
    }
    
    private func extractFeatures(from image: UIImage) async -> [Float]? {
        // Implementation similar to previous examples
        return nil
    }
}
```

### Performance Monitoring
```swift
class FrancaPerformanceMonitor {
    private var inferenceTimes: [TimeInterval] = []
    private var memoryUsage: [UInt64] = []
    
    func measureInference<T>(_ operation: () async throws -> T) async rethrows -> T {
        let startTime = CFAbsoluteTimeGetCurrent()
        let startMemory = getMemoryUsage()
        
        let result = try await operation()
        
        let endTime = CFAbsoluteTimeGetCurrent()
        let endMemory = getMemoryUsage()
        
        inferenceTimes.append(endTime - startTime)
        memoryUsage.append(endMemory - startMemory)
        
        return result
    }
    
    func getStats() -> (avgTime: TimeInterval, avgMemory: UInt64) {
        let avgTime = inferenceTimes.reduce(0, +) / Double(inferenceTimes.count)
        let avgMemory = memoryUsage.reduce(0, +) / UInt64(memoryUsage.count)
        return (avgTime, avgMemory)
    }
    
    private func getMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        return kerr == KERN_SUCCESS ? info.resident_size : 0
    }
}
```

## 📦 Distribution Strategies

### 1. App Bundle (Simple)
- Add `.mlpackage` to Xcode project
- 165MB added to app size
- Immediate availability

### 2. On-Demand Resources (iOS)
```swift
let request = NSBundleResourceRequest(tags: ["franca-model"])
request.beginAccessingResources { error in
    if error == nil {
        // Model is now available
        loadModel()
    }
}
```

### 3. Custom Server (Flexible)
```swift
func downloadModel() async throws {
    let url = URL(string: "https://your-server.com/models/franca_v1.mlpackage.zip")!
    let (data, _) = try await URLSession.shared.data(from: url)
    
    // Verify checksum
    let checksum = data.sha256
    guard checksum == expectedChecksum else {
        throw ModelError.checksumMismatch
    }
    
    // Extract and cache
    try extractAndCacheModel(data)
}
```

## 🎯 Best Practices

### Performance
- Use FP16 for 19% speed improvement
- Enable all compute units (`.all`)
- Warm up model with dummy inference
- Process images in background queue

### Memory Management
- Load model once, reuse for multiple predictions
- Use autoreleasepool for batch processing
- Monitor memory usage in production

### Error Handling
```swift
enum FrancaError: Error {
    case modelNotFound
    case loadingFailed(Error)
    case predictionFailed(Error)
    case invalidInput
}

func safePredict(image: UIImage) async -> Result<[Float], FrancaError> {
    do {
        guard let features = await extractFeatures(from: image) else {
            return .failure(.invalidInput)
        }
        return .success(features)
    } catch {
        return .failure(.predictionFailed(error))
    }
}
```

## 📚 Additional Resources

- [Core ML Formats Guide](COREML_FORMATS_GUIDE.md) - Detailed format comparison
- [Evaluation README](EVALUATION_README.md) - Performance benchmarking tools
- [Model Card](model_card.md) - Model specifications and capabilities
- [Franca Paper](https://arxiv.org/abs/2507.14137) - Technical details

## 🔍 Troubleshooting

### Common Issues
1. **Model not found**: Ensure `.mlpackage` is in app bundle
2. **Slow inference**: Use FP16 format and enable all compute units
3. **Memory issues**: Process images individually, use autoreleasepool
4. **iOS 14 compatibility**: Use Neural Network format instead of ML Program

### Performance Tips
- Resize images to 518x518 for optimal performance
- Use batch processing for multiple images
- Cache model instance, don't reload for each prediction
- Monitor performance metrics in production

The FP16 Core ML model provides the best balance of performance, accuracy, and compatibility for production iOS/macOS deployment.
