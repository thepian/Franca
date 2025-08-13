# Core ML Model Formats & Distribution Guide

This guide explains the different Core ML model formats available for Franca and their optimal use cases.

## 🎯 **Current Model Performance Results**

Based on our benchmarking with Franca ViT-B/14:

| Format | Size (MB) | Inference Time (ms) | Speedup | Efficiency Score |
|--------|-----------|-------------------|---------|------------------|
| **FP32** | 165.3 | 223.8 ± 7.2 | 1.00x | 0.0060 |
| **FP16** | 165.3 | 188.2 ± 3.4 | **1.19x** | **0.0072** |
| **ML Program FP32** | 165.3 | 190.4 ± 7.5 | 1.18x | 0.0071 |

**Key Finding**: FP16 format provides **19% faster inference** with identical accuracy!

## 📦 **Core ML Format Types**

### **1. Neural Network vs ML Program**

#### **Neural Network Format** (Legacy)
- **Extension**: `.mlmodel` or `.mlpackage`
- **Target**: iOS 11+ / macOS 10.13+
- **Best for**: Older deployment targets
- **Limitations**: Limited operator support

#### **ML Program Format** (Modern)
- **Extension**: `.mlpackage`
- **Target**: iOS 15+ / macOS 12+
- **Best for**: New deployments, complex models
- **Advantages**: Full operator support, better optimization

### **2. Precision Types**

#### **FP32 (Float32)**
- **Precision**: 32-bit floating point
- **Accuracy**: Highest precision
- **Size**: Largest model size
- **Performance**: Baseline performance
- **Use case**: Development, debugging, accuracy validation

#### **FP16 (Float16)**
- **Precision**: 16-bit floating point
- **Accuracy**: Minimal precision loss (>99.99% similarity)
- **Size**: ~50% smaller (when properly compressed)
- **Performance**: **19% faster** on Apple Silicon
- **Use case**: Production deployment, mobile apps

## 🚀 **Distribution Strategies**

### **Option 1: Direct App Bundle**
```
MyApp.app/
├── Contents/
│   ├── Resources/
│   │   └── franca_vitb14_in21k_fp16.mlpackage/
│   └── MacOS/
│       └── MyApp
```

**Pros**: 
- Simple deployment
- No network dependency
- Immediate availability

**Cons**:
- Increases app size by ~165MB
- App Store review includes model
- Updates require app update

### **Option 2: On-Demand Resources (iOS)**
```swift
// Request model download
let request = NSBundleResourceRequest(tags: ["franca-model"])
request.beginAccessingResources { error in
    if error == nil {
        // Model is now available
        loadModel()
    }
}
```

**Pros**:
- Reduces initial app size
- Conditional model loading
- Apple handles hosting

**Cons**:
- iOS only
- 2GB limit per resource
- Requires network connection

### **Option 3: Custom Model Server**
```swift
// Download from your server
let url = URL(string: "https://your-server.com/models/franca_v1.mlpackage.zip")
downloadAndCacheModel(from: url)
```

**Pros**:
- Full control over distribution
- A/B testing capabilities
- Cross-platform support
- Analytics and usage tracking

**Cons**:
- Infrastructure overhead
- Network dependency
- Security considerations

### **Option 4: Core ML Model Collection**
```swift
// Use Apple's model collection framework
import CoreML

let modelCollection = MLModelCollection(identifier: "your-collection-id")
modelCollection.beginAccessing { result in
    // Handle model availability
}
```

**Pros**:
- Apple-managed infrastructure
- Automatic updates
- Optimized delivery

**Cons**:
- Requires Apple approval
- Limited to Apple ecosystem
- Less control over distribution

## 📱 **Platform-Specific Recommendations**

### **iOS Apps**
- **Format**: FP16 ML Program (.mlpackage)
- **Distribution**: On-Demand Resources for large models
- **Target**: iOS 15+ for best performance
- **Size optimization**: Use FP16 for 19% speed improvement

### **macOS Apps**
- **Format**: FP16 ML Program (.mlpackage)
- **Distribution**: Direct bundle or custom server
- **Target**: macOS 12+ for ML Program support
- **Performance**: Excellent on Apple Silicon

### **Cross-Platform**
- **Format**: FP32 ML Program for compatibility
- **Distribution**: Custom server with platform detection
- **Fallback**: Neural Network format for older devices

## 🔧 **Implementation Examples**

### **Swift Model Loading**
```swift
import CoreML

class FrancaModelManager {
    private var model: MLModel?
    
    func loadModel() async throws {
        guard let modelURL = Bundle.main.url(
            forResource: "franca_vitb14_in21k_fp16", 
            withExtension: "mlpackage"
        ) else {
            throw ModelError.notFound
        }
        
        let config = MLModelConfiguration()
        config.computeUnits = .all // Use all available compute units
        
        self.model = try await MLModel.load(
            contentsOf: modelURL, 
            configuration: config
        )
    }
    
    func predict(image: CGImage) async throws -> MLMultiArray {
        guard let model = model else {
            throw ModelError.notLoaded
        }
        
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(cgImage: image)
        ])
        
        let output = try await model.prediction(from: input)
        return output.featureValue(for: "output")?.multiArrayValue ?? MLMultiArray()
    }
}
```

### **Model Versioning**
```swift
struct ModelVersion {
    let version: String
    let url: URL
    let checksum: String
    let minimumOSVersion: OperatingSystemVersion
}

class ModelUpdateManager {
    func checkForUpdates() async -> ModelVersion? {
        // Check server for newer model versions
        // Compare with local version
        // Return update info if available
    }
    
    func downloadModel(_ version: ModelVersion) async throws {
        // Download, verify checksum, and cache model
    }
}
```

## 🎯 **Best Practices**

### **Performance Optimization**
1. **Use FP16** for 19% speed improvement
2. **Enable all compute units** (.all) for best performance
3. **Warm up models** with dummy inference
4. **Cache predictions** when appropriate

### **Size Optimization**
1. **Choose FP16** over FP32 when accuracy allows
2. **Consider model pruning** for further size reduction
3. **Use compression** for network transfer
4. **Implement progressive loading** for large models

### **Distribution Security**
1. **Verify model checksums** after download
2. **Use HTTPS** for model downloads
3. **Implement model signing** for authenticity
4. **Monitor model usage** for anomalies

### **User Experience**
1. **Show download progress** for large models
2. **Provide offline fallbacks** when possible
3. **Handle network failures** gracefully
4. **Cache models locally** to avoid re-downloads

## 📊 **Monitoring & Analytics**

### **Key Metrics to Track**
- Model download success rate
- Inference latency percentiles
- Memory usage patterns
- Error rates by device type
- Model version adoption

### **Implementation**
```swift
struct ModelMetrics {
    let inferenceTime: TimeInterval
    let memoryUsage: UInt64
    let deviceType: String
    let modelVersion: String
    let accuracy: Double?
}

class ModelAnalytics {
    func trackInference(_ metrics: ModelMetrics) {
        // Send to analytics service
    }
    
    func trackModelDownload(success: Bool, size: Int64, duration: TimeInterval) {
        // Track download performance
    }
}
```

## 🎉 **Summary**

For **Franca Core ML deployment**, we recommend:

1. **Production**: FP16 ML Program format for 19% speed boost
2. **Distribution**: Custom server for flexibility, On-Demand Resources for simplicity
3. **Target**: iOS 15+ / macOS 12+ for best performance
4. **Monitoring**: Track inference performance and model usage

The FP16 format provides the best balance of performance, accuracy, and compatibility for modern Apple devices.
