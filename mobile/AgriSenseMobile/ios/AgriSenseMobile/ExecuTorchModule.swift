import Foundation
import UIKit

// ExecuTorch's iOS runtime ships as a Swift-friendly Objective-C++ framework
// (ExecuTorch.xcframework, from the `executorch` CocoaPod / SPM package).
// Import name depends on how it was integrated — this assumes CocoaPods with
// `pod 'ExecuTorch', '~> 0.4.0'`.
import ExecuTorch

private let IMG_SIZE = 64
private let CLASSES = ["healthy", "leaf_blight", "leaf_rust", "powdery_mildew"]

/// React Native bridge around the ExecuTorch iOS runtime.
/// Mirrors ExecuTorchModule.kt on the Android side: same method names,
/// same input/output shapes, so the JS layer (src/native/executorch.ts)
/// doesn't need any platform branching beyond NativeModules routing.
@objc(ExecuTorchModule)
class ExecuTorchModule: NSObject {

  private var classifierModule: Module?
  private var segmentationModule: Module?
  private var loadedQuantized: Bool? = nil

  @objc
  static func requiresMainQueueSetup() -> Bool { return false }

  private func modelPath(for assetName: String) -> String? {
    // .pte files are added to the iOS app bundle directly (drag into Xcode
    // project, "Copy Bundle Resources" build phase) rather than copied at
    // runtime like the Android assets/ approach, since iOS app bundles are
    // already a real filesystem path.
    return Bundle.main.path(forResource: (assetName as NSString).deletingPathExtension,
                             ofType: "pte")
  }

  @objc
  func preloadModels(_ useQuantized: Bool,
                      resolver resolve: @escaping RCTPromiseResolveBlock,
                      rejecter reject: @escaping RCTPromiseRejectBlock) {
    do {
      let classifierAsset = useQuantized ? "cnn_classifier_int8" : "cnn_classifier_fp32"
      let segAsset = useQuantized ? "unet_segmentation_int8" : "unet_segmentation_fp32"

      guard let classifierPath = modelPath(for: classifierAsset),
            let segPath = modelPath(for: segAsset) else {
        reject("MODEL_LOAD_FAILED", "Could not find .pte files in app bundle", nil)
        return
      }

      if loadedQuantized != useQuantized {
        classifierModule = nil
        segmentationModule = nil
        loadedQuantized = nil
      }

      if classifierModule == nil {
        classifierModule = try Module(filePath: classifierPath)
      }
      if segmentationModule == nil {
        segmentationModule = try Module(filePath: segPath)
      }
      loadedQuantized = useQuantized
      resolve(true)
    } catch {
      reject("MODEL_LOAD_FAILED", "Failed to load ExecuTorch models: \(error.localizedDescription)", error)
    }
  }

  /// Decodes base64 image -> resized 64x64 RGB Float32 buffer, CHW, [0,1].
  /// Matches the channel ordering used by PyTorch's ToTensor() at train time.
  private func imageToInputTensor(_ base64: String) throws -> Tensor {
    let cleaned = base64.contains(",") ? String(base64.split(separator: ",").last!) : base64
    guard let data = Data(base64Encoded: cleaned),
          let uiImage = UIImage(data: data),
          let cgImage = uiImage.cgImage else {
      throw NSError(domain: "ExecuTorchModule", code: 1,
                     userInfo: [NSLocalizedDescriptionKey: "Could not decode image"])
    }

    let size = CGSize(width: IMG_SIZE, height: IMG_SIZE)
    UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
    UIImage(cgImage: cgImage).draw(in: CGRect(origin: .zero, size: size))
    let resized = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()

    guard let resizedCg = resized?.cgImage else {
      throw NSError(domain: "ExecuTorchModule", code: 2,
                     userInfo: [NSLocalizedDescriptionKey: "Resize failed"])
    }

    var pixelData = [UInt8](repeating: 0, count: IMG_SIZE * IMG_SIZE * 4)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(data: &pixelData, width: IMG_SIZE, height: IMG_SIZE,
                             bitsPerComponent: 8, bytesPerRow: IMG_SIZE * 4,
                             space: colorSpace,
                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    context?.draw(resizedCg, in: CGRect(x: 0, y: 0, width: IMG_SIZE, height: IMG_SIZE))

    var chwBuffer = [Float32](repeating: 0, count: 3 * IMG_SIZE * IMG_SIZE)
    let plane = IMG_SIZE * IMG_SIZE
    for i in 0..<plane {
      let px = i * 4
      chwBuffer[i]             = Float32(pixelData[px])     / 255.0  // R
      chwBuffer[plane + i]     = Float32(pixelData[px + 1]) / 255.0  // G
      chwBuffer[2 * plane + i] = Float32(pixelData[px + 2]) / 255.0  // B
    }

    return try Tensor(data: chwBuffer, shape: [1, 3, NSNumber(value: IMG_SIZE), NSNumber(value: IMG_SIZE)])
  }

  private func softmax(_ logits: [Float32]) -> [Float32] {
    let maxVal = logits.max() ?? 0
    let exps = logits.map { expf($0 - maxVal) }
    let sum = exps.reduce(0, +)
    return exps.map { $0 / sum }
  }

  @objc
  func classifyLeaf(_ imageBase64: String,
                     useQuantized: Bool,
                     resolver resolve: @escaping RCTPromiseResolveBlock,
                     rejecter reject: @escaping RCTPromiseRejectBlock) {
    guard loadedQuantized == useQuantized else {
      reject("MODEL_MODE_MISMATCH", "Load the requested FP32/INT8 model first", nil)
      return
    }
    guard let module = classifierModule else {
      reject("MODEL_NOT_LOADED", "Call preloadModels() first", nil)
      return
    }
    do {
      let inputTensor = try imageToInputTensor(imageBase64)
      let start = Date()
      let outputs = try module.forward([inputTensor])
      let latencyMs = Date().timeIntervalSince(start) * 1000

      let logits = try outputs[0].data() as [Float32]
      let scores = softmax(logits)
      let bestIdx = scores.indices.max(by: { scores[$0] < scores[$1] }) ?? 0

      let allScores = CLASSES.indices.map { i -> [String: Any] in
        ["label": CLASSES[i], "score": Double(scores[i])]
      }

      resolve([
        "label": CLASSES[bestIdx],
        "confidence": Double(scores[bestIdx]),
        "allScores": allScores,
        "latencyMs": latencyMs
      ])
    } catch {
      reject("INFERENCE_FAILED", "Classifier inference failed: \(error.localizedDescription)", error)
    }
  }

  @objc
  func segmentLesion(_ imageBase64: String,
                      useQuantized: Bool,
                      resolver resolve: @escaping RCTPromiseResolveBlock,
                      rejecter reject: @escaping RCTPromiseRejectBlock) {
    guard loadedQuantized == useQuantized else {
      reject("MODEL_MODE_MISMATCH", "Load the requested FP32/INT8 model first", nil)
      return
    }
    guard let module = segmentationModule else {
      reject("MODEL_NOT_LOADED", "Call preloadModels() first", nil)
      return
    }
    do {
      let inputTensor = try imageToInputTensor(imageBase64)
      let start = Date()
      let outputs = try module.forward([inputTensor])
      let latencyMs = Date().timeIntervalSince(start) * 1000

      // Sigmoid already applied inside the exported graph (UNetWithSigmoid).
      let maskProbs = try outputs[0].data() as [Float32]

      var affectedPixels = 0
      var pixelData = [UInt8](repeating: 0, count: IMG_SIZE * IMG_SIZE * 4)
      for y in 0..<IMG_SIZE {
        for x in 0..<IMG_SIZE {
          let prob = maskProbs[y * IMG_SIZE + x]
          if prob > 0.5 { affectedPixels += 1 }
          let idx = (y * IMG_SIZE + x) * 4
          let alpha = UInt8(min(max(prob * 255, 0), 255))
          pixelData[idx] = 255      // R
          pixelData[idx + 1] = 59   // G  -> matches Android's 0xFF3B30 red overlay
          pixelData[idx + 2] = 48   // B
          pixelData[idx + 3] = alpha
        }
      }

      let affectedAreaPct = (Double(affectedPixels) / Double(IMG_SIZE * IMG_SIZE)) * 100.0
      let maskBase64 = pngBase64(from: pixelData, width: IMG_SIZE, height: IMG_SIZE)

      resolve([
        "maskBase64": "data:image/png;base64,\(maskBase64)",
        "affectedAreaPct": affectedAreaPct,
        "latencyMs": latencyMs
      ])
    } catch {
      reject("INFERENCE_FAILED", "Segmentation inference failed: \(error.localizedDescription)", error)
    }
  }

  private func pngBase64(from rgba: [UInt8], width: Int, height: Int) -> String {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    var data = rgba
    let context = CGContext(data: &data, width: width, height: height,
                             bitsPerComponent: 8, bytesPerRow: width * 4,
                             space: colorSpace,
                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    guard let cgImage = context?.makeImage() else { return "" }
    let uiImage = UIImage(cgImage: cgImage)
    return uiImage.pngData()?.base64EncodedString() ?? ""
  }

  @objc
  func unloadModels(_ resolve: @escaping RCTPromiseResolveBlock,
                     rejecter reject: @escaping RCTPromiseRejectBlock) {
    classifierModule = nil
    segmentationModule = nil
    loadedQuantized = nil
    resolve(true)
  }
}
