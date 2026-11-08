import { NativeModules } from 'react-native';

const { ExecuTorchModule } = NativeModules;

function requireNativeModule() {
  if (!ExecuTorchModule) {
    throw new Error('ExecuTorch native module is not installed. Run the Android setup script and rebuild the native app.');
  }
  return ExecuTorchModule;
}

export interface ClassScore {
  label: string;
  score: number;
}

export interface ClassificationResult {
  label: string;
  confidence: number;
  allScores: ClassScore[];
  latencyMs: number;
}

export interface SegmentationResult {
  maskBase64: string; // data:image/png;base64,... — 64x64 RGBA overlay, alpha = lesion confidence
  affectedAreaPct: number;
  latencyMs: number;
}

/**
 * Thin typed wrapper over the native ExecuTorchModule (Kotlin on Android,
 * Swift on iOS). Both platforms expose the identical method signature, so
 * nothing here branches on Platform.OS.
 */
export const ExecuTorch = {
  /**
   * Loads both .pte models into memory. Call once, ideally on app start
   * (e.g. from a splash/loading screen) — first inference after a cold
   * start otherwise pays the model-load cost too.
   */
  preloadModels(useQuantized: boolean = true): Promise<boolean> {
    return requireNativeModule().preloadModels(useQuantized);
  },

  /** Runs the CNN classifier on a base64-encoded leaf photo. */
  classifyLeaf(imageBase64: string, useQuantized: boolean = true): Promise<ClassificationResult> {
    return requireNativeModule().classifyLeaf(imageBase64, useQuantized);
  },

  /** Runs the UNet segmentation model on a base64-encoded leaf photo. */
  segmentLesion(imageBase64: string, useQuantized: boolean = true): Promise<SegmentationResult> {
    return requireNativeModule().segmentLesion(imageBase64, useQuantized);
  },

  /** Frees both models from memory. Call on app background/unmount if memory-constrained. */
  unloadModels(): Promise<boolean> {
    return requireNativeModule().unloadModels();
  },
};
