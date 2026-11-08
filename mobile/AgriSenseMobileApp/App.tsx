import React, { useCallback, useEffect, useState } from 'react';
import {
  SafeAreaView,
  StatusBar,
  View,
  Text,
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
  Alert,
} from 'react-native';
import RNFS from 'react-native-fs';

import CameraScreen from './src/screens/CameraScreen';
import ResultsScreen from './src/screens/ResultsScreen';
import { ExecuTorch } from './src/native/executorch';
import type { ClassificationResult, SegmentationResult } from './src/native/executorch';
import { colors, spacing, typography } from './src/utils/theme';

type Stage = 'loading-models' | 'model-load-failed' | 'camera' | 'processing' | 'results' | 'preview-results';

// Toggle this to compare fp32 vs int8 models live — wire to a settings
// screen for a real app; hardcoded here since the coursework demo's point
// is to show the tradeoff, not hide it behind a menu.
const DEFAULT_USE_QUANTIZED = false;

export default function App() {
  const [stage, setStage] = useState<Stage>('loading-models');
  const [loadError, setLoadError] = useState<string | null>(null);
  const [useQuantized, setUseQuantized] = useState(DEFAULT_USE_QUANTIZED);

  const [previewOnly, setPreviewOnly] = useState(false);
  const [photoPath, setPhotoPath] = useState<string | null>(null);
  const [classification, setClassification] = useState<ClassificationResult | null>(null);
  const [segmentation, setSegmentation] = useState<SegmentationResult | null>(null);

  const loadModels = useCallback(async (quantized: boolean) => {
    setStage('loading-models');
    setLoadError(null);
    try {
      await ExecuTorch.preloadModels(quantized);
      setPreviewOnly(false);
      setStage('camera');
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : 'Unknown error loading models');
      setStage('model-load-failed');
    }
  }, []);

  useEffect(() => {
    loadModels(useQuantized);
  }, [loadModels, useQuantized]);

  const handleCaptured = useCallback(
    async (base64: string) => {
      if (previewOnly) {
        try {
          const previewPath = `${RNFS.CachesDirectoryPath}/leaf_preview_${Date.now()}.jpg`;
          await RNFS.writeFile(previewPath, base64, 'base64');
          setPhotoPath(previewPath);
          setStage('preview-results');
        } catch (err) {
          Alert.alert('Preview failed', err instanceof Error ? err.message : 'Could not save the photo.');
          setStage('camera');
        }
        return;
      }

      setStage('processing');

      try {
        // Persist the raw capture to a temp file path purely so ResultsScreen
        // can display it with <Image source={{uri: file://...}}> — the
        // inference calls themselves consume the base64 string directly.
        const tempPath = `${RNFS.CachesDirectoryPath}/leaf_${Date.now()}.jpg`;
        await RNFS.writeFile(tempPath, base64, 'base64');
        setPhotoPath(tempPath);

        const [classResult, segResult] = await Promise.all([
          ExecuTorch.classifyLeaf(base64, useQuantized),
          ExecuTorch.segmentLesion(base64, useQuantized),
        ]);

        setClassification(classResult);
        setSegmentation(segResult);
        setStage('results');
      } catch (err) {
        Alert.alert(
          'Diagnosis failed',
          err instanceof Error ? err.message : 'Could not run on-device models. Try again.',
        );
        setStage('camera');
      }
    },
    [previewOnly, useQuantized],
  );

  const handleRetake = useCallback(() => {
    setPhotoPath(null);
    setClassification(null);
    setSegmentation(null);
    setStage('camera');
  }, []);

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor={colors.soilBlack} />

      {stage === 'loading-models' && (
        <View style={styles.centered}>
          <ActivityIndicator size="large" color={colors.chlorophyll} />
          <Text style={styles.loadingText}>Loading on-device models…</Text>
          <Text style={styles.loadingSubtext}>
            {useQuantized ? 'INT8 quantized' : 'FP32 baseline'} · CNN classifier + UNet segmentation
          </Text>
        </View>
      )}

      {stage === 'model-load-failed' && (
        <View style={styles.centered}>
          <Text style={styles.errorTitle}>Couldn't load models</Text>
          <Text style={styles.errorText}>{loadError}</Text>
          <Text style={styles.errorHint}>
            Check that cnn_classifier_{'{fp32,int8}'}.pte and
            unet_segmentation_{'{fp32,int8}'}.pte are bundled in
            android/app/src/main/assets/models/ (Android) or added to the
            Xcode target (iOS).
          </Text>
          <TouchableOpacity style={styles.retryButton} onPress={() => loadModels(useQuantized)}>
            <Text style={styles.retryButtonText}>Retry model loading</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.previewButton}
            onPress={() => {
              setPreviewOnly(true);
              setStage('camera');
            }}
          >
            <Text style={styles.previewButtonText}>Open camera preview only</Text>
          </TouchableOpacity>
        </View>
      )}

      {stage === 'camera' && (
        <>
          <CameraScreen onCaptured={handleCaptured} />
          {!previewOnly && <View style={styles.precisionToggle}>
            <Text style={styles.precisionLabel}>
              {useQuantized ? 'INT8' : 'FP32'}
            </Text>
            <TouchableOpacity
              style={styles.precisionSwitch}
              onPress={() => setUseQuantized((q) => !q)}
            >
              <Text style={styles.precisionSwitchText}>Switch model</Text>
            </TouchableOpacity>
          </View>}
          {previewOnly && (
            <View style={styles.previewBanner}>
              <Text style={styles.previewBannerText}>PREVIEW ONLY · AI models are not loaded</Text>
            </View>
          )}
        </>
      )}

      {stage === 'processing' && (
        <View style={styles.centered}>
          <ActivityIndicator size="large" color={colors.chlorophyll} />
          <Text style={styles.loadingText}>Running on-device diagnosis…</Text>
        </View>
      )}

      {(stage === 'results' || stage === 'preview-results') && photoPath && (
        <ResultsScreen
          photoUri={photoPath}
          classification={classification}
          segmentation={segmentation}
          isQuantized={useQuantized}
          previewOnly={stage === 'preview-results'}
          onRetake={handleRetake}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: colors.soilBlack },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.lg,
  },
  loadingText: { ...typography.body, color: colors.paper, marginTop: spacing.md },
  loadingSubtext: { ...typography.caption, color: colors.paperDim, marginTop: spacing.xs },
  errorTitle: { ...typography.title, color: colors.blight, marginBottom: spacing.sm },
  errorText: { ...typography.body, color: colors.paper, textAlign: 'center', marginBottom: spacing.md },
  errorHint: {
    ...typography.caption,
    color: colors.paperDim,
    textAlign: 'center',
    marginBottom: spacing.lg,
  },
  retryButton: {
    backgroundColor: colors.chlorophyll,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: 999,
  },
  retryButtonText: { ...typography.button, color: colors.soilBlack },
  previewButton: { marginTop: spacing.md, borderWidth: 1, borderColor: colors.chlorophyll, paddingVertical: spacing.sm, paddingHorizontal: spacing.lg, borderRadius: 999 },
  previewButtonText: { ...typography.button, color: colors.chlorophyll },
  previewBanner: { position: 'absolute', bottom: spacing.md, left: spacing.md, right: spacing.md, backgroundColor: 'rgba(15,27,18,0.88)', borderWidth: 1, borderColor: colors.harvest, borderRadius: 10, padding: spacing.sm },
  previewBannerText: { ...typography.caption, color: colors.harvest, textAlign: 'center' },
  precisionToggle: {
    position: 'absolute',
    top: spacing.md,
    right: spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(15,27,18,0.75)',
    borderRadius: 999,
    paddingVertical: 6,
    paddingHorizontal: spacing.sm,
    gap: spacing.sm,
  },
  precisionLabel: { ...typography.caption, color: colors.chlorophyll, fontWeight: '700' },
  precisionSwitch: { paddingHorizontal: spacing.sm, paddingVertical: 4 },
  precisionSwitchText: { ...typography.caption, color: colors.paper },
});
