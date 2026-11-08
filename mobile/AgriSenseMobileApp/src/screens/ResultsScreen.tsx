import React from 'react';
import { View, Text, Image, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { colors, spacing, typography, riskColor } from '../utils/theme';
import type { ClassificationResult, SegmentationResult } from '../native/executorch';

interface Props {
  photoUri: string;
  classification: ClassificationResult | null;
  segmentation: SegmentationResult | null;
  isQuantized: boolean;
  previewOnly?: boolean;
  onRetake: () => void;
}

const LABEL_DISPLAY: Record<string, string> = {
  healthy: 'Healthy',
  leaf_blight: 'Leaf Blight',
  leaf_rust: 'Leaf Rust',
  powdery_mildew: 'Powdery Mildew',
};

export default function ResultsScreen({
  photoUri,
  classification,
  segmentation,
  isQuantized,
  previewOnly = false,
  onRetake,
}: Props) {
  const topColor = classification ? riskColor(classification.label) : colors.paperDim;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.imageStack}>
        <Image source={{ uri: `file://${photoUri}` }} style={styles.leafImage} />
        {segmentation && (
          <Image
            source={{ uri: segmentation.maskBase64 }}
            style={[styles.leafImage, styles.maskOverlay]}
            resizeMode="stretch"
          />
        )}
      </View>

      {previewOnly && (
        <View style={styles.previewNotice}>
          <Text style={styles.previewNoticeTitle}>Photo preview only</Text>
          <Text style={styles.previewNoticeText}>No AI diagnosis was generated. Add the exported .pte model files and load them to run the real classifier and lesion segmentation.</Text>
        </View>
      )}

      {classification && (
        <View style={styles.diagnosisCard}>
          <View style={styles.diagnosisHeader}>
            <View style={[styles.riskDot, { backgroundColor: topColor }]} />
            <Text style={styles.diagnosisLabel}>
              {LABEL_DISPLAY[classification.label] ?? classification.label}
            </Text>
          </View>
          <Text style={styles.confidenceText}>
            {(classification.confidence * 100).toFixed(1)}% confidence
          </Text>

          <View style={styles.scoreBars}>
            {classification.allScores
              .slice()
              .sort((a, b) => b.score - a.score)
              .map((s) => (
                <View key={s.label} style={styles.scoreRow}>
                  <Text style={styles.scoreLabel}>{LABEL_DISPLAY[s.label] ?? s.label}</Text>
                  <View style={styles.scoreBarTrack}>
                    <View
                      style={[
                        styles.scoreBarFill,
                        { width: `${s.score * 100}%`, backgroundColor: riskColor(s.label) },
                      ]}
                    />
                  </View>
                  <Text style={styles.scoreValue}>{(s.score * 100).toFixed(0)}%</Text>
                </View>
              ))}
          </View>
        </View>
      )}

      {segmentation && (
        <View style={styles.diagnosisCard}>
          <Text style={styles.sectionTitle}>Lesion coverage</Text>
          <Text style={styles.affectedArea}>{segmentation.affectedAreaPct.toFixed(1)}%</Text>
          <Text style={styles.affectedAreaLabel}>of imaged leaf area affected</Text>
        </View>
      )}

      {!previewOnly && <View style={styles.perfCard}>
        <Text style={styles.sectionTitle}>On-device performance</Text>
        <View style={styles.perfRow}>
          <Text style={styles.perfLabel}>Model precision</Text>
          <Text style={styles.perfValue}>{isQuantized ? 'INT8 (quantized)' : 'FP32 (baseline)'}</Text>
        </View>
        {classification && (
          <View style={styles.perfRow}>
            <Text style={styles.perfLabel}>Classifier latency</Text>
            <Text style={styles.perfValue}>{classification.latencyMs.toFixed(1)} ms</Text>
          </View>
        )}
        {segmentation && (
          <View style={styles.perfRow}>
            <Text style={styles.perfLabel}>Segmentation latency</Text>
            <Text style={styles.perfValue}>{segmentation.latencyMs.toFixed(1)} ms</Text>
          </View>
        )}
      </View>}

      <TouchableOpacity style={styles.retakeButton} onPress={onRetake}>
        <Text style={styles.retakeButtonText}>Scan another leaf</Text>
      </TouchableOpacity>

      <Text style={styles.disclaimer}>
        {previewOnly
          ? 'Preview mode does not analyse plant health. Do not use this screen to make crop-treatment decisions.'
          : 'On-device model output is an aid, not a definitive diagnosis. Confirm disease symptoms before taking crop-treatment decisions.'}
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.soilBlack },
  content: { padding: spacing.md, paddingBottom: spacing.xl },
  imageStack: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: spacing.md,
    backgroundColor: colors.soilBlack2,
  },
  leafImage: { width: '100%', height: '100%', position: 'absolute' },
  maskOverlay: { opacity: 0.85 },
  previewNotice: { backgroundColor: colors.soilBlack2, borderRadius: 10, padding: spacing.md, marginBottom: spacing.md, borderWidth: 1, borderColor: colors.harvest },
  previewNoticeTitle: { ...typography.subtitle, color: colors.harvest, marginBottom: spacing.xs },
  previewNoticeText: { ...typography.body, color: colors.paperDim },
  diagnosisCard: {
    backgroundColor: colors.soilBlack2,
    borderRadius: 10,
    padding: spacing.md,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: 'rgba(245,241,230,0.08)',
  },
  diagnosisHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: spacing.xs },
  riskDot: { width: 10, height: 10, borderRadius: 5, marginRight: spacing.sm },
  diagnosisLabel: { ...typography.title, color: colors.paper },
  confidenceText: { ...typography.body, color: colors.paperDim, marginBottom: spacing.md },
  scoreBars: { gap: spacing.sm },
  scoreRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  scoreLabel: { ...typography.caption, color: colors.paperDim, width: 110 },
  scoreBarTrack: {
    flex: 1,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(245,241,230,0.1)',
    overflow: 'hidden',
  },
  scoreBarFill: { height: '100%', borderRadius: 4 },
  scoreValue: { ...typography.caption, color: colors.paper, width: 36, textAlign: 'right' },
  sectionTitle: {
    ...typography.subtitle,
    color: colors.harvest,
    marginBottom: spacing.sm,
    textTransform: 'uppercase',
    fontSize: 12,
    letterSpacing: 0.5,
  },
  affectedArea: { fontSize: 32, fontWeight: '700', color: colors.paper },
  affectedAreaLabel: { ...typography.caption, color: colors.paperDim },
  perfCard: {
    backgroundColor: colors.soilBlack2,
    borderRadius: 10,
    padding: spacing.md,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: 'rgba(245,241,230,0.08)',
  },
  perfRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 6,
  },
  perfLabel: { ...typography.body, color: colors.paperDim },
  perfValue: { ...typography.body, color: colors.chlorophyll, fontWeight: '600' },
  retakeButton: {
    backgroundColor: colors.chlorophyll,
    borderRadius: 999,
    paddingVertical: spacing.sm + 4,
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  retakeButtonText: { ...typography.button, color: colors.soilBlack },
  disclaimer: {
    ...typography.caption,
    color: colors.paperDim,
    textAlign: 'center',
    opacity: 0.7,
    paddingHorizontal: spacing.md,
  },
});
