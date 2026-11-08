import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import RNFS from 'react-native-fs';
import { colors, spacing, typography } from '../utils/theme';

interface Props {
  onCaptured: (base64: string) => void;
}

/**
 * Full-screen camera view for photographing a leaf. Captures a photo,
 * reads it back as base64, and hands it to the parent (which runs both
 * the classifier and segmentation model against the same image).
 *
 * Uses the back camera by default since that's what you point at a leaf;
 * a flip-to-front toggle isn't included since it isn't part of this flow.
 */
export default function CameraScreen({ onCaptured }: Props) {
  const camera = useRef<Camera>(null);
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const [isCapturing, setIsCapturing] = useState(false);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const handleCapture = useCallback(async () => {
    if (!camera.current || isCapturing) return;
    setIsCapturing(true);
    try {
      const photo = await camera.current.takePhoto({
        flash: 'off',
        enableShutterSound: true,
      });
      const base64 = await RNFS.readFile(photo.path, 'base64');
      onCaptured(base64);
    } catch (err) {
      Alert.alert(
        'Capture failed',
        err instanceof Error ? err.message : 'Could not capture photo. Try again.',
      );
    } finally {
      setIsCapturing(false);
    }
  }, [isCapturing, onCaptured]);

  if (!hasPermission) {
    return (
      <View style={styles.centered}>
        <Text style={styles.permissionText}>
          AgriSense needs camera access to photograph leaves for diagnosis.
        </Text>
        <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
          <Text style={styles.permissionButtonText}>Grant camera access</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.centered}>
        <Text style={styles.permissionText}>No back camera found on this device.</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        photo={true}
      />

      <View style={styles.frameOverlay} pointerEvents="none">
        <View style={styles.frameGuide} />
        <Text style={styles.frameHint}>Center a single leaf in the frame</Text>
      </View>

      <View style={styles.controls}>
        <TouchableOpacity
          style={[styles.shutterButton, isCapturing && styles.shutterButtonDisabled]}
          onPress={handleCapture}
          disabled={isCapturing}
        >
          {isCapturing ? (
            <ActivityIndicator color={colors.background} />
          ) : (
            <View style={styles.shutterInner} />
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.black },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.lg,
    backgroundColor: colors.black,
  },
  permissionText: {
    ...typography.body,
    color: colors.paper,
    textAlign: 'center',
    marginBottom: spacing.md,
  },
  permissionButton: {
    backgroundColor: colors.chlorophyll,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: 999,
  },
  permissionButtonText: { ...typography.button, color: colors.black },
  frameOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
  },
  frameGuide: {
    width: 260,
    height: 260,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: 'rgba(245,241,230,0.6)',
    borderStyle: 'dashed',
  },
  frameHint: {
    ...typography.caption,
    color: colors.paper,
    marginTop: spacing.md,
    backgroundColor: 'rgba(15,27,18,0.6)',
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
    borderRadius: 6,
  },
  controls: {
    position: 'absolute',
    bottom: spacing.xl,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  shutterButton: {
    width: 76,
    height: 76,
    borderRadius: 38,
    backgroundColor: 'rgba(245,241,230,0.15)',
    borderWidth: 3,
    borderColor: colors.paper,
    justifyContent: 'center',
    alignItems: 'center',
  },
  shutterButtonDisabled: { opacity: 0.6 },
  shutterInner: {
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: colors.paper,
  },
});
