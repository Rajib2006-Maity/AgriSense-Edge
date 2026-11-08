// Reuses the palette from the AgriSense AI web dashboard so the mobile app
// reads as part of the same product, not a bolted-on demo.
export const colors = {
  soilBlack: '#0F1B12',
  soilBlack2: '#142218',
  paper: '#F5F1E6',
  paperDim: '#C9C4B3',
  chlorophyll: '#8FBC6A',
  chlorophyllDim: '#5E8A47',
  harvest: '#D9A441',
  irrigation: '#4A7C6F',
  blight: '#C4472F',
  black: '#000000',
  background: '#0F1B12',
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 40,
};

export const typography = {
  title: { fontSize: 26, fontWeight: '700' as const },
  subtitle: { fontSize: 16, fontWeight: '500' as const },
  body: { fontSize: 15, fontWeight: '400' as const },
  caption: { fontSize: 12, fontWeight: '500' as const },
  button: { fontSize: 15, fontWeight: '700' as const },
  mono: { fontFamily: 'JetBrainsMono-Regular', fontSize: 13 },
};

export const riskColor = (label: string): string => {
  if (label === 'healthy') return colors.chlorophyll;
  if (label === 'leaf_rust') return colors.harvest;
  return colors.blight; // leaf_blight, powdery_mildew
};
