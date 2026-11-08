# Trained model checkpoints go here

The uploaded ZIP did **not** contain the trained PyTorch checkpoint files. Place the two files from the AgriSense training project here (or point the export command at their existing folder):

- `cnn_classifier.pt`
- `unet_segmentation.pt`

The export command expects them in the same directory. They must match the architecture and preprocessing defined in `export/models.py` and the mobile Kotlin/Swift preprocessing. Do not substitute a different checkpoint without checking the architecture, class order, image size, and normalization first.
