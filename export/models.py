"""
Model definitions for ExecuTorch export.
Must match module3_cnn/train_cnn_classifier.py and train_unet_segmentation.py
EXACTLY (layer names, shapes) so the .pt state_dicts load without remapping.
"""
import torch
import torch.nn as nn

CLASSES = ["healthy", "leaf_blight", "leaf_rust", "powdery_mildew"]
IMG_SIZE = 64


class SimpleCNN(nn.Module):
    """Compact CNN trained from scratch for 64x64 leaf images. (module3_cnn)"""

    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )


class MiniUNet(nn.Module):
    """Compact UNet for 64x64 lesion segmentation. (module3_cnn)"""

    def __init__(self, in_c=3, base=16):
        super().__init__()
        self.enc1 = conv_block(in_c, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.out_conv = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)  # raw logits [N,1,64,64] — apply sigmoid at inference


class UNetWithSigmoid(nn.Module):
    """Wraps MiniUNet so the exported graph outputs probabilities directly,
    since mobile inference code shouldn't have to know about logits."""

    def __init__(self, unet: MiniUNet):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        return torch.sigmoid(self.unet(x))
