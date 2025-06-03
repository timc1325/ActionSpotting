import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
NUM_CLASSES = 12
FEATURE_DIM = 1280  # From EfficientNet-B0 output
CLIP_LENGTH = 15

# 3D Inverted Residual Block (simplified version)
class InvertedResidual3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        self.expand_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=(stride, 1, 1), padding=1, groups=out_channels),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels)
        )
        self.use_res_connect = (in_channels == out_channels)

    def forward(self, x):
        out = self.expand_conv(x)
        out = self.depthwise_conv(out)
        out = self.project_conv(out)
        if self.use_res_connect:
            return F.relu(x + out)
        else:
            return F.relu(out)

# Main model
class ActionSpottingModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.temporal_encoder = nn.Sequential(
            InvertedResidual3DBlock(FEATURE_DIM, 512),
            InvertedResidual3DBlock(512, 512),
            InvertedResidual3DBlock(512, 256),
            InvertedResidual3DBlock(256, 128)
        )
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Input: x of shape (B, C=FEATURE_DIM, T=15, 1, 1)
        Output: predictions of shape (B, NUM_CLASSES)
        """
        x = self.temporal_encoder(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return torch.sigmoid(self.classifier(x))

# Example usage
if __name__ == "__main__":
    logger.info("Initializing ActionSpottingModel...")
    model = ActionSpottingModel()

    # Simulate a batch of pre-extracted 2D features
    dummy_input = torch.randn(2, FEATURE_DIM, CLIP_LENGTH, 1, 1)
    logger.info(f"Dummy input shape: {dummy_input.shape}")

    output = model(dummy_input)
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Predictions: {output}")
