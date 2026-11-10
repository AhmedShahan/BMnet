import torch.nn as nn
from torchvision.models import mnasnet1_0


class BMnet(nn.Module):
    """MNASNet (depth multiplier 1) -> global avg pool -> MLP(128) -> 14 outputs."""

    def __init__(self, n_out=14, pretrained=False):
        super().__init__()
        m = mnasnet1_0(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = m.layers                    # (B, 1280, h, w)
        self.head = nn.Sequential(nn.Linear(1280, 128), nn.ReLU(inplace=True),
                                  nn.Linear(128, n_out))

    def forward(self, x):
        return self.head(self.backbone(x).mean(dim=(2, 3)))