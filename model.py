import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from data import CLASSES, ROOT

OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "model.pt"

# ImageNet mean and std averaged over R, G, B, because our X-rays have one channel
GRAY_MEAN, GRAY_STD = 0.449, 0.226


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    """U-Net expanding step: 2x2 up-convolution, concatenate the skip connection, two 3x3 convs."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = double_conv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class LungUNet(nn.Module):
    """U-Net with an ImageNet-pretrained ResNet-18 encoder.

    Input: grayscale X-ray, 1 x H x W, values 0..1, H and W divisible by 32.
    Output: lung mask logits (1 x H x W) and class logits (normal / lung opacity / COVID-19).
    """

    def __init__(self, pretrained=True):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # The first conv expects RGB. Summing its weights over R, G, B gives the same response to a gray image.
        conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1.weight.data = resnet.conv1.weight.data.sum(dim=1, keepdim=True)

        # contracting path (encoder): pretrained ResNet-18 stages
        self.stem = nn.Sequential(conv1, resnet.bn1, resnet.relu)  # 64 x H/2
        self.encoder = nn.ModuleList([
            nn.Sequential(resnet.maxpool, resnet.layer1),  # 64 x H/4
            resnet.layer2,  # 128 x H/8
            resnet.layer3,  # 256 x H/16
            resnet.layer4,  # 512 x H/32 (bottleneck)
        ])

        # expanding path (decoder), trained from scratch
        self.decoder = nn.ModuleList([
            UpBlock(512, 256, 128),  # -> H/16
            UpBlock(128, 128, 64),  # -> H/8
            UpBlock(64, 64, 32),  # -> H/4
            UpBlock(32, 64, 16),  # -> H/2
            UpBlock(16, 1, 16),  # -> H, skip = the input image itself
        ])
        self.mask_head = nn.Conv2d(16, 1, kernel_size=1)

        # classification head on the bottleneck features
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, len(CLASSES)),
        )

    def encoder_parameters(self):
        return [*self.stem.parameters(), *self.encoder.parameters()]

    def forward(self, image):
        x = (image - GRAY_MEAN) / GRAY_STD
        skips = [image, self.stem(x)]
        x = skips[-1]
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
        bottleneck = skips.pop()
        x = bottleneck
        for block in self.decoder:
            x = block(x, skips.pop())
        return self.mask_head(x), self.class_head(bottleneck)


def dice_scores(mask_logits, masks):
    """Dice overlap between predicted and true lung mask, one score per image (1 = perfect)."""
    predicted = (mask_logits > 0).float()
    overlap = (predicted * masks).sum(dim=(1, 2, 3))
    total = predicted.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return (2 * overlap + 1e-6) / (total + 1e-6)


def save_model(model):
    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)


def load_model():
    model = LungUNet(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    return model.eval()
