import torch
from torch import nn
from torchvision.models import swin_b, Swin_B_Weights

# class InputProcessor(nn.Module):


class SwinEncoder(nn.Module):
    def __init__(self, pretrained = True, freeze_backbone = True):
        super().__init__()

        # Load pre-trained Swin model
        if pretrained:
            weights = Swin_B_Weights.IMAGENET1K_V1
        else:
            weights = None
        swin = swin_b(weights = weights, progress = True)

        self.encoder = swin.features

        self.extract1 = nn.Sequential(
            nn.LayerNorm(128, eps=1e-5, elementwise_affine=True),
            swin.permute,
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )
        self.extract2 = nn.Sequential(
            nn.LayerNorm(256, eps=1e-5, elementwise_affine=True),
            swin.permute,
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512)
        )
        self.extract3 = nn.Sequential(
            nn.LayerNorm(512, eps=1e-5, elementwise_affine=True),
            swin.permute,
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024)
        )
        self.extract4 = nn.Sequential(
            swin.norm,
            swin.permute,
            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048)
        )

        # Freeze the backbone
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Initialize extractors
        for extractor in [self.extract1, self.extract2, self.extract3, self.extract4]:
            for layer in extractor:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        features = []

        extractors = [self.extract1, self.extract2, self.extract3, self.extract4]
        extraction_indices = {1, 3, 5, 7}
        extractor_idx = 0
        
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in extraction_indices:
                features.append(extractors[extractor_idx](x))
                extractor_idx += 1
        
        return features
    
# Example usage
if __name__ == "__main__":
    model = SwinEncoder(pretrained = True, freeze_backbone = True)
    dummy_input = torch.randn(1, 3, 608, 608)
    fmaps = model(dummy_input)
    for fmap in fmaps:
        print(fmap.shape)