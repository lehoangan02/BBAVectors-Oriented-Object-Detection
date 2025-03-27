import torch
from torchvision.models import swin_b, Swin_B_Weights

# class InputProcessor(torch.nn.Module):


class SwinEncoder(torch.nn.Module):
    def __init__(self, pretrained = True, freeze_backbone = True):
        super.__init__()

        # Load pre-trained Swin model
        if pretrained:
            weights = Swin_B_Weights.IMAGENET1K_V1
        else:
            weights = None
        swin = swin_b(weights = weights, progress = True)

        self.encoder = torch.nn.Sequential(
            swin.features,
            swin.norm
        )

        # Freeze the backbone
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        return x