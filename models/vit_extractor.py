import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights

class InputProcessor(torch.nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        # x is output of conv_proj with shape: (batch_size, hidden_dim, H', W')
        n, c, h, w = x.shape  # H' = W' = image_size // patch_size

        # (n, c, h, w) --> (n, c, h*w) --> (n, h*w, c)
        x = x.reshape(n, c, h * w).permute(0, 2, 1)

        # Concat class token --> (n, 1 + h*w, c)
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Positional embeddings
        x = x + self.vit.encoder.pos_embedding
        return x

class ViTExtractor(torch.nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()

        # Load pre-trained ViT model
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.vit = vit_b_16(weights=weights, progress=True)
        
        # Separate for freezing option
        self.feature_extractor = torch.nn.Sequential(
            self.vit.conv_proj,
            InputProcessor(self.vit)
        )
        
        # Separate for freezing option
        self.encoder = self.vit.encoder
        
        # Freeze if needed
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Convolutions for multi-scale feature maps
        # 1/4 scale
        self.conv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(768, 256, kernel_size=8, stride=4, padding=2),
            torch.nn.BatchNorm2d(256),
            # torch.nn.LayerNorm([152, 152]),
            torch.nn.ReLU()
        )
        # 1/8 scale
        self.conv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            # torch.nn.LayerNorm([76, 76]),
            torch.nn.ReLU()
        )
        # 1/16 scale
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(768, 1024, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(1024),
            # torch.nn.LayerNorm([38, 38]),
            torch.nn.ReLU()
        )
        # 1/32 scale
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(768, 2048, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(2048),
            # torch.nn.LayerNorm([19, 19]),
            torch.nn.ReLU()
        )

        # Initialize multi-scale feature map convolutions
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            for layer in conv:
                if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ConvTranspose2d):
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        # Scale image_size to 224
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # (batch_size, channels, image_size, image_size) --> (batch_size, 1 + num_patches, hidden_dim)
        x = self.feature_extractor(x)
        x = self.vit.encoder.dropout(x)
        
        features = self.encoder.layers(x)
        features = self.encoder.ln(features)
        
        # Remove class token for detection
        features = features[:, 1:, :]  # (batch_size, num_patches, hidden_dim)
        batch_size, num_tokens, hidden_dim = features.shape
        grid_size = int(num_tokens ** 0.5)  # 14 for 196 patches
        feature_map = features.reshape(batch_size, grid_size, grid_size, hidden_dim)
        feature_map = feature_map.permute(0, 3, 1, 2)  # (batch_size, hidden_dim, grid_size, grid_size)

        # Scale to 38x38
        feature_map = torch.nn.functional.interpolate(feature_map, size=(38, 38), mode='bilinear', align_corners=False)

        # Create feature maps
        feat = []
        feat.append(self.conv1(feature_map))
        feat.append(self.conv2(feature_map))
        feat.append(self.conv3(feature_map))
        feat.append(self.conv4(feature_map))
        
        return feat

# Example usage:
if __name__ == "__main__":
    model = ViTExtractor(pretrained=True, freeze_backbone=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    fmap = model(dummy_input)
    print("Feature map shape:", fmap.shape)  # Expected: (1, 768, 14, 14)
