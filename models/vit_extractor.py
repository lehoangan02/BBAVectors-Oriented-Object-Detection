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
    
    def forward(self, x):
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
        
        return feature_map