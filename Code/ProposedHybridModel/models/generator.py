import torch
import torch.nn as nn
import math
from components.vit_modules import PatchEmbedding, PositionalEncoding, TransformerBlock

class Generator(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, out_channels=3,
                 embed_dim=768, num_heads=8, num_transformer_blocks=3, mlp_dim=3072):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_transformer_blocks = num_transformer_blocks

        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, overlap_stride=patch_size // 2)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            num_patches = self.patch_embedding(dummy).shape[1]
        self.pos_encoding = PositionalEncoding(embed_dim, num_patches)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim) 
            for _ in range(num_transformer_blocks)
        ])

        self.downsample_layers = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.upsample_layers = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 4, out_channels, kernel_size=4, stride=2, padding=1),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, blurred_image):
        B, C, H, W = blurred_image.shape

        x = self.patch_embedding(blurred_image)  

        x = self.pos_encoding(x)  

        for block in self.transformer_blocks:
            x = block(x)  

        num_patches = x.shape[1]
        sqrt_num_patches = int(math.sqrt(num_patches))
        assert sqrt_num_patches * sqrt_num_patches == num_patches, f"num_patches ({num_patches}) is not a perfect square!"

        x = x.permute(0, 2, 1).reshape(B, self.embed_dim, sqrt_num_patches, sqrt_num_patches)

        x = self.downsample_layers(x)

        x = self.upsample_layers(x)

        deblurred_image = self.final_conv(x)
        if deblurred_image.shape[-2:] != (H, W):
            deblurred_image = torch.nn.functional.interpolate(
                deblurred_image, size=(H, W), mode="bilinear", align_corners=False
            )
        return deblurred_image, deblurred_image
