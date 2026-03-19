import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, overlap_stride=None):
        super().__init__()
        stride = overlap_stride if overlap_stride is not None else patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride)
        self.patch_size = patch_size
        self.stride = stride
        self.img_size = img_size
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.proj(x)  
        num_patches_h, num_patches_w = x.shape[2], x.shape[3]
        x = x.flatten(2)  
        x = x.transpose(1, 2) 
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        if x.size(1) != self.pos_embed.size(1):
            pos_embed_interp = nn.functional.interpolate(
                self.pos_embed.permute(0,2,1), 
                size=x.size(1),
                mode='linear',
                align_corners=False
            ).permute(0,2,1) 
            return x + pos_embed_interp
        return x + self.pos_embed

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, mlp_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForwardNetwork(embed_dim, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
