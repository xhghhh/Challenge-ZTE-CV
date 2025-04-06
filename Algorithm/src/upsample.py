import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=960, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
        return x, (H, W)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerUpsampler(nn.Module):
    def __init__(self, embed_dim=768, num_layers=4, out_channels=3):
        super().__init__()
        self.embed = PatchEmbedding(in_channels=960, embed_dim=embed_dim)
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim) for _ in range(num_layers)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=4),  # 60x34 → 240x136
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),        # 240x136 → 480x272
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_channels, kernel_size=2, stride=2),  # 480x272 → 960x544
            nn.ReLU()
        )

    def forward(self, x):
        B = x.shape[0]
        x, (H, W) = self.embed(x)                          # [B, 60×34, embed_dim]
        x = self.transformer(x)                            # [B, N, embed_dim]
        x = self.proj(x)                                   # [B, N, embed_dim]
        x = x.transpose(1, 2).reshape(B, -1, H, W)         # [B, embed_dim, 60, 34]
        x = self.upsample(x)                               # [B, 3, 960, 544]
        x = F.interpolate(x, size=(1920, 1080), mode='bilinear', align_corners=False)
        return x                                           # [B, 3, 1920, 1080]

