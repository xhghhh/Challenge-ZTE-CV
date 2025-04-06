import torch
import torch.nn as nn
import torch.nn.functional as F

# Combines noise anchor + noise offset
def FinalNoise(noise_anchor, noise_offset):
    return noise_anchor + noise_offset  # Element-wise fusion

# Fusion head that uses a transformer to refine the denoised output
class FusionHead(nn.Module):
    def __init__(self, config):
        super(FusionHead, self).__init__()
        self.transformer = FusionTransformer(config)

    def forward(self, final_noise, input_img):
        B, C, H, W = input_img.shape
        pos = self.transformer.get_img_pos((B, C, H, W))

        denoised_img = self.transformer(
            query=input_img,
            key=final_noise,
            value=input_img,
            query_pos=pos,
            key_pos=pos
        )
        return denoised_img


# Transformer block to fuse noise and image
class FusionTransformer(nn.Module):
    def __init__(self, config):
        super(FusionTransformer, self).__init__()
        d_model = config.get("d_model", 256)
        nhead = config.get("nhead", 8)
        num_layers = config.get("num_layers", 4)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Reduce input channels to match transformer dimension
        self.query_proj = nn.Conv2d(config["input_channels"], d_model, kernel_size=1)
        self.key_proj = nn.Conv2d(config["input_channels"], d_model, kernel_size=1)
        self.value_proj = nn.Conv2d(config["input_channels"], d_model, kernel_size=1)

        # Final projection to restore image channels
        self.output_proj = nn.Conv2d(d_model, config["input_channels"], kernel_size=1)

    def get_img_pos(self, shape):
        # Very simple sinusoidal positional encoding based on height & width
        B, C, H, W = shape
        pos = torch.linspace(0, 1, steps=H * W, device='cuda' if torch.cuda.is_available() else 'cpu')
        pos = pos.unsqueeze(0).repeat(B, 1)  # [B, H*W]
        return pos  # [B, H*W]

    def forward(self, query, key, value, query_pos, key_pos):
        B, C, H, W = query.shape

        # Project to transformer dimension
        q = self.query_proj(query).flatten(2).permute(2, 0, 1)  # [H*W, B, d_model]
        k = self.key_proj(key).flatten(2).permute(2, 0, 1)
        v = self.value_proj(value).flatten(2).permute(2, 0, 1)

        # Transformer encoder (shared q=k=v)
        fused = self.transformer(q)  # [H*W, B, d_model]

        # Restore spatial dimensions
        fused = fused.permute(1, 2, 0).reshape(B, -1, H, W)  # [B, d_model, H, W]
        return self.output_proj(fused)  # [B, C, H, W]
