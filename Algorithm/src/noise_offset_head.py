import torch
import torch.nn as nn
import torch.nn.functional as F



class NoiseOffsetHead(nn.Module):
    def __init__(self, in_channels=960, hidden_channels=512, out_channels=960):
        super(NoiseOffsetHead, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, img_feat):
        offset = self.net(img_feat)
        return offset  # [B, 960, H, W]

# noise_offset_head = NoiseOffsetHead()
# img_feat = torch.randn(2, 960, 60, 34)
# offset = noise_offset_head(img_feat)
# print(offset.shape)  # [2, 960, 60, 34]
