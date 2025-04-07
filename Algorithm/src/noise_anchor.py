import torch
import torch.nn as nn
import torch.nn.functional as F



class NoiseAnchorGenerator(nn.Module):
    def __init__(self, in_channels=960, anchor_dim=16, num_anchors=8):
        super(NoiseAnchorGenerator, self).__init__()
        
        self.num_anchors = num_anchors
        self.anchor_dim = anchor_dim
        self.in_channels = in_channels

        # Initialize anchor centers randomly — updated later
        self.register_buffer("anchor_centers", torch.randn(num_anchors, in_channels))  # [K, C]

        # Embedding for anchor index → vector
        self.anchor_embedding = nn.Embedding(num_anchors, anchor_dim)

        self.noise_net = nn.Sequential(
            nn.Conv2d(in_channels + in_channels + anchor_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, in_channels, kernel_size=3, padding=1)
        )

    def cluster_anchor(self, img_feat):
        B, C, H, W = img_feat.shape
        features = img_feat.permute(0, 2, 3, 1).reshape(-1, C)  # [BHW, C]
        anchors = self.anchor_centers  # [K, C]

        # Compute L2 distance to anchors
        dists = torch.cdist(features, anchors, p=2)  # [BHW, K]
        assignments = torch.argmin(dists, dim=1)  # [BHW]

        # Reshape back to [B, H, W]
        anchor_type = assignments.view(B, H, W)
        return anchor_type  # [B, H, W]

    def forward(self, img_feat):
        B, C, H, W = img_feat.shape

        # Step 1: Gaussian noise
        gaussian_noise = torch.randn_like(img_feat)

        # Step 2: Anchor assignment via K-means
        anchor_type = self.cluster_anchor(img_feat)  # [B, H, W]

        # Step 3: Convert anchor type to embedding
        anchor_emb = self.anchor_embedding(anchor_type)  # [B, H, W, D]
        anchor_emb = anchor_emb.permute(0, 3, 1, 2)  # [B, D, H, W]

        # Step 4: Concatenate [img_feat, gaussian_noise, anchor_emb]
        concat_input = torch.cat([img_feat, gaussian_noise, anchor_emb], dim=1)  # [B, 2C+D, H, W]

        # Step 5: Learn noise anchors
        noise_anchor = self.noise_net(concat_input)  # [B, C, H, W]

        return noise_anchor

# noise_anchor_generator = NoiseAnchorGenerator()
# img_feat = torch.randn(1, 960, 60, 34)
# noise_anchor = noise_anchor_generator(img_feat)
# print(noise_anchor.shape)  # [1, 960, 60, 34]