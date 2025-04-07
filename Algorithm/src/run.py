import torch
from data_loader import create_dataloader
from backbone import MobileNetV4FeatureExtractor
from noise_anchor import NoiseAnchorGenerator
from noise_offset_head import NoiseOffsetHead
from fusion import FinalNoise, FusionHead
from upsample import TransformerUpsampler
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




def main():
    clean_data_path = r"D:\智能图像-数据\给参赛者下载的数据\示例图片\GT"
    noise_data_path = r"D:\智能图像-数据\给参赛者下载的数据\示例图片\noise"

    # Configuration
    batch_size = 1  # the batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Dataloader
    dataloader = create_dataloader(clean_data_path, noise_data_path, batch_size=batch_size)

    # Create models and modules
    backbone = MobileNetV4FeatureExtractor().to(device)
    noise_anchor_generator = NoiseAnchorGenerator().to(device)
    head = NoiseOffsetHead().to(device)

    config = {
        "d_model": 960,
        "nhead": 8,
        "num_layers": 4,
        "input_channels": 960
    }

    fusion_head = FusionHead(config).to(device)

    # Training Loop
    for batch in dataloader:
        noise_img = batch["noise_image"].to(device)
        clean_img = batch["clean_image"].to(device)
        img_feat = backbone(noise_img)
        print(img_feat.shape)
        noise_anchor = noise_anchor_generator(img_feat)
        noise_offset = head(img_feat)
        noise_feature = FinalNoise(noise_anchor, noise_offset)
        final_noise_feature = fusion_head(noise_feature, img_feat)
        print(final_noise_feature.shape)  # torch.Size([1, 960, 60, 34])
        model = TransformerUpsampler(embed_dim=768)
        output = noise_img - model(final_noise_feature)  # torch.Size([1, 3, 960, 544])
        print(output.shape)
        # Step 1: Convert the tensor to a NumPy array
        # Remove the batch dimension by selecting the first element (since batch size is 1)
        output_image = output.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

        # Step 2: Normalize the image (optional if the tensor values are between 0 and 1)
        # If your tensor is between 0 and 1, multiply by 255 for visualization
        output_image = (output_image * 255).astype(np.uint8)

        # Step 3: Display the image
        plt.imshow(output_image)
        plt.axis('off')  # Hide axes
        plt.show()
        break


if __name__ == "__main__":
    main()
