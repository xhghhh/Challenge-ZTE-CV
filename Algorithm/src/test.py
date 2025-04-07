import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import create_dataloader
from backbone import MobileNetV4FeatureExtractor
from noise_anchor import NoiseAnchorGenerator
from noise_offset_head import NoiseOffsetHead
from fusion import FinalNoise, FusionHead
from upsample import TransformerUpsampler
from torchvision.utils import save_image


# Workaround for OpenMP warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def visualize_image(tensor_img, save_path=None):
    img = tensor_img.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    plt.imshow(img)
    plt.axis('off')
    if save_path:
        plt.imsave(save_path, img)
    else:
        plt.show()

def main():
    clean_data_path = r"D:\智能图像-数据\给参赛者下载的数据\示例图片\GT"
    noise_data_path = r"D:\智能图像-数据\给参赛者下载的数据\示例图片\noise"
    checkpoint_path = r"D:\智能图像-数据\给参赛者下载的数据\黄立晨_电子科技大学\Algorithm\src\checkpoint_epoch_1.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = create_dataloader(clean_data_path, noise_data_path, shuffle=False, batch_size=1)

    # Load model components
    backbone = MobileNetV4FeatureExtractor().to(device)
    noise_anchor_generator = NoiseAnchorGenerator().to(device)
    head = NoiseOffsetHead().to(device)
    fusion_head = FusionHead({
        "d_model": 960,
        "nhead": 8,
        "num_layers": 4,
        "input_channels": 960
    }).to(device)
    model = TransformerUpsampler(embed_dim=768).to(device)

    # Load checkpoint weights (only TransformerUpsampler has checkpoint)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            noise_img = batch["noise_image"].to(device)
            clean_img = batch["clean_image"].to(device)  # optional, for computing metrics

            # Forward pass
            img_feat = backbone(noise_img)
            noise_anchor = noise_anchor_generator(img_feat)
            noise_offset = head(img_feat)
            noise_feature = FinalNoise(noise_anchor, noise_offset)
            final_noise_feature = fusion_head(noise_feature, img_feat)
            output = model(final_noise_feature)
            denoised_img = noise_img - output

            # Save or show output
            print(f"Inference image {i+1} done. Shape: {denoised_img.shape}")
            visualize_image(denoised_img, save_path=f"output_denoised_{i+1}.png")


if __name__ == "__main__":
    main()
