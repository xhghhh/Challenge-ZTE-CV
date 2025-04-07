import torch
from data_loader import create_dataloader
from backbone import MobileNetV4FeatureExtractor
from noise_anchor import NoiseAnchorGenerator
from noise_offset_head import NoiseOffsetHead
from fusion import FinalNoise, FusionHead
from upsample import TransformerUpsampler




def main():
    clean_data_path = r"D:\智能图像-数据\给参赛者下载的数据\示例图片\GT"
    noise_data_path = r"D:\智能图像-数据\给参赛者下载的数据\示例图片\noise"

    # Configuration
    batch_size = 1  # the batch size
    num_epochs = 10  # number of training epochs
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
    model = TransformerUpsampler(embed_dim=768).to(device)

    # Define the loss function
    loss_fn = torch.nn.MSELoss()

    # Set up the optimizer
    optimizer = torch.optim.Adam([
        {'params': backbone.parameters()},
        {'params': noise_anchor_generator.parameters()},
        {'params': head.parameters()},
        {'params': fusion_head.parameters()},
        {'params': model.parameters()},
    ], lr=1e-4)  # Adjust learning rate if necessary

    # Training Loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            noise_img = batch["noise_image"].to(device)
            clean_img = batch["clean_image"].to(device)

            # Forward pass through backbone
            img_feat = backbone(noise_img)
            noise_anchor = noise_anchor_generator(img_feat)
            noise_offset = head(img_feat)
            noise_feature = FinalNoise(noise_anchor, noise_offset)
            final_noise_feature = fusion_head(noise_feature, img_feat)

            # Pass through the Transformer upsampler
            output = model(final_noise_feature)

            # Subtract the clean image from the output to get the denoised output
            denoised_img = noise_img - output

            # Compute the loss (MSE between clean image and the denoised output)
            loss = loss_fn(denoised_img, clean_img)

            # Backward pass (compute gradients)
            optimizer.zero_grad()
            loss.backward()

            # Optimize the model (update weights)
            optimizer.step()

            # Print out loss every few iterations (for monitoring)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Save the model checkpoint after each epoch (optional)
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
