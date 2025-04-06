import torch
import torch.nn as nn
import timm

# for running on windows: bash: `D:\ProgramData\anaconda3\python.exe D:\智能图像-数据\给参赛者下载的数据\黄立晨_电子科技大学\Algorithm\src\backbone.py`


class MobileNetV4FeatureExtractor(nn.Module):
    def __init__(self, model_name='mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True):
        super(MobileNetV4FeatureExtractor, self).__init__()
        # Load the pretrained model from timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=False)
        
        # Remove the classifier (we only need features)
        if hasattr(self.backbone, 'reset_classifier'):
            self.backbone.reset_classifier(0)  # this removes the head safely

    def forward(self, x):
        """
        Extract features from input of shape [B, C, H, W].
        Output shape is typically [B, C_feat, H_feat, W_feat] or flattened features depending on the model.
        """
        # This gives the output before the classification head
        features = self.backbone.forward_features(x)
        return features


# backbone = MobileNetV4FeatureExtractor()
# input= torch.rand(1, 3, 1920, 1080)  # [B, C, H, W]

# features = backbone(input)
# print(features.shape)  # [1 (B), 960, 60, 34]