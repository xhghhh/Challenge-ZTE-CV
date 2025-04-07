import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


# 数据集路径
clean_data_path = r"D:\智能图像-数据\给参赛者下载的数据\示例图片\GT"
noise_data_path = r"D:\智能图像-数据\给参赛者下载的数据\示例图片\noise"

# 将PIL图像转换为Tensor
def to_tensor(input_img):
    transforms = torchvision.transforms.ToTensor()
    tensor = transforms(input_img)
    return tensor

# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, transform=None):
        """
        初始化数据集
        :param clean_dir: 干净图像的目录路径
        :param noise_dir: 噪声图像的目录路径
        :param transform: 转换操作
        """
        self.clean_dir = clean_dir
        self.noise_dir = noise_dir
        self.transform = transform
        self.clean_images = os.listdir(clean_dir)
        self.noise_images = os.listdir(noise_dir)

    def __len__(self):
        """
        返回数据集中图像的数量
        """
        return len(self.clean_images)

    def __getitem__(self, index):
        """
        根据索引获取图像对
        """
        clean_image_path = os.path.join(self.clean_dir, self.clean_images[index])
        noise_image_path = os.path.join(self.noise_dir, self.noise_images[index])

        clean_image = Image.open(clean_image_path).convert('RGB')
        noise_image = Image.open(noise_image_path).convert('RGB')

        if self.transform:
            clean_image = self.transform(clean_image)
            noise_image = self.transform(noise_image)

        return {
            'clean_image': clean_image,
            'noise_image': noise_image,
            'index': index
        }

# 创建数据加载器
def create_dataloader(clean_dir, noise_dir, batch_size=32, shuffle=True, num_workers=4):
    """
    创建数据加载器
    :param clean_dir: 干净图像的目录路径
    :param noise_dir: 噪声图像的目录路径
    :param batch_size: 批量大小
    :param shuffle: 是否打乱数据
    :param num_workers: 工作线程数
    :return: 数据加载器
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((1920, 1080)),  # 调整图像大小
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    dataset = ImageDataset(clean_dir, noise_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

