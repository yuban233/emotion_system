# utils/transforms.py
# 统一的预处理管线，训练和推理共用

from torchvision import transforms

IMG_SIZE = 48  # 与 FER2013 原始分辨率一致

# ---------- 推理 / 验证用 ----------
inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),          # 自动 0-1 归一化
])

# ---------- 训练用（含数据增强） ----------
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])
