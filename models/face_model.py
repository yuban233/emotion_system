# models/face_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

# -------------------------
# 1. 定义模型结构
# -------------------------
class FaceEmotionModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        # 修改第一层为单通道输入
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改输出层为目标类别数
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

# -------------------------
# 2. 初始化模型并加载权重
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceEmotionModel(num_classes=4).to(device)
LABELS = ["angry", "happy", "sad", "neutral"]

# 这里改成你训练好的模型路径
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_emotion_model.pth")

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    print(f"Warning: 模型权重 {MODEL_PATH} 不存在，使用随机初始化模型")

# -------------------------
# 3. 图像预处理（与训练保持一致：48x48 灰度）
# -------------------------
from utils.transforms import inference_transform
preprocess = inference_transform

# -------------------------
# 4. 封装推理函数
# -------------------------
def predict_emotion(image) -> str:
    """
    输入: PIL Image / OpenCV ndarray
    输出: emotion 标签 (str)
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            # OpenCV 默认是 BGR，这里转成 RGB 给 PIL/torchvision 处理
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    img = preprocess(image).unsqueeze(0).to(device)  # 增加 batch 维度
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    pred_idx = int(predicted.item())
    if 0 <= pred_idx < len(LABELS):
        return LABELS[pred_idx]
    return "neutral"


def predict_emotion_with_scores(image) -> dict:
    """
    输入: PIL Image / OpenCV ndarray
    输出: {emotion, confidence, scores}
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    img = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()

    pred_idx = int(np.argmax(probs))
    if pred_idx < 0 or pred_idx >= len(LABELS):
        pred_idx = LABELS.index("neutral") if "neutral" in LABELS else 0

    scores = {label: round(float(probs[i]), 6) for i, label in enumerate(LABELS)}
    return {
        "emotion": LABELS[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "scores": scores,
    }