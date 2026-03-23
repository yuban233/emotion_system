"""
RAVDESS 语音情绪识别训练脚本
模型: CNN + BiGRU
特征: MFCC (40维)
数据集: RAVDESS Speech-only，保留4类情绪（与人脸模型对齐）
    angry(05) / happy(03) / sad(04) / neutral(01)
    丢弃: calm(02) / fearful(06) / disgust(07) / surprised(08)
"""

import os
import sys
import warnings
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.io import wavfile
import torchaudio.transforms as T
from models.audio_model import AudioEmotionModel

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
RAVDESS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "RAVDESS")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "models", "audio_emotion_model.pth")

SAMPLE_RATE = 16000
MAX_DURATION = 4.0          # 秒，截断/补零到此长度
MAX_LEN = int(SAMPLE_RATE * MAX_DURATION)  # 64000 采样点
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 64

BATCH_SIZE = 32
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

# RAVDESS 情绪编码 → 统一4类（calm 直接丢弃）
# None 表示丢弃该类
EMOTION_MAP = {
    "01": 3,     # neutral
    "02": None,  # calm     → 丢弃
    "03": 1,     # happy    → happy
    "04": 2,     # sad      → sad
    "05": 0,     # angry    → angry
    "06": None,  # fearful  → 丢弃
    "07": None,  # disgust  → 丢弃
    "08": None,  # surprised→ 丢弃
}
EMOTION_LABELS = ["angry", "happy", "sad", "neutral"]
NUM_CLASSES = len(EMOTION_LABELS)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────
# 特征提取
# ──────────────────────────────────────────────
_mfcc_transform = None

def get_mfcc_transform():
    global _mfcc_transform
    if _mfcc_transform is None:
        _mfcc_transform = T.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": N_MELS},
        )
    return _mfcc_transform


def load_mfcc(wav_path):
    """读取WAV → MFCC (n_mfcc, T)"""
    sr, data = wavfile.read(wav_path)
    # int16 → float32
    data = data.astype(np.float32) / 32768.0
    # 多声道 → 单声道
    if data.ndim > 1:
        data = data.mean(axis=1)
    # 重采样（如果必要）
    if sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sr
        new_len = int(len(data) * ratio)
        data = np.interp(
            np.linspace(0, len(data) - 1, new_len),
            np.arange(len(data)),
            data
        )
    # 截断 / 补零
    if len(data) > MAX_LEN:
        data = data[:MAX_LEN]
    else:
        data = np.pad(data, (0, MAX_LEN - len(data)))

    waveform = torch.FloatTensor(data).unsqueeze(0)  # (1, T)
    mfcc = get_mfcc_transform()(waveform)             # (1, n_mfcc, frames)
    return mfcc.squeeze(0)                             # (n_mfcc, frames)


def collect_files(root_dir):
    """遍历RAVDESS目录，返回 [(wav_path, label_idx), ...]，跳过不需要的类"""
    items = []
    for actor_dir in sorted(os.listdir(root_dir)):
        actor_path = os.path.join(root_dir, actor_dir)
        if not os.path.isdir(actor_path):
            continue
        for fname in os.listdir(actor_path):
            if not fname.endswith(".wav"):
                continue
            parts = fname.replace(".wav", "").split("-")
            if len(parts) < 7:
                continue
            emotion_code = parts[2]
            label = EMOTION_MAP.get(emotion_code)
            if label is None:          # 丢弃 fearful/disgust/surprised
                continue
            items.append((os.path.join(actor_path, fname), label))
    return items


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class RAVDESSDataset(Dataset):
    def __init__(self, items, augment=False):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def _augment(self, mfcc):
        """简单数据增强：随机加噪 + 时间掩码"""
        # 随机高斯噪声
        if random.random() < 0.5:
            mfcc = mfcc + torch.randn_like(mfcc) * 0.015

        # 时间掩码（SpecAugment）
        if random.random() < 0.5:
            T = mfcc.shape[1]
            mask_len = random.randint(1, max(1, T // 8))
            mask_start = random.randint(0, T - mask_len)
            mfcc = mfcc.clone()
            mfcc[:, mask_start: mask_start + mask_len] = 0.0

        return mfcc

    def __getitem__(self, idx):
        wav_path, label = self.items[idx]
        mfcc = load_mfcc(wav_path)
        if self.augment:
            mfcc = self._augment(mfcc)
        return mfcc, label


# ──────────────────────────────────────────────
# 训练 & 验证
# ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 收集文件
    print("扫描RAVDESS数据集...")
    all_items = collect_files(RAVDESS_DIR)
    print(f"共 {len(all_items)} 个音频文件（已过滤），{NUM_CLASSES} 类情绪: {EMOTION_LABELS}")

    # ── 划分训练/验证集（按文件随机，stratify保证类别均衡）
    labels = [label for _, label in all_items]
    train_items, val_items = train_test_split(
        all_items, test_size=0.2, random_state=SEED, stratify=labels
    )
    print(f"训练集: {len(train_items)}  验证集: {len(val_items)}")

    # ── 数据加载
    train_ds = RAVDESSDataset(train_items, augment=True)
    val_ds   = RAVDESSDataset(val_items,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── 模型
    model = AudioEmotionModel(n_mfcc=N_MFCC, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    # 类别权重：neutral样本较少（96），其余多数类约192
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = torch.FloatTensor(1.0 / (class_counts + 1e-6)).to(device)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── 训练
    best_val_acc = 0.0
    print(f"\n开始训练，共 {EPOCHS} 轮...\n")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        flag = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            flag = "  ← best"

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.3f}{flag}")

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.3f}")
    print(f"模型已保存至: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
