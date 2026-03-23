import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile
import torchaudio.transforms as T


class AudioEmotionModel(nn.Module):
    """
    CNN + BiGRU 语音情绪识别模型
    输入: MFCC特征 (batch, n_mfcc, time_frames)
    输出: 情绪 logits (batch, num_classes)
    """

    def __init__(self, n_mfcc=40, num_classes=8):
        super().__init__()

        # CNN: 提取局部时频特征
        self.conv = nn.Sequential(
            nn.Conv1d(n_mfcc, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # BiGRU: 捕捉时序依赖
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, n_mfcc, T)
        x = self.conv(x)           # (B, 128, T/4)
        x = x.permute(0, 2, 1)    # (B, T/4, 128)
        x, _ = self.gru(x)        # (B, T/4, 256)
        x = x.mean(dim=1)         # 全局平均池化 (B, 256)
        return self.classifier(x)  # (B, num_classes)


# ──────────────────────────────────────────────
# 推理封装（新增）
# ──────────────────────────────────────────────
LABELS = ["angry", "happy", "sad", "neutral"]

# 配置（与训练保持一致）
SAMPLE_RATE = 16000
MAX_DURATION = 4.0
MAX_LEN = int(SAMPLE_RATE * MAX_DURATION)
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
_model = None
_mfcc_transform = None


def _get_mfcc_transform():
    global _mfcc_transform
    if _mfcc_transform is None:
        _mfcc_transform = T.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": N_MELS},
        )
    return _mfcc_transform


def _load_audio(audio_path):
    """读取音频文件 → MFCC特征"""
    sr, data = wavfile.read(audio_path)
    # int16 → float32
    data = data.astype(np.float32) / 32768.0
    # 多声道 → 单声道
    if data.ndim > 1:
        data = data.mean(axis=1)
    # 重采样
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

    waveform = torch.FloatTensor(data).unsqueeze(0)
    mfcc = _get_mfcc_transform()(waveform)
    return mfcc.squeeze(0)


def _get_model():
    global _model
    if _model is None:
        _model = AudioEmotionModel(n_mfcc=N_MFCC, num_classes=len(LABELS)).to(device)
        model_path = "models/audio_emotion_model.pth"
        if os.path.exists(model_path):
            _model.load_state_dict(torch.load(model_path, map_location=device))
            _model.eval()
            print(f"Loaded audio model from {model_path}")
        else:
            print(f"Warning: {model_path} not found, using random weights")
    return _model


import os


def predict_emotion_from_audio(audio_path) -> str:
    """
    输入: 音频文件路径(.wav)
    输出: 情绪标签 (str)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    mfcc = _load_audio(audio_path).unsqueeze(0).to(device)  # (1, n_mfcc, T)
    model = _get_model()
    
    with torch.no_grad():
        logits = model(mfcc)
        pred_idx = logits.argmax(1).item()
    
    return LABELS[pred_idx]


def predict_emotion_from_audio_with_scores(audio_path) -> dict:
    """
    输入: 音频文件路径(.wav)
    输出: {emotion, confidence, scores}
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    mfcc = _load_audio(audio_path).unsqueeze(0).to(device)
    model = _get_model()

    with torch.no_grad():
        logits = model(mfcc)
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()

    pred_idx = int(np.argmax(probs))
    scores = {label: round(float(probs[i]), 6) for i, label in enumerate(LABELS)}
    return {
        "emotion": LABELS[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "scores": scores,
    }


def predict_emotion_from_array(audio_array: np.ndarray, sample_rate: int = 16000) -> str:
    """
    输入: numpy数组 (samples,) 或 (samples, channels)
    输出: 情绪标签 (str)
    """
    global SAMPLE_RATE
    SAMPLE_RATE = sample_rate
    
    # 处理输入
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    audio_array = audio_array.astype(np.float32)
    
    # 归一化到 [-1, 1]
    max_val = np.abs(audio_array).max()
    if max_val > 1.0:
        audio_array = audio_array / max_val
    
    # 截断/补零
    max_len = int(SAMPLE_RATE * MAX_DURATION)
    if len(audio_array) > max_len:
        audio_array = audio_array[:max_len]
    else:
        audio_array = np.pad(audio_array, (0, max_len - len(audio_array)))
    
    waveform = torch.FloatTensor(audio_array).unsqueeze(0)
    mfcc = _get_mfcc_transform()(waveform).squeeze(0).unsqueeze(0).to(device)
    
    model = _get_model()
    with torch.no_grad():
        logits = model(mfcc)
        pred_idx = logits.argmax(1).item()
    
    return LABELS[pred_idx]
