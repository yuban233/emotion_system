"""
语音转文本模块
使用 Whisper 模型进行语音识别
"""

import os
import torch

# 默认使用轻量级模型以保证推理速度
DEFAULT_MODEL = "base"
_model = None
_processor = None


def _load_model():
    """加载 Whisper 模型"""
    global _model, _processor

    if _model is not None:
        return _model, _processor

    try:
        import whisper
    except ImportError:
        raise ImportError(
            "whisper is not installed. Please install with: pip install openai-whisper"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model on {device}...")

    _model = whisper.load_model(DEFAULT_MODEL, device=device)
    _processor = None  # Whisper 不需要processor

    print(f"Whisper model loaded successfully")
    return _model, _processor


def speech_to_text(audio_path: str) -> str:
    """
    将语音文件转换为文本

    Args:
        audio_path: 音频文件路径 (.wav, .mp3 等)

    Returns:
        识别的文本内容
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 检查文件是否为空
    if os.path.getsize(audio_path) == 0:
        return ""

    model, _ = _load_model()

    # 语音识别
    try:
        result = model.transcribe(
            audio_path,
            language="zh",  # 默认中文，可根据需要修改
            fp16=False,     # CPU上不使用fp16
            verbose=False
        )
        return result.get("text", "").strip()
    except Exception as e:
        print(f"Speech to text error: {e}")
        return ""


def speech_to_text_en(audio_path: str) -> str:
    """
    将语音文件转换为英文文本

    Args:
        audio_path: 音频文件路径

    Returns:
        识别的英文文本内容
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if os.path.getsize(audio_path) == 0:
        return ""

    model, _ = _load_model()

    try:
        result = model.transcribe(
            audio_path,
            language="en",
            fp16=False,
            verbose=False
        )
        return result.get("text", "").strip()
    except Exception as e:
        print(f"Speech to text error: {e}")
        return ""


if __name__ == "__main__":
    # 测试
    import sys

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"Processing: {audio_file}")
        text = speech_to_text(audio_file)
        print(f"Result: {text}")
    else:
        print("Usage: python speech2text.py <audio_file>")
        print("Example: python speech2text.py audio.wav")
