# test_audio.py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.audio_model import predict_emotion_from_audio

# 用RAVDESS数据集的第一个音频测试
test_audio = "data/RAVDESS/03-01-07-02-01-01-01.wav"

if not os.path.exists(test_audio):
    # 找任意一个存在的音频
    for root, dirs, files in os.walk("data/RAVDESS"):
        for f in files:
            if f.endswith(".wav"):
                test_audio = os.path.join(root, f)
                break
        if os.path.exists(test_audio):
            break

print(f"测试音频: {test_audio}")

emotion = predict_emotion_from_audio(test_audio)
print(f"预测情绪: {emotion}")