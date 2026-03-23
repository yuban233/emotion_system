"""
消融实验测试脚本
用于测试各模态及融合方法的识别效果
"""

import json
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.audio_model import predict_emotion_from_audio_with_scores
from models.text_model import predict_text_emotion
from preprocess.audio_extract import extract_audio
from preprocess.speech2text import speech_to_text
from preprocess.video2frame import analyze_video


def test_single_modality(video_path, audio_path=None, transcript=""):
    """测试单个模态的识别效果"""
    results = {}

    # 1. 视觉模态
    print("Testing visual modality...")
    try:
        timeline = analyze_video(video_path)
        if timeline:
            emotions = [item["emotion"] for item in timeline]
            dominant = Counter(emotions).most_common(1)[0][0]
            results["visual"] = {
                "dominant_emotion": dominant,
                "timeline": timeline,
                "distribution": dict(Counter(emotions)),
            }
        else:
            results["visual"] = {
                "dominant_emotion": None,
                "error": "No faces detected",
            }
    except Exception as e:
        results["visual"] = {"error": str(e)}

    # 2. 音频模态
    print("Testing audio modality...")
    if audio_path and os.path.exists(audio_path):
        try:
            audio_result = predict_emotion_from_audio_with_scores(audio_path)
            results["audio"] = audio_result
        except Exception as e:
            results["audio"] = {"error": str(e)}
    else:
        results["audio"] = {"error": "No audio file"}

    # 3. 文本模态
    print("Testing text modality...")
    if transcript:
        try:
            text_result = predict_text_emotion(transcript)
            results["text"] = text_result
        except Exception as e:
            results["text"] = {"error": str(e)}
    else:
        results["text"] = {"error": "No transcript"}

    return results


def test_fusion_methods(results):
    """测试不同的融合方法"""
    fusion_results = {}

    # 1. 只使用视觉
    if "visual" in results and results["visual"].get("dominant_emotion"):
        fusion_results["visual_only"] = results["visual"]["dominant_emotion"]

    # 2. 只使用音频
    if "audio" in results and results["audio"].get("emotion"):
        fusion_results["audio_only"] = results["audio"]["emotion"]

    # 3. 只使用文本
    if "text" in results and results["text"].get("emotion"):
        fusion_results["text_only"] = results["text"]["emotion"]

    # 4. 视觉+音频融合
    if "visual" in results and "audio" in results:
        visual_emotion = results["visual"].get("dominant_emotion")
        audio_emotion = results["audio"].get("emotion")
        if visual_emotion and audio_emotion:
            votes = Counter([visual_emotion, audio_emotion])
            fusion_results["visual_audio"] = votes.most_common(1)[0][0]

    # 5. 视觉+文本融合
    if "visual" in results and "text" in results:
        visual_emotion = results["visual"].get("dominant_emotion")
        text_emotion = results["text"].get("emotion")
        if visual_emotion and text_emotion:
            votes = Counter([visual_emotion, text_emotion])
            fusion_results["visual_text"] = votes.most_common(1)[0][0]

    # 6. 音频+文本融合
    if "audio" in results and "text" in results:
        audio_emotion = results["audio"].get("emotion")
        text_emotion = results["text"].get("emotion")
        if audio_emotion and text_emotion:
            votes = Counter([audio_emotion, text_emotion])
            fusion_results["audio_text"] = votes.most_common(1)[0][0]

    # 7. 三模态融合
    if all(
        [
            results.get("visual", {}).get("dominant_emotion"),
            results.get("audio", {}).get("emotion"),
            results.get("text", {}).get("emotion"),
        ]
    ):
        all_emotions = [
            results["visual"]["dominant_emotion"],
            results["audio"]["emotion"],
            results["text"]["emotion"],
        ]
        votes = Counter(all_emotions)
        fusion_results["trimodal"] = votes.most_common(1)[0][0]

    return fusion_results


def run_ablation_experiment(video_path, sample_name="sample"):
    """运行完整的消融实验"""
    print(f"\n{'=' * 50}")
    print(f"Running ablation experiment for: {sample_name}")
    print(f"{'=' * 50}\n")

    # 提取音频
    audio_path = f"uploads/temp_audio_{sample_name}.wav"
    if not os.path.exists(audio_path):
        try:
            extract_audio(video_path, audio_path)
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            audio_path = None

    # 语音转文本
    transcript = ""
    if audio_path and os.path.exists(audio_path):
        try:
            transcript = speech_to_text(audio_path)
            print(f"Transcript: {transcript}")
        except Exception as e:
            print(f"Speech to text failed: {e}")

    # 测试单模态
    print("\n--- Single Modality Results ---")
    single_results = test_single_modality(video_path, audio_path, transcript)
    for modality, result in single_results.items():
        print(
            f"  {modality}: "
            f"{result.get('dominant_emotion') or result.get('emotion') or result.get('error')}"
        )

    # 测试融合方法
    print("\n--- Fusion Methods Results ---")
    fusion_results = test_fusion_methods(single_results)
    for method, emotion in fusion_results.items():
        print(f"  {method}: {emotion}")

    # 保存结果
    output = {
        "sample": sample_name,
        "single_modality": single_results,
        "fusion_methods": fusion_results,
    }

    output_dir = "output/ablation"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{sample_name}_ablation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")

    # 清理临时文件
    if audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except Exception:
            pass

    return output


if __name__ == "__main__":
    # 可以传入多个视频进行批量测试
    test_videos = [
        "uploads/548062b362b5bc6a2dc32d96ae36ed92.mp4",
    ]

    for video_path in test_videos:
        if os.path.exists(video_path):
            sample_name = os.path.splitext(os.path.basename(video_path))[0]
            run_ablation_experiment(video_path, sample_name)
        else:
            print(f"Video not found: {video_path}")
