from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import cv2
import base64
import numpy as np
from collections import Counter
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Optional

try:
    from imageio_ffmpeg import get_ffmpeg_exe
except Exception:
    get_ffmpeg_exe = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.video2frame import analyze_video
from preprocess.audio_extract import extract_audio
from preprocess.speech2text import speech_to_text
from models.face_model import predict_emotion
from models.audio_model import predict_emotion_from_audio
from models.text_model import predict_text_emotion
from models.audio_model import predict_emotion_from_audio_with_scores
from models.face_model import predict_emotion_with_scores
from utils.face_selector import crop_face, detect_faces, select_primary_face


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_dominant_emotion(timeline):
    emotions = [item.get("emotion") for item in timeline if isinstance(item, dict) and item.get("emotion")]
    if not emotions:
        return None
    return Counter(emotions).most_common(1)[0][0]


FUSION_LABELS = ["angry", "happy", "sad", "neutral"]
FUSION_WEIGHTS = {
    "video": 0.45,
    "audio": 0.35,
    "text": 0.20,
}


def _one_hot_scores(label: Optional[str], confidence: float) -> Dict[str, float]:
    scores = {k: 0.0 for k in FUSION_LABELS}
    if label in scores:
        scores[label] = max(0.0, min(1.0, float(confidence)))
    return scores


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    fixed = {k: float(scores.get(k, 0.0)) for k in FUSION_LABELS}
    total = sum(max(0.0, v) for v in fixed.values())
    if total <= 1e-8:
        return {k: 0.0 for k in FUSION_LABELS}
    return {k: round(max(0.0, v) / total, 6) for k, v in fixed.items()}


def fuse_modalities(video: Optional[Dict] = None, audio: Optional[Dict] = None, text: Optional[Dict] = None) -> Dict:
    weighted_scores = {k: 0.0 for k in FUSION_LABELS}
    active = []

    sources = {
        "video": video,
        "audio": audio,
        "text": text,
    }

    for name, payload in sources.items():
        if not payload:
            continue
        emotion = payload.get("emotion")
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        raw_scores = payload.get("scores")

        if isinstance(raw_scores, dict):
            score_map = _normalize_scores(raw_scores)
        else:
            score_map = _one_hot_scores(emotion, confidence if confidence > 0 else 1.0)

        weight = FUSION_WEIGHTS.get(name, 0.0) * max(0.0, min(1.0, confidence if confidence > 0 else 1.0))
        if weight <= 0.0:
            continue

        active.append({
            "modality": name,
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "weight": round(weight, 4),
        })

        for label in FUSION_LABELS:
            weighted_scores[label] += weight * float(score_map.get(label, 0.0))

    if not active:
        return {
            "emotion": None,
            "confidence": 0.0,
            "scores": {k: 0.0 for k in FUSION_LABELS},
            "activeModalities": [],
        }

    norm_scores = _normalize_scores(weighted_scores)
    best_emotion = max(norm_scores, key=lambda k: norm_scores[k])
    best_conf = float(norm_scores[best_emotion])

    return {
        "emotion": best_emotion,
        "confidence": round(best_conf, 4),
        "scores": norm_scores,
        "activeModalities": active,
    }


@app.route("/analyze_video", methods=["POST"])
def analyze_video_api():

    if "video" not in request.files:
        return jsonify({"error": "no video uploaded"}), 400

    video = request.files["video"]

    path = os.path.join(UPLOAD_FOLDER, video.filename)

    video.save(path)

    timeline = analyze_video(path)
    video_emotion = get_dominant_emotion(timeline)
    video_confidence = 0.0
    if timeline:
        total = len(timeline)
        matched = sum(1 for item in timeline if item.get("emotion") == video_emotion)
        video_confidence = (matched / total) if total > 0 else 0.0

    # ===== 音频处理 =====
    audio_emotion = None
    audio_error = None
    audio_result = None
    audio_path = os.path.join(UPLOAD_FOLDER, f"audio_{os.path.splitext(video.filename)[0]}.wav")
    try:
        extract_audio(path, audio_path)
        audio_result = predict_emotion_from_audio_with_scores(audio_path)
        audio_emotion = audio_result.get("emotion")
    except Exception as e:
        audio_error = str(e)
        audio_result = None

    # ===== 文本处理（语音转文本） =====
    text_emotion = None
    text_error = None
    text_result = None
    transcript = ""
    audio_path_for_st = None
    if audio_result is not None and os.path.exists(audio_path):
        # 复制音频文件用于语音转文本
        audio_path_for_st = os.path.join(UPLOAD_FOLDER, f"audio_st_{os.path.splitext(video.filename)[0]}.wav")
        try:
            import shutil
            shutil.copy(audio_path, audio_path_for_st)
            transcript = speech_to_text(audio_path_for_st)
            if transcript and transcript.strip():
                text_result = predict_text_emotion(transcript)
                text_emotion = text_result.get("emotion")
        except Exception as e:
            text_error = str(e)
            text_result = None
        finally:
            if audio_path_for_st and os.path.exists(audio_path_for_st):
                try:
                    os.remove(audio_path_for_st)
                except OSError:
                    pass

    # 清理音频文件
    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except OSError:
            pass

    fusion = fuse_modalities(
        video={
            "emotion": video_emotion,
            "confidence": round(video_confidence, 4),
            "scores": _one_hot_scores(video_emotion, video_confidence),
        },
        audio=audio_result,
        text=text_result,
    )

    return jsonify({
        "timeline": timeline,
        "videoEmotion": video_emotion,
        "videoConfidence": round(video_confidence, 4),
        "audioEmotion": audio_emotion,
        "audioConfidence": audio_result.get("confidence") if audio_result else 0.0,
        "textEmotion": text_emotion,
        "textConfidence": text_result.get("confidence") if text_result else 0.0,
        "transcript": transcript,
        "fusedEmotion": fusion.get("emotion"),
        "fusedConfidence": fusion.get("confidence", 0.0),
        "fusedScores": fusion.get("scores", {}),
        "fusionModalities": fusion.get("activeModalities", []),
        "audioError": audio_error,
        "textError": text_error,
    })


# 实时摄像头帧分析
@app.route("/analyze_frame", methods=["POST"])
def analyze_frame():
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "missing image payload"}), 400

    data = payload["image"]
    if not isinstance(data, str) or "," not in data:
        return jsonify({"error": "invalid image format"}), 400

    try:
        img_data = base64.b64decode(data.split(",", 1)[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "decode frame failed"}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        primary_face = select_primary_face(faces, gray.shape)

        if primary_face is None:
            return jsonify({
                "emotion": None,
                "faceDetected": False,
                "faceBox": None
            })

        face = crop_face(gray, primary_face)
        vision_result = predict_emotion_with_scores(face)
        emotion = vision_result.get("emotion")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    x, y, width, height = primary_face

    return jsonify({
        "emotion": emotion,
        "confidence": vision_result.get("confidence", 0.0),
        "scores": vision_result.get("scores", {}),
        "faceDetected": True,
        "faceBox": {
            "x": x,
            "y": y,
            "w": width,
            "h": height
        }
    })


@app.route("/analyze_audio_chunk", methods=["POST"])
def analyze_audio_chunk():
    if "audio" not in request.files:
        return jsonify({"error": "no audio uploaded"}), 400

    audio_file = request.files["audio"]

    input_suffix = Path(audio_file.filename or "chunk.webm").suffix or ".webm"
    input_tmp = tempfile.NamedTemporaryFile(suffix=input_suffix, delete=False)
    input_path = input_tmp.name
    input_tmp.close()

    output_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_path = output_tmp.name
    output_tmp.close()

    try:
        audio_file.save(input_path)

        ffmpeg_bin = "ffmpeg"
        if get_ffmpeg_exe is not None:
            try:
                ffmpeg_bin = get_ffmpeg_exe()
            except Exception:
                ffmpeg_bin = "ffmpeg"

        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            input_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            output_path,
        ]
        proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return jsonify({"error": f"ffmpeg convert failed: {proc.stderr.strip()}"}), 500

        audio_result = predict_emotion_from_audio_with_scores(output_path)

        return jsonify({
            "audioEmotion": audio_result.get("emotion"),
            "confidence": audio_result.get("confidence", 0.0),
            "scores": audio_result.get("scores", {}),
        })
    except FileNotFoundError:
        return jsonify({"error": "ffmpeg not found. Please install ffmpeg or ensure imageio-ffmpeg is available."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for p in (input_path, output_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    payload = request.get_json(silent=True)
    if not payload or "text" not in payload:
        return jsonify({"error": "missing text payload"}), 400

    text = payload.get("text", "")
    result = predict_text_emotion(text)

    if result.get("error") and result.get("confidence", 0.0) == 0.0:
        return jsonify({
            "error": result["error"],
            "textEmotion": result["emotion"],
            "confidence": result["confidence"],
            "scores": result["scores"],
        }), 500

    return jsonify({
        "textEmotion": result["emotion"],
        "confidence": result["confidence"],
        "scores": result["scores"],
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )