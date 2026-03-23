import sys
import os
# 获取项目根目录（EMOTION_SYSTEM 文件夹路径）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import json
import csv
from collections import Counter, deque
import matplotlib.pyplot as plt

from models.face_model import predict_emotion
from utils.face_selector import crop_face, detect_faces, select_primary_face


# ========================
# 视频分析
# ========================

def analyze_video(video_path, frame_step=10, smooth_window=5):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25

    frame_id = 0

    results = []

    smooth_buffer = deque(maxlen=smooth_window)
    previous_face = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # 每N帧预测一次
        if frame_id % frame_step != 0:

            frame_id += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detect_faces(gray)

        primary_face = select_primary_face(faces, gray.shape, previous_face=previous_face)

        if primary_face is not None:

            previous_face = primary_face

            face = crop_face(gray, primary_face)

            emotion = predict_emotion(face)

            # ===== 情绪平滑 =====

            smooth_buffer.append(emotion)

            smooth_emotion = Counter(smooth_buffer).most_common(1)[0][0]

            time = frame_id / fps

            results.append({
                "time": round(time,2),
                "emotion": smooth_emotion
            })
        else:
            previous_face = None

        frame_id += 1

    cap.release()

    return results


# ========================
# 保存JSON
# ========================

def save_json(data, path):

    with open(path,"w") as f:
        json.dump(data,f,indent=4)


# ========================
# 保存CSV
# ========================

def save_csv(data, path):

    with open(path,"w",newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["time","emotion"])

        for d in data:

            writer.writerow([d["time"], d["emotion"]])


# ========================
# 情绪统计图
# ========================

def plot_distribution(data):

    emotions = [d["emotion"] for d in data]

    counter = Counter(emotions)

    plt.figure()

    plt.bar(counter.keys(), counter.values())

    plt.title("Emotion Distribution")

    plt.xlabel("Emotion")

    plt.ylabel("Count")

    os.makedirs("output",exist_ok=True)

    plt.savefig("output/emotion_distribution.png")

    plt.close()


# ========================
# 主函数
# ========================

if __name__ == "__main__":

    video_path = "d:\\xwechat_files\\wxid_2lxle93fqdsh22_adc0\\msg\\\\video\\\\2024-01\\\\548062b362b5bc6a2dc32d96ae36ed92.mp4"

    os.makedirs("output",exist_ok=True)

    timeline = analyze_video(video_path)

    save_json(timeline,"output/emotion_timeline.json")

    save_csv(timeline,"output/emotion_timeline.csv")

    plot_distribution(timeline)

    print("分析完成")

# import cv2
# import os

# def video_to_frames(video_path, output_folder, frame_rate=5):

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     cap = cv2.VideoCapture(video_path)

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     interval = int(fps / frame_rate)

#     frame_count = 0
#     saved_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_count % interval == 0:
#             frame_name = os.path.join(output_folder, f"frame_{saved_count}.jpg")
#             cv2.imwrite(frame_name, frame)
#             saved_count += 1

#         frame_count += 1

#     cap.release()
#     print("frames saved:", saved_count)


# if __name__ == "__main__":
#     video_to_frames("test.mp4", "frames")