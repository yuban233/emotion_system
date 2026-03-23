import math
import cv2


face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(gray_frame, scale_factor=1.3, min_neighbors=5):
    faces = face_detector.detectMultiScale(gray_frame, scale_factor, min_neighbors)
    return [tuple(map(int, face)) for face in faces]


def select_primary_face(faces, frame_shape, previous_face=None):
    if not faces:
        return None

    frame_height, frame_width = frame_shape[:2]
    frame_center_x = frame_width / 2.0
    frame_center_y = frame_height / 2.0
    max_center_distance = math.hypot(frame_center_x, frame_center_y) or 1.0

    best_face = None
    best_score = -1.0

    for face in faces:
        x, y, width, height = face
        face_area = width * height
        area_ratio = face_area / float(frame_width * frame_height)

        face_center_x = x + width / 2.0
        face_center_y = y + height / 2.0
        center_distance = math.hypot(face_center_x - frame_center_x, face_center_y - frame_center_y)
        center_score = max(0.0, 1.0 - (center_distance / max_center_distance))

        continuity_score = _intersection_over_union(face, previous_face) if previous_face else 0.0

        score = (0.55 * area_ratio) + (0.30 * center_score) + (0.15 * continuity_score)

        if score > best_score:
            best_score = score
            best_face = face

    return best_face


def crop_face(gray_frame, face_box):
    x, y, width, height = face_box
    return gray_frame[y:y + height, x:x + width]


def _intersection_over_union(face_a, face_b):
    if not face_a or not face_b:
        return 0.0

    ax1, ay1, aw, ah = face_a
    bx1, by1, bw, bh = face_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / float(union_area)