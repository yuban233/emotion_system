import cv2
import torch
from models.face_model import FaceEmotionModel

labels = ["angry","happy","sad","neutral"]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = FaceEmotionModel()
model.load_state_dict(torch.load("face_emotion_model.pth", map_location=device))
model.to(device)
model.eval()

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face,(48,48))

        face = face / 255.0

        face = torch.tensor(face).float().unsqueeze(0).unsqueeze(0).to(device)

        output = model(face)

        pred = torch.argmax(output)

        emotion = labels[pred]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.putText(
            frame,
            emotion,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Emotion Recognition",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()