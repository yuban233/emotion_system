import torch
import cv2
from models.face_model import FaceEmotionModel

labels = ["angry","happy","sad","neutral"]

model = FaceEmotionModel()
model.load_state_dict(torch.load("face_emotion_model.pth"))
model.eval()

img = cv2.imread("d:\FacialExpressionRecognition\dataset\\fer2013\PublicTest\sad\\28802.jpg",0)

img = cv2.resize(img,(48,48))

img = img/255.0

img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)

output = model(img)

pred = torch.argmax(output)

print("emotion:",labels[pred])