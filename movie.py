import cv2
from ultralytics import YOLO
from collections import Counter

cap = cv2.VideoCapture("./854100-hd_1920_1080_25fps.mp4")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# print(width)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(height)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# print(frames)
fps = cap.get(cv2.CAP_PROP_FPS)
# print(fps)
# print(frames/fps)

model = YOLO('yolov8n.pt')

writer = cv2.VideoWriter('./result.mp4', cv2.VideoWriter_fourcc(*'MP4V',), fps, frameSize=(int(width), int(height)))

cls = model.names

num = 0
while cap.isOpened():
    ret, img = cap.read()
    if ret:
        results = model(img, conf=0.5, verbose=False)
        img = results[0].plot(labels=True, conf=True)
        writer.write(img)
    else:
        writer.release()
        break

cap.release()
