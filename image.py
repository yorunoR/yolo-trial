from ultralytics import YOLO
from collections import Counter

model = YOLO('yolov8n.pt')
res = model.predict('https://ultralytics.com/images/bus.jpg', save=True)
# print(model.names)
# print(res[0].boxes)
cls = res[0].names
counts = Counter(res[0].boxes.cls.tolist())
for number, count in counts.items():
    print(f"{number}は{cls[number]}で、個数は{count}個")
