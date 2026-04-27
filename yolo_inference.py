from ultralytics import YOLO
model = YOLO("models/best.pt")

results = model.predict("input/215475.mp4", save=True)

 