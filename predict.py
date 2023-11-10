from ultralytics import YOLO

# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('./runs/detect/train/weights/best.pt')  # load a custom training model (recommended for training)

print("model load 완료")

results = model(source='./test/images', save=True)
# results = model(source='./tempTest', save=True)