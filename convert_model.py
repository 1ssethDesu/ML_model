from ultralytics import YOLO

MODEL_PATH = "app/models/model.pt"

model = YOLO(MODEL_PATH)

model.export(format='onnx', opset=12)