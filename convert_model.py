from ultralytics import YOLO

MODEL_PATH = "app/models/model.pt"

model = YOLO(MODEL_PATH)

model.export(format='onnx', simplify=False, dynamic=True, optimize=False)