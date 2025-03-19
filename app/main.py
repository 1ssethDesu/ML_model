from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import numpy as np
import io
import uvicorn
import cv2
import onnxruntime
from typing import List
from pydantic import BaseModel
import os
import download_model
from mangum import Mangum

from util import helper

# ONNX Model Path
MODEL_PATH = "app/models/model.onnx"

# Try to load the model
try:
    if not os.path.exists(MODEL_PATH):
        download_model.download(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Initialize ONNX Runtime Session
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if onnxruntime.get_device() == 'GPU' else ['CPUExecutionProvider']
session = onnxruntime.InferenceSession(MODEL_PATH, providers=providers)

device = "GPU" if 'CUDAExecutionProvider' in session.get_providers() else "CPU"
print(f"Using {device} for inference.")
print("Model loaded successfully!")

# Class names
CLASSES = ['gum-disease', 'tooth-decay', 'tooth-loss']

# Initialize FastAPI
app = FastAPI(title="Dental X-Ray Detection API")
handler = Mangum(app)

# Enable CORS (For frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Model for API Response
class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]

class PredictionResponse(BaseModel):
    detections: List[Detection]

@app.get("/")
def read_root():
    return {"message": "Welcome to Dental X-Ray Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Process the uploaded image, detect objects, and return the image with bounding boxes."""
    contents = await file.read()
    
    # Convert to OpenCV format
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process image
    img, img_h, img_w = helper.image_process(image)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img})
    
    # Filter detections
    results = helper.filter_detection(output)
    
    # Get rescaled results
    rescaled_results, confidences = helper.rescale_back(results, img_w, img_h)
    
    # Draw results on image and prepare response
    detections = []
    
    for res, conf in zip(rescaled_results, confidences):
        x1, y1, x2, y2, cls_id = res
        cls_id = int(cls_id)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_name = CLASSES[cls_id]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw label
        label = f"{class_name}:{conf:.2f}"
        cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add to detections list
        detections.append(Detection(
            class_name=class_name,
            confidence=float(conf),
            bbox=[x1, y1, x2, y2]
        ))
    
    # Convert OpenCV image back to bytes
    _, encoded_img = cv2.imencode(".jpg", image)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

@app.post("/predict_json")
async def predict_json(file: UploadFile = File(...)):
    """Process the uploaded image and return JSON with detection information only."""
    contents = await file.read()
    
    # Convert to OpenCV format
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process image
    img, img_h, img_w = helper.image_process(image)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img})
    
    # Filter detections
    results = helper.filter_detection(output)
    
    # Get rescaled results
    rescaled_results, confidences = helper.rescale_back(results, img_w, img_h)
    
    # Prepare response
    detections = []
    for res, conf in zip(rescaled_results, confidences):
        x1, y1, x2, y2, cls_id = res
        cls_id = int(cls_id)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_name = CLASSES[cls_id]
        
        detections.append(Detection(
            class_name=class_name,
            confidence=float(conf),
            bbox=[x1, y1, x2, y2]
        ))
    
    return PredictionResponse(detections=detections)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)