from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import uvicorn
import cv2
from typing import List
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(title="Dental X-Ray Detection API")

# Enable CORS (For frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO Model
try:
    model = YOLO('app/models/model.pt')  # Ensure correct path
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

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
    image = Image.open(io.BytesIO(contents))
    
    # Convert PIL Image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Run YOLO Model Prediction
    results = model.predict(image)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        confidence = float(box.conf[0])
        class_name = results[0].names[int(box.cls[0])]

        # Draw bounding box
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        
        # Put label above bounding box
        cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save detection info
        detections.append(Detection(class_name=class_name, confidence=confidence, bbox=[x1, y1, x2, y2]))

    # Convert OpenCV image back to bytes
    _, encoded_img = cv2.imencode(".jpg", image_cv)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
