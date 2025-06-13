# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse, StreamingResponse
# import numpy as np
# import io
# import uvicorn
# import cv2
# import onnxruntime
# from typing import List
# from pydantic import BaseModel
# import os
# import base64
# import download_model
# from mangum import Mangum

# from util import helper

# # ONNX Model Path
# MODEL_PATH = "ML_model/app/models/model.pth"

# # Try to load the model
# try:
#     if not os.path.exists(MODEL_PATH):
#         download_model.download(MODEL_PATH)
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     raise

# # Initialize ONNX Runtime Session
# providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if onnxruntime.get_device() == 'GPU' else ['CPUExecutionProvider']
# session = onnxruntime.InferenceSession(MODEL_PATH, providers=providers)

# device = "GPU" if 'CUDAExecutionProvider' in session.get_providers() else "CPU"
# print(f"Using {device} for inference.")
# print("Model loaded successfully!")

# # Class names
# CLASSES = ['gum-disease', 'tooth-decay', 'tooth-loss']

# # Define colors for different diseases
# disease_colors = {
#     'tooth-decay': (0, 0, 255),      # Red
#     'gum-disease': (0, 255, 0),      # Green
#     'tooth-loss': (255, 0, 0),       # Blue
#     'cavity': (0, 165, 255),         # Orange
#     'gingivitis': (255, 0, 255),     # Magenta
#     'tartar': (0, 255, 255),         # Yellow
#     'abscess': (128, 0, 128),        # Purple
#     'cyst': (255, 255, 0),           # Cyan
#     'fracture': (255, 128, 0),       # Light Blue
#     'impaction': (0, 128, 255),      # Orange-Red
# }

# # Default color if disease not in predefined colors
# default_color = (255, 255, 255)

# # Initialize FastAPI
# app = FastAPI(title="Dental X-Ray Detection API")
# handler = Mangum(app)

# # Enable CORS (For frontend communication)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change this to specific domains in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Pydantic Model for API Response
# class Detection(BaseModel):
#     class_name: str
#     confidence: float
#     bbox: List[float]

# class PredictionResponse(BaseModel):
#     detections: List[Detection]

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to Dental X-Ray Detection API"}

# @app.post("/v1/predict")
# async def predict(file: UploadFile = File(...)):
#     """Process the uploaded image, detect objects, and return the image with bounding boxes."""
#     contents = await file.read()
    
#     # Convert to OpenCV format
#     nparr = np.frombuffer(contents, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#     # Process image
#     img, img_h, img_w = helper.image_process(image)
    
#     # Run inference
#     input_name = session.get_inputs()[0].name
#     output = session.run(None, {input_name: img})
    
#     # Filter detections
#     results = helper.filter_detection(output)
    
#     # Get rescaled results
#     rescaled_results, confidences = helper.rescale_back(results, img_w, img_h)
    
#     # Draw results on image and prepare response
#     detections = []
    
#     for res, conf in zip(rescaled_results, confidences):
#         x1, y1, x2, y2, cls_id = res
#         cls_id = int(cls_id)
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         class_name = CLASSES[cls_id]
        
#         # Draw bounding box
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
#         # Draw label
#         label = f"{class_name}:{conf:.2f}"
#         cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         # Add to detections list
#         detections.append(Detection(
#             class_name=class_name,
#             confidence=float(conf),
#             bbox=[x1, y1, x2, y2]
#         ))
    
#     # Convert OpenCV image back to bytes
#     _, encoded_img = cv2.imencode(".jpg", image)
#     return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

# @app.post("/v2/predict")
# async def predict(file: UploadFile = File(...)):
#     """Process the uploaded image, detect objects, and return the image with bounding boxes."""
#     contents = await file.read()
    
#     # Convert to OpenCV format
#     nparr = np.frombuffer(contents, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#     # Process image
#     img, img_h, img_w = helper.image_process(image)
    
#     # Run inference
#     input_name = session.get_inputs()[0].name
#     output = session.run(None, {input_name: img})
    
#     # Filter detections
#     results = helper.filter_detection(output)
    
#     # Get rescaled results
#     rescaled_results, confidences = helper.rescale_back(results, img_w, img_h)
    
#     # Draw results on image and prepare response
#     detections = []

#     # Disease information - use set to track unique diseases
#     unique_diseases = set()
#     disease_info = []
    
#     for res, conf in zip(rescaled_results, confidences):
#         x1, y1, x2, y2, cls_id = res
#         cls_id = int(cls_id)
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         class_name = CLASSES[cls_id]


        
#         # Get color for this disease
#         color = disease_colors.get(class_name, default_color)
        
#         # Draw bounding box with disease-specific color
#         cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
#         # Draw confidence score
#         label = f"{conf:.2f}"
#         cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
#         # Add to detections list
#         detections.append(Detection(
#             class_name=class_name,
#             confidence=float(conf),
#             bbox=[x1, y1, x2, y2]
#         ))

#         # Only add disease info if we haven't seen this disease before
#         if class_name not in unique_diseases:
#             unique_diseases.add(class_name)
#             info = helper.get_disease_info(class_name)
#             disease_info.append({
#                 "class_name": class_name,
#                 "description": info.get("description", "No description available"),
#                 "symptoms": info.get("symptoms", "No symptoms information available"),
#                 "causes": info.get("causes", "No causes information available"),
#                 "treatment": info.get("treatment", "No treatment information available"),
#                 "prevention": info.get("prevention", "No prevention information available"),
#                 "color": f"rgb({color[2]}, {color[1]}, {color[0]})"  # Convert BGR to RGB for frontend
#             })
    
#     # Convert OpenCV image back to bytes
#     _, encoded_img = cv2.imencode(".jpg", image)
#     base64_img = base64.b64encode(encoded_img.tobytes()).decode('utf-8')

#     return JSONResponse(content={
#         "message": "Prediction successful",
#         "image": base64_img,
#         "disease_info": disease_info
#     })

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import torch
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont  # Import ImageDraw and ImageFont
import io
from torchvision.transforms import functional as F
from app.model import get_model # Import the function to get the loaded model
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager # IMPORTANT: Keep this import
import base64
# torchvision.transforms is still useful for initial image tensor conversion
import torchvision.transforms as T 
import logging # Import logging for better error handling
from app.util.helper import get_disease_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global font variables ---
# IMPORTANT: Provide a valid path to your font file.
# If no font is found, PIL will fall back to a default, but it might not look good.
FONT_PATH = "arialbd.ttf" # Replace with your actual font path, e.g., "fonts/Inter-Regular.ttf"

try:
    # Attempt to load a common font for scores
    score_font = ImageFont.truetype(FONT_PATH, 20) # Adjust font size as needed
    logger.info(f"Successfully loaded font from {FONT_PATH}")
except IOError:
    logger.warning(f"Could not load font from {FONT_PATH}. Using default PIL font. "
                   "Ensure the font file exists and the path is correct.")
    score_font = ImageFont.load_default()
except Exception as e:
    logger.error(f"An unexpected error occurred loading font: {e}. Using default PIL font.")
    score_font = ImageFont.load_default()


# --- Lifespan Context Manager for FastAPI Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model... This might take a moment.")
    try:
        get_model() 
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.exception(f"Failed to load model during startup: {e}") 
        raise RuntimeError("Application startup failed due to model loading error. Check logs for details.") from e
    yield
    logger.info("Application is shutting down.")

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Consider changing to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for API Response ---
class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]

class DiseaseInfo(BaseModel):
    # Note: Column names from your CSV are used here
    disease: str
    description: str
    causes: str
    symptoms: str
    treatment: str
    prevention: str

class PredictionResponse(BaseModel):
    filename: str
    predictions: List[Detection]
    image: str # Base64 encoded image
    disease_info: List[DiseaseInfo] # List of unique disease information


@app.get("/")
async def root():
    return {"message": "Welcome to the Faster R-CNN Inference API!"}

@app.post("/v2/predict/")
async def predict_object_detection(file: UploadFile = File(...)):
    """
    Performs object detection on an uploaded image, draws bounding boxes
    and confidence scores, and returns the processed image as a base64
    encoded string along with prediction details.
    """
    if not file.content_type.startswith("image/"):
        logger.warning(f"Uploaded file '{file.filename}' is not an image (Content-Type: {file.content_type}).")
        raise HTTPException(status_code=400, detail="Uploaded file is not an image. Please upload a valid image file (e.g., JPEG, PNG).")

    try:
        # 1. Read and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info(f"Received image '{file.filename}' for prediction.")

        # 2. Get the globally loaded model instance
        model = get_model()
        
        # 3. Prepare image for model input (convert PIL Image to PyTorch Tensor)
        img_tensor = F.to_tensor(image)
        input_batch = [img_tensor]

        # 4. Perform inference
        model.eval() # Ensure model is in evaluation mode
        with torch.no_grad():
            prediction = model(input_batch)
        logger.info("Inference completed.")

        # 5. Define your model's class label mapping (CRUCIAL!)
        idx_to_class_name = {
            0: "background", 
            1: "gum-disease",
            2: "tooth-decay",
            3: "tooth-loss"
            # IMPORTANT: Confirm these IDs match your model's actual output labels
        }
        
        # Define colors for bounding boxes per class (RGB format for PIL)
        class_colors = {
            "gum-disease": (0, 255, 0),  # Green
            "tooth-decay": (255, 0, 0),  # Red
            "tooth-loss": (0, 0, 255),   # Blue
            "unknown": (255, 255, 0)     # Yellow (fallback)
        }
        
        # 6. Initialize results list for JSON response
        results = []


        # Disease information - use set to track unique diseases
        unique_diseases = set()
        disease_info_list = []
        
            
        # Create a drawing object for the PIL Image
        draw = ImageDraw.Draw(image)
        
        # Define confidence threshold (can be made configurable)
        CONFIDENCE_THRESHOLD = 0.7 # Adjust as needed

        # 7. Process detections if any are found
        if prediction and len(prediction) > 0:
            pred_data = prediction[0] # Assuming single image batch
            boxes = pred_data["boxes"].cpu()
            labels = pred_data["labels"].cpu()
            scores = pred_data["scores"].cpu()

            # Filter detections by a confidence threshold
            keep = scores > CONFIDENCE_THRESHOLD 
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            logger.info(f"Filtered {len(boxes)} detections above confidence threshold {CONFIDENCE_THRESHOLD}.")

            # 8. Draw bounding boxes and scores using PIL
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = [int(coord) for coord in box.tolist()] # Convert to int for drawing
                class_name = idx_to_class_name.get(int(label), f"unknown_label_{int(label)}")
                
                # Add to the JSON results
                results.append(Detection(
                    class_name=class_name,
                    confidence=float(score.item()), # Changed 'score' to 'confidence'
                    bbox=[float(x1), float(y1), float(x2), float(y2)] # Changed 'box' to 'bbox'
                ))
                
                # Get the color based on the class name
                box_color = class_colors.get(class_name, class_colors["unknown"])
                
                # Draw the bounding box
                draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=4) # Width 3 for visibility

                # Prepare score text
                score_text = f"{score.item():.2f}" # Format score to two decimal places (e.g., 0.68)

                # --- Draw Score (above the box) ---
                # Position for score text: top-left of the box
                score_display_x = x1 
                
                # Calculate text dimensions
                score_bbox = draw.textbbox((score_display_x, y1), score_text, font=score_font)
                score_text_width = score_bbox[2] - score_bbox[0]
                score_text_height = score_bbox[3] - score_bbox[1]
                
                # Adjust y-position for score text to be above the box
                # Add a small padding (e.g., 5 pixels) between the text and the box
                score_bg_y = y1 - score_text_height - 5 

                # Ensure the text doesn't go off the top of the image
                if score_bg_y < 0: 
                    score_bg_y = 0 # Place at the very top if it would go off
                    score_display_y = score_bg_y + 2 # Add a little padding from the top edge
                else:
                    score_display_y = score_bg_y + 2 # General padding within the background rectangle

                

                # Draw the score text, using the box_color for fill
                draw.text((score_display_x + 2, score_display_y), score_text, fill=box_color, font=score_font) # Small padding for text
                
                if class_name != "background" and class_name not in unique_diseases:
                    unique_diseases.add(class_name)
                    info = get_disease_info(class_name) # Call the imported helper function
                    # Only add if the disease info was found (no error)
                    if "error" not in info:
                        # Ensure 'disease' key is present, which is expected by DiseaseInfo Pydantic model
                        # If 'disease' key is missing from info, it will cause a Pydantic validation error.
                        if 'disease' in info:
                            disease_info_list.append(DiseaseInfo(**info)) 
                        else:
                            logger.warning(f"Disease info for '{class_name}' retrieved without 'disease' key: {info}")
                    else:
                        logger.warning(f"Disease info not found for '{class_name}': {info['error']}")
        else:
            logger.info("No significant detections found for the uploaded image above the confidence threshold.")


        # 9. Convert the processed PIL Image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG") 
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        logger.info("Processed image converted to base64.")

        # 10. Return the final JSON response
        return PredictionResponse(
            filename=file.filename,
            predictions=results, 
            image=img_base64, 
            disease_info=disease_info_list 
        )

    except Exception as e:
        logger.exception(f"Prediction failed for '{file.filename}': {e}") # Log full traceback
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}. Please ensure the uploaded file is a valid image or check server logs for more details.")