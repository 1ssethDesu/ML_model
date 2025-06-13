import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os # Import os for environment variables

# ... (other imports)

def load_faster_rcnn_model(num_classes: int = 4):
    print("Starting model loading process...")
    print("Attempting to load pre-trained Faster R-CNN with default weights...")
    # This line will download weights if not cached
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    print("Pre-trained model loaded. Modifying classifier head...")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print("Classifier head modified. Attempting to load custom state_dict...")

    # Get model path from environment variable, or use a default relative path
    # Adjust 'default_model_path_in_deployment' to reflect where your model will be
    # in the deployed environment (e.g., inside a 'models' folder at the app root).
    # Example: if your model is in 'my_app_root/models/my_model.pth'
    model_path = os.getenv('CUSTOM_MODEL_PATH', '/Users/chhortchhorraseth/Desktop/Machine_learning/ML_model/model/faster_rcnn_model.pth')

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Custom model weights loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Custom model file not found at {model_path}. "
              "Please ensure the path is correct and the file exists in the deployment environment.")
        raise # Re-raise to ensure startup fails if model isn't found
    except Exception as e:
        print(f"Error loading custom weights: {e}")
        raise # Re-raise for debugging purposes
    print("Model initialization complete.")
    return model

_global_model = None # Keep this global variable
def get_model():
    global _global_model
    if _global_model is None:
        _global_model = load_faster_rcnn_model(num_classes=4)
    return _global_model