from fastapi import FastAPI, File, UploadFile
from typing import List
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

app = FastAPI()

project_root = Path(__file__).resolve().parent.parent
save_models_dir = project_root / "models"
model_file_name = save_models_dir/ "best_resnet50_model.pth"

# Load the checkpoint
checkpoint = torch.load(str(model_file_name))

state_dict = checkpoint["state_dict"]
class_mapping = checkpoint["class_mapping"]

# Extract the number of output classes from the classifier layer
num_classes = state_dict["fc.weight"].shape[0]

# Recreate the model architecture
loaded_model = resnet50(num_classes=num_classes)
loaded_model.load_state_dict(state_dict)

test_transformers = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.post("/predict/")
async def predict(files: List[UploadFile] = File(...)):
    """
    Predict the classes of multiple uploaded images.
    Send a POST request with multiple image files.
    """
    results = {}

    for file in files:

        # Load the images and preprocess
        image = Image.open(file.file).convert("RGB")
        input_tensor = test_transformers(image).unsqueeze(0)

    
        # Perform inference
        with torch.no_grad():
            output = loaded_model(input_tensor)
            predicted_class = output.argmax(1).item()  # Get the class index
            predicted_class_name = class_mapping[predicted_class]
    
        results[file.filename] = {"class_id": predicted_class, "class_name": predicted_class_name}

    return {"predictions": results}

@app.get("/predict/")
async def predict_info():
    return {"message": "This endpoint only supports POST requests for predictions."}

@app.get("/")
async def root():
    return {"message": "Welcome to the Car Classification API!",
            "instructions": "Send a POST request to /predict/ with an image of your car to get predictions."}