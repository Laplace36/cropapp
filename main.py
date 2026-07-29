import gc
import io
import json
import os

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Cap PyTorch to a single thread. On a memory-constrained instance handling
# one request at a time, extra threads don't speed things up but do add
# overhead — keeping this low helps stay under a tight RAM ceiling.
torch.set_num_threads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_efficientnet_b3.pth")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
IMAGE_SIZE = 224

app = FastAPI(title="Crop Health AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")  # force CPU explicitly; no GPU on this host

# ---- Load class names ----
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"class_names.json not found at {CLASS_NAMES_PATH}")
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# ---- Rebuild model architecture (must match training exactly) ----
model = models.efficientnet_b3()
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(class_names))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
del state_dict  # free the raw state dict once weights are copied into the model
model = model.to(device)
model.eval()
gc.collect()

# ---- Preprocessing (must match validation transforms used in training) ----
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def format_label(raw_label: str):
    """'Cashew_gumosis' -> crop='Cashew', condition='Gumosis'"""
    crop, _, condition = raw_label.partition("_")
    condition = condition.replace("_", " ").strip().title()
    return crop, condition


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "num_classes": len(class_names)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        raw_bytes = await file.read()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    input_tensor = preprocess(image).unsqueeze(0).to(device)
    del raw_bytes, image  # free the raw upload and decoded image promptly

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, k=min(3, len(class_names)))
    top_probs = top_probs.cpu().numpy().tolist()
    top_indices = top_indices.cpu().numpy().tolist()

    del input_tensor, logits, probabilities  # free inference tensors before responding
    gc.collect()

    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        crop, condition = format_label(class_names[idx])
        predictions.append({
            "crop": crop,
            "condition": condition,
            "is_healthy": condition.lower() == "healthy",
            "confidence": round(prob * 100, 2),
        })

    top_confidence = predictions[0]["confidence"]
    top_is_healthy = predictions[0]["is_healthy"]

    # A false "healthy" verdict is more costly than an uncertain one — missing
    # early-stage disease means the farmer takes no action. So we demand more
    # confidence before committing to "Healthy" than we do before flagging a
    # possible issue.
    HEALTHY_THRESHOLD = 80
    ISSUE_THRESHOLD = 65
    required_confidence = HEALTHY_THRESHOLD if top_is_healthy else ISSUE_THRESHOLD
    low_confidence = top_confidence < required_confidence

    return JSONResponse({
        "predictions": predictions,
        "top_prediction": predictions[0],
        "low_confidence": low_confidence,
    })


# Serve the frontend
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")