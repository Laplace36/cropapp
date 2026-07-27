# Crop Health AI — Web App

A FastAPI backend that serves your trained EfficientNet-B3 model, with a simple
drag-and-drop frontend for uploading leaf images and viewing predictions.

## Folder contents
```
cropapp/
├── main.py                    # FastAPI backend (loads model, serves /predict)
├── requirements.txt
├── best_efficientnet_b3.pth   # your trained model weights
├── class_names.json           # the 22 class labels, in training order
└── static/
    └── index.html              # frontend (upload UI + results display)
```

## Setup (run this in VS Code / your terminal)

1. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Note: this will download PyTorch, which is a large package (a few hundred MB).
   It'll run on CPU on your laptop — that's completely fine for *inference*
   (running predictions), even though it was too slow for *training*.

3. Run the app:
   ```
   uvicorn main:app --reload
   ```

4. Open your browser to:
   ```
   http://127.0.0.1:8000
   ```
   Upload a leaf image and you should see the top-3 predictions with confidence scores.

## How it works

- On startup, `main.py` rebuilds the exact same EfficientNet-B3 architecture used
  in training, then loads your saved weights (`best_efficientnet_b3.pth`) into it.
- `class_names.json` maps the model's numeric output back to readable labels
  (e.g. index 5 → `"Cassava_bacterial blight"` → displayed as **Cassava — Bacterial Blight**).
- The `/predict` endpoint accepts an uploaded image, applies the same resize/
  normalize preprocessing used during training (224×224, ImageNet mean/std),
  and returns the top 3 predicted classes with confidence percentages.
- The frontend just calls `/predict` and renders the JSON response — no
  framework, plain HTML/CSS/JS, so it's easy to swap out later.

## Testing it

Test with a mix of real leaf photos:
- A few clearly diseased leaves per crop (should predict the right disease
  with reasonably high confidence)
- A few healthy leaves per crop (should predict "Healthy" — this is the
  class worth watching, since it was your smallest training class for Maize)
- A photo that isn't a leaf at all (sanity check — the model will still force
  a prediction since it has no "not a leaf" class; this is a known limitation,
  not a bug)

## Known limitations to keep in mind
- The model only knows the 22 classes it was trained on (4 crops). Any other
  crop or a non-leaf image will still get force-classified into one of these
  22 — there's no "unknown/out of scope" category.
- Predictions reflect visible symptoms at time of photo — this is not
  pre-symptomatic/early-stage detection (the training data doesn't support that).
- CPU inference is fine for single-image predictions (well under a second)
  but if you ever need to process many images at once, consider batching or
  a GPU-backed deployment.

## Deployment (when ready)
This app is deploy-ready as-is on platforms like Render or Railway — just
point their build process at `requirements.txt` and run command at
`uvicorn main:app --host 0.0.0.0 --port $PORT`. No code changes needed.
