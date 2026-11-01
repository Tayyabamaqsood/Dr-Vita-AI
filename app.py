
import io
import os
import torch

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoImageProcessor, DeiTForImageClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Vitiligo AI")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = DeiTForImageClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": None,
            "confidence": 0,
            "error": None,
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if image.content_type not in {
        "image/jpeg", "image/png", "image/webp"
    }:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, or WEBP image."
        )

    contents = await image.read()

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB."
        )

    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]

        predicted_id = int(probs.argmax().item())
        label = model.config.id2label[predicted_id]

        return {
            "prediction": label,
            "confidence": round(float(probs[predicted_id]), 4),
            "probabilities": {
                model.config.id2label[i]: round(float(probs[i]), 4)
                for i in range(len(probs))
            },
            "disclaimer": (
                "Experimental prediction only; not a medical diagnosis."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to process image: {exc}"
        )
