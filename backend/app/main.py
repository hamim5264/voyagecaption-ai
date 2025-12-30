from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from PIL import Image
from io import BytesIO

from .schemas import CaptionResponse
from .models import load_models
from .pipeline import run_multi_image_pipeline

app = FastAPI(title="VoyageCaption AI Backend", version="1.0.0")

# CORS (for Flutter/web testing). Tighten later if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
travel_clf = None
caption_gen = None

@app.on_event("startup")
def _startup():
    global travel_clf, caption_gen
    travel_clf, caption_gen = load_models()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate-caption", response_model=CaptionResponse)
async def generate_caption(
    images: List[UploadFile] = File(...),
    platform: str = Form("instagram"),
    tone: str = Form("fun"),
    imageCount: Optional[int] = Form(None),
    context: str = Form(""),
    userTitle: str = Form(""),
    userDescription: str = Form(""),
):
    """
    Multipart form-data:
      - images: multiple files
      - platform, tone, context, userTitle, userDescription (same as your old JSON idea)
    """

    if travel_clf is None or caption_gen is None:
        return CaptionResponse(valid=False, message="Models not loaded.", suggestions=["Restart server."])

    pil_images: List[Image.Image] = []

    for f in images:
        content = await f.read()
        try:
            img = Image.open(BytesIO(content)).convert("RGB")
            pil_images.append(img)
        except Exception:
            continue

    if not pil_images:
        return CaptionResponse(
            valid=False,
            message="No valid images received.",
            suggestions=["Upload JPG/PNG travel photos."]
        )

    # 1) Classify travel probability per image
    travel_probs = travel_clf.predict_proba(pil_images)

    # 2) Run pipeline: validate + caption + merge
    result, debug = run_multi_image_pipeline(
        images=pil_images,
        platform=platform,
        tone=tone,
        context=context,
        user_title=userTitle,
        user_description=userDescription,
        travel_probs=travel_probs,
        caption_fn=caption_gen.caption,
        travel_threshold=0.60
    )

    # Return response (debug available internally if you want later)
    return CaptionResponse(**result)
