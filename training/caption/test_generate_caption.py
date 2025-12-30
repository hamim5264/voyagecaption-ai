import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel
import requests
from PIL import Image
from io import BytesIO

# =============================
# CONFIG
# =============================
BASE_MODEL = "Salesforce/blip-image-captioning-base"
LORA_PATH = "training/caption/voyage_caption_lora"
DEVICE = "cpu"
IMG_SIZE = 384

# ✅ Test with any travel image
TEST_IMAGE_URL = (
    "https://plus.unsplash.com/premium_photo-1677343210638-5d3ce6ddbf85?q=80&w=688&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
)

# =============================
# IMAGE LOADER
# =============================
def load_image(url: str) -> Image.Image:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    img = Image.open(BytesIO(r.content))
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((IMG_SIZE, IMG_SIZE))
    return img

# =============================
# TITLE LOGIC (SHORT & POETIC)
# =============================
def make_title(caption: str) -> str:
    caption = caption.lower().strip()

    # Remove common filler words
    remove_words = [
        "a", "an", "the", "with", "of", "and",
        "in", "on", "at", "by", "from", "into",
        "surrounded", "standing", "looking", "view"
    ]

    words = [w for w in caption.split() if w not in remove_words]

    # Keep only first 2–4 meaningful words
    phrase = words[:4]

    if not phrase:
        return "Travel moment"

    # Capitalize each word (phrase style)
    title = " ".join(phrase).title()
    return title

# =============================
# DESCRIPTION LOGIC
# =============================
def make_description(caption: str) -> str:
    caption = caption.strip().capitalize()
    if not caption.endswith("."):
        caption += "."
    return caption

# =============================
# HASHTAG LOGIC
# =============================
def make_hashtags(caption: str):
    keywords = [
        "travel", "adventure", "hiking", "mountain", "beach",
        "nature", "trip", "journey", "vacation", "explore",
        "wanderlust", "sunset", "landscape", "camping",
        "outdoors", "photography"
    ]

    caption_lower = caption.lower()
    tags = [f"#{k}" for k in keywords if k in caption_lower]

    base_tags = ["#travel", "#nature", "#photography"]
    for t in base_tags:
        if t not in tags:
            tags.append(t)

    return tags[:15]

# =============================
# LOAD MODEL + LoRA
# =============================
print("⏳ Loading base model + LoRA adapter...")
processor = BlipProcessor.from_pretrained(BASE_MODEL)
base_model = BlipForConditionalGeneration.from_pretrained(BASE_MODEL).to(DEVICE)

model = PeftModel.from_pretrained(base_model, LORA_PATH).to(DEVICE)
model.eval()

# =============================
# GENERATE CAPTION
# =============================
print("🖼️ Loading test image...")
image = load_image(TEST_IMAGE_URL)

inputs = processor(images=image, return_tensors="pt").to(DEVICE)

print("✨ Generating caption...")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=40,
        num_beams=4
    )

caption = processor.decode(output[0], skip_special_tokens=True).strip()

# =============================
# POST-PROCESSING
# =============================
title = make_title(caption)
description = make_description(caption)
hashtags = make_hashtags(caption)

# =============================
# RESULT
# =============================
print("\n✅ RESULT")
print("--------------------------------------------------")
print("Title       :", title)
print("Description :", description)
print("Hashtags    :", " ".join(hashtags))
