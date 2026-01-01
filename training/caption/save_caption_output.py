import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import textwrap

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = "Salesforce/blip-image-captioning-base"
LORA_PATH = "training/caption/voyage_caption_lora"
DEVICE = "cpu"
IMG_SIZE = 384
MODE = "travel"

TEST_IMAGE_URL = "https://images.unsplash.com/photo-1606166187734-a4cb74079037"

OUTPUT_PATH = "output_caption.jpg"

# -----------------------------
# LOAD IMAGE
# -----------------------------
def load_image(url):
    r = requests.get(url, timeout=15)
    img = Image.open(BytesIO(r.content)).convert("RGB")
    return img.resize((IMG_SIZE, IMG_SIZE))

# -----------------------------
# LOAD MODEL
# -----------------------------
processor = BlipProcessor.from_pretrained(BASE_MODEL)
base_model = BlipForConditionalGeneration.from_pretrained(BASE_MODEL).to(DEVICE)
model = PeftModel.from_pretrained(base_model, LORA_PATH).to(DEVICE)
model.eval()

# -----------------------------
# GENERATE RAW CAPTION
# -----------------------------
image = load_image(TEST_IMAGE_URL)
inputs = processor(images=image, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=40)

raw_caption = processor.decode(out[0], skip_special_tokens=True)

# -----------------------------
# SIMPLE TITLE & DESCRIPTION
# -----------------------------
title = raw_caption.title().split()[:4]
title = " ".join(title)

description = raw_caption.capitalize() + ". A moment captured naturally."

# -----------------------------
# DRAW ON IMAGE
# -----------------------------
canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE + 180), "white")
canvas.paste(image, (0, 0))

draw = ImageDraw.Draw(canvas)

try:
    font_title = ImageFont.truetype("arial.ttf", 24)
    font_desc = ImageFont.truetype("arial.ttf", 16)
except:
    font_title = font_desc = ImageFont.load_default()

# Draw title
draw.text((10, IMG_SIZE + 10), title, fill="black", font=font_title)

# Draw description (wrapped)
wrapped = textwrap.fill(description, width=45)
draw.text((10, IMG_SIZE + 50), wrapped, fill="gray", font=font_desc)

canvas.save(OUTPUT_PATH)

print(f"✅ Output saved as {OUTPUT_PATH}")
