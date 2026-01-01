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
    "https://images.unsplash.com/photo-1758526387794-551feb00eb1d?q=80&w=1632&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
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


### NEW CODE
# import torch
# import random
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from peft import PeftModel
# import requests
# from PIL import Image
# from io import BytesIO

# # =============================
# # CONFIG
# # =============================
# BASE_MODEL = "Salesforce/blip-image-captioning-base"
# LORA_PATH = "training/caption/voyage_caption_lora"
# DEVICE = "cpu"
# IMG_SIZE = 384

# # Change mode here:
# # general | emotional | travel | social
# MODE = "general"

# TEST_IMAGE_URL = (
#     "https://images.unsplash.com/photo-1606166187734-a4cb74079037?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
# )

# # =============================
# # IMAGE LOADER
# # =============================
# def load_image(url: str) -> Image.Image:
#     headers = {"User-Agent": "Mozilla/5.0"}
#     r = requests.get(url, headers=headers, timeout=15)
#     r.raise_for_status()

#     img = Image.open(BytesIO(r.content))
#     if img.mode != "RGB":
#         img = img.convert("RGB")

#     return img.resize((IMG_SIZE, IMG_SIZE))

# # =============================
# # SEMANTIC ANALYSIS
# # =============================
# def analyze_caption(caption: str) -> dict:
#     caption = caption.lower()

#     emotion = "neutral"
#     subject = "moment"

#     if any(k in caption for k in ["tear", "cry", "sad", "emotional"]):
#         emotion = "emotional"
#     elif any(k in caption for k in ["smile", "happy", "laugh"]):
#         emotion = "happy"
#     elif any(k in caption for k in ["mountain", "beach", "forest", "nature"]):
#         emotion = "travel"

#     if "woman" in caption or "girl" in caption:
#         subject = "woman"
#     elif "man" in caption:
#         subject = "man"
#     elif "child" in caption:
#         subject = "child"
#     elif "player" in caption or "cricket" in caption:
#         subject = "sport"
#     elif "bear" in caption or "toy" in caption:
#         subject = "comfort"

#     return {"emotion": emotion, "subject": subject}

# # =============================
# # HUMAN TITLE GENERATOR
# # =============================
# def make_title(analysis: dict, mode: str) -> str:
#     titles = {
#         "general": [
#             "A Moment Worth Remembering ✨",
#             "Life, Captured Gently 🤍",
#             "Where Feelings Find Space 💭",
#             "A Quiet Frame in Time 🌿",
#         ],
#         "emotional": [
#             "When Emotions Speak Softly 🤍",
#             "Quiet Strength, Gentle Feelings ✨",
#             "A Pause Filled With Emotion 💭",
#         ],
#         "travel": [
#             "Chasing Calm Beyond the Noise 🌍",
#             "Where the Journey Feels Alive ✨",
#             "Moments Found Along the Way 🏞️",
#         ],
#         "social": [
#             "Real Moments, Unfiltered 📸",
#             "Captured Just as It Felt 🤍",
#             "Everyday Stories, Beautifully Told ✨",
#         ],
#     }

#     return random.choice(titles.get(mode, titles["general"]))

# # =============================
# # HUMAN DESCRIPTION GENERATOR
# # =============================
# def make_description(caption: str, analysis: dict, mode: str) -> str:
#     emotion = analysis["emotion"]

#     descriptions = {
#         "general": [
#             "A simple moment captured naturally, reminding us that beauty often lives in the quiet details.",
#             "Sometimes the most powerful stories are found in the smallest, most honest moments.",
#         ],
#         "emotional": [
#             "Some moments don’t need words. A soft expression, a gentle pause, and emotions quietly taking the lead.",
#             "There is beauty in vulnerability — a reminder that feeling deeply is part of being human.",
#         ],
#         "travel": [
#             "Surrounded by new paths and open skies, this moment reflects the calm and freedom that travel brings.",
#             "Every journey leaves behind moments that stay with us long after the road ends.",
#         ],
#         "social": [
#             "Life doesn’t need filters to be meaningful. Just real moments, captured as they are.",
#             "A candid frame that speaks louder than posed perfection.",
#         ],
#     }

#     pool = descriptions.get(mode, descriptions["general"])
#     return random.choice(pool)

# # =============================
# # HASHTAGS (GENERAL PURPOSE)
# # =============================
# def make_hashtags(analysis: dict, mode: str):
#     base = ["#photography", "#moments", "#storytelling"]

#     mode_tags = {
#         "general": ["#life", "#memories"],
#         "emotional": ["#emotions", "#feelings", "#human"],
#         "travel": ["#travel", "#wanderlust", "#nature"],
#         "social": ["#social", "#everyday", "#real"],
#     }

#     return base + mode_tags.get(mode, [])

# # =============================
# # LOAD MODEL
# # =============================
# print("⏳ Loading base model + LoRA adapter...")
# processor = BlipProcessor.from_pretrained(BASE_MODEL)
# base_model = BlipForConditionalGeneration.from_pretrained(BASE_MODEL).to(DEVICE)
# model = PeftModel.from_pretrained(base_model, LORA_PATH).to(DEVICE)
# model.eval()

# # =============================
# # GENERATE CAPTION
# # =============================
# print("🖼️ Loading image...")
# image = load_image(TEST_IMAGE_URL)

# inputs = processor(images=image, return_tensors="pt").to(DEVICE)

# print("✨ Generating caption...")
# with torch.no_grad():
#     output = model.generate(**inputs, max_new_tokens=40, num_beams=4)

# raw_caption = processor.decode(output[0], skip_special_tokens=True).strip()

# # =============================
# # POST PROCESSING
# # =============================
# analysis = analyze_caption(raw_caption)
# title = make_title(analysis, MODE)
# description = make_description(raw_caption, analysis, MODE)
# hashtags = make_hashtags(analysis, MODE)

# # =============================
# # RESULT
# # =============================
# print("\n✅ RESULT")
# print("--------------------------------------------------")
# print("Raw Caption :", raw_caption)
# print("Title       :", title)
# print("Description :", description)
# print("Hashtags    :", " ".join(hashtags))
