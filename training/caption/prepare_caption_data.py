from datasets import load_dataset
import pickle

print("⏳ Loading Conceptual Captions dataset...")

# Load small subset for safety
dataset = load_dataset(
    "conceptual_captions",
    split="train[:2%]"
)

TRAVEL_KEYWORDS = [
    "travel", "trip", "journey", "vacation",
    "mountain", "beach", "forest", "river",
    "lake", "sea", "ocean", "hiking",
    "camping", "adventure", "outdoor",
    "sunset", "nature", "road", "island"
]

def is_travel_caption(caption: str) -> bool:
    caption = caption.lower()
    return any(k in caption for k in TRAVEL_KEYWORDS)

travel_samples = []

for item in dataset:
    caption = item["caption"]
    image_url = item["image_url"]

    if caption and is_travel_caption(caption):
        travel_samples.append({
            "image_url": image_url,
            "caption": caption
        })

print(f"✅ Travel captions collected: {len(travel_samples)}")

# Save limited subset (safe for CPU LoRA later)
MAX_SAMPLES = 2000
travel_samples = travel_samples[:MAX_SAMPLES]

with open("training/caption/travel_captions.pkl", "wb") as f:
    pickle.dump(travel_samples, f)

print("📦 Saved travel_captions.pkl")
