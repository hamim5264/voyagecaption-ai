from pathlib import Path
import os
import torch
import torch.nn as nn
from transformers import CLIPModel, BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel
from torchvision import transforms
from PIL import Image

# ---- Paths (defaults point to your project root training outputs) ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../voyagecaption-ai
DEFAULT_CLASSIFIER_PATH = PROJECT_ROOT / "training" / "classifier" / "travel_classifier.pt"
DEFAULT_LORA_PATH = PROJECT_ROOT / "training" / "caption" / "voyage_caption_lora"

CLIP_NAME = "openai/clip-vit-base-patch32"
BLIP_NAME = "Salesforce/blip-image-captioning-base"

DEVICE = "cpu"

# ---- Image transforms for classifier ----
classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def pil_from_bytes(b: bytes) -> Image.Image:
    img = Image.open(torch.io.BytesIO(b))  # type: ignore[attr-defined]
    return img.convert("RGB")

class TravelClassifier:
    def __init__(self, weight_path: Path):
        self.clip = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        # same head you trained: 512 -> 256 -> 2
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        ).to(DEVICE)

        state = torch.load(str(weight_path), map_location=DEVICE)
        self.head.load_state_dict(state)
        self.head.eval()

    @torch.no_grad()
    def predict_proba(self, images: list[Image.Image]) -> list[float]:
        # returns travel probability per image
        batch = torch.stack([classifier_transform(img) for img in images]).to(DEVICE)
        feats = self.clip.get_image_features(pixel_values=batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = self.head(feats)
        probs = torch.softmax(logits, dim=1)[:, 1]  # travel=1
        return probs.cpu().tolist()

class CaptionGenerator:
    def __init__(self, lora_path: Path):
        self.processor = BlipProcessor.from_pretrained(BLIP_NAME)
        base = BlipForConditionalGeneration.from_pretrained(BLIP_NAME).to(DEVICE)
        self.model = PeftModel.from_pretrained(base, str(lora_path)).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def caption(self, image: Image.Image) -> str:
        # BLIP expects PIL; keep it RGB and decent size
        image = image.convert("RGB").resize((384, 384))
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
        out = self.model.generate(**inputs, max_new_tokens=40, num_beams=4)
        text = self.processor.decode(out[0], skip_special_tokens=True).strip()
        return text

def load_models():
    classifier_path = Path(os.getenv("TRAVEL_CLASSIFIER_PATH", str(DEFAULT_CLASSIFIER_PATH)))
    lora_path = Path(os.getenv("LORA_PATH", str(DEFAULT_LORA_PATH)))

    if not classifier_path.exists():
        raise FileNotFoundError(f"Travel classifier weights not found: {classifier_path}")

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA adapter folder not found: {lora_path}")

    clf = TravelClassifier(classifier_path)
    cap = CaptionGenerator(lora_path)
    return clf, cap
