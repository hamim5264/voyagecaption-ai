import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model
import pickle
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# =============================
# CONFIG
# =============================
MODEL_NAME = "Salesforce/blip-image-captioning-base"
DEVICE = "cpu"
BATCH_SIZE = 2
EPOCHS = 2
LR = 1e-4
MAX_SAMPLES = 2000
TIMEOUT_SEC = 10
MAX_TEXT_LEN = 64
IMG_SIZE = 384  # BLIP recommended

# =============================
# LOAD DATA
# =============================
with open("training/caption/travel_captions.pkl", "rb") as f:
    data = pickle.load(f)

data = data[:MAX_SAMPLES]

# =============================
# SAFE IMAGE LOADER
# =============================
def load_image(url):
    try:
        r = requests.get(url, timeout=TIMEOUT_SEC)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content))

        # ✅ FORCE RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # ✅ FORCE SIZE
        img = img.resize((IMG_SIZE, IMG_SIZE))
        return img

    except Exception:
        return None

# =============================
# LOAD MODEL & PROCESSOR
# =============================
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(DEVICE)

# =============================
# APPLY LoRA
# =============================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =============================
# DATASET
# =============================
class TravelCaptionDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = load_image(sample["image_url"])
        if image is None:
            return None

        enc = self.processor(
            images=image,
            text=sample["caption"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LEN
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# =============================
# COLLATE FN
# =============================
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }

dataset = TravelCaptionDataset(data, processor)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# =============================
# TRAINING
# =============================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print("🚀 Starting LoRA caption training...")

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    steps = 0

    for batch in tqdm(loader):
        if batch is None:
            continue

        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    avg_loss = total_loss / max(steps, 1)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

# =============================
# SAVE MODEL
# =============================
model.save_pretrained("training/caption/voyage_caption_lora")
processor.save_pretrained("training/caption/voyage_caption_lora")

print("✅ LoRA caption model saved successfully")
