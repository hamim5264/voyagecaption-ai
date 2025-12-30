import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPModel
from torchvision import transforms
from PIL import Image

# =============================
# CONFIG
# =============================
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
DEVICE = "cpu"

# =============================
# LOAD CLIP MODEL
# =============================
clip_model = CLIPModel.from_pretrained(MODEL_NAME)
clip_model.eval()
clip_model.to(DEVICE)

for p in clip_model.parameters():
    p.requires_grad = False

# =============================
# CLASSIFIER HEAD
# =============================
classifier = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 2)  # 0 = non-travel, 1 = travel
).to(DEVICE)

# =============================
# LOAD CIFAR-10 DATASET
# =============================
print("⏳ Loading CIFAR-10 dataset...")
dataset = load_dataset("cifar10", split="train")

CLASS_NAMES = dataset.features["label"].names
TRAVEL_CLASSES = ["airplane", "ship"]

# =============================
# IMAGE TRANSFORM
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =============================
# CUSTOM DATASET
# =============================
class TravelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # CIFAR-10 uses "img"
        image = item["img"]
        label_id = item["label"]

        label_name = CLASS_NAMES[label_id]
        label = 1 if label_name in TRAVEL_CLASSES else 0

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        image = transform(image)
        return image, label

train_dataset = TravelDataset(dataset)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =============================
# TRAINING SETUP
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

# =============================
# TRAIN LOOP
# =============================
print("🚀 Training started...")
for epoch in range(EPOCHS):
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            features = clip_model.get_image_features(pixel_values=images)
            features = features / features.norm(dim=-1, keepdim=True)

        outputs = classifier(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

# =============================
# SAVE MODEL
# =============================
torch.save(
    classifier.state_dict(),
    "training/classifier/travel_classifier.pt"
)

print("✅ Travel classifier saved successfully")
