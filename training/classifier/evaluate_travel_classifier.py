import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPModel
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import numpy as np

# =============================
# CONFIG
# =============================
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 16
DEVICE = "cpu"

# =============================
# LOAD CLIP
# =============================
clip_model = CLIPModel.from_pretrained(MODEL_NAME)
clip_model.eval()
clip_model.to(DEVICE)

for p in clip_model.parameters():
    p.requires_grad = False

# =============================
# LOAD CLASSIFIER
# =============================
classifier = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 2)
)

classifier.load_state_dict(
    torch.load("training/classifier/travel_classifier.pt", map_location=DEVICE)
)
classifier.eval()
classifier.to(DEVICE)

# =============================
# LOAD TEST DATA
# =============================
print("⏳ Loading CIFAR-10 test dataset...")
dataset = load_dataset("cifar10", split="test")

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
# DATASET
# =============================
class TravelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["img"]
        label_id = item["label"]

        label_name = CLASS_NAMES[label_id]
        label = 1 if label_name in TRAVEL_CLASSES else 0

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        image = transform(image)
        return image, label

test_dataset = TravelDataset(dataset)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =============================
# EVALUATION
# =============================
y_true = []
y_pred = []

print("📊 Evaluating model...")

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        features = clip_model.get_image_features(pixel_values=images)
        features = features / features.norm(dim=-1, keepdim=True)

        outputs = classifier(features)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# =============================
# METRICS
# =============================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n✅ Evaluation Results")
print("---------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)
