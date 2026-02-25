# VoyageCaption AI

VoyageCaption AI is an AI-powered backend system that automatically generates **human-like travel captions** (title, description, and hashtags) from uploaded images.

The system is intentionally designed to **work only with travel-related photos**.  
If uploaded images do not appear to be travel-related, the API safely rejects them and provides **clear suggestions** instead of generating misleading captions.

This project is built to be consumed by **mobile or web apps** (Flutter, React, etc.) as a clean REST API.

---

## 🔍 What This Project Does

- Accepts **one or multiple images**
- Detects whether images are **travel-related**
- Generates:
  - Short, readable **title**
  - Story-like **description**
  - Relevant **hashtags**
- Rejects non-travel images with helpful suggestions
- Supports **tone-based regeneration** (fun, formal, romantic, etc.)
- Designed for **Instagram / social content workflows**

---

## 🧠 AI Models Used

### 1️⃣ Travel Image Classifier
Used to validate whether an image is travel-related before captioning.

- **Architecture:** CNN-based image classifier
- **Training Dataset:** CIFAR-10 (custom binary labeling: travel vs non-travel)
- **Purpose:** Prevent caption generation on irrelevant images

#### 📊 Classifier Performance
| Metric | Score |
|------|------|
| Accuracy | **97.15%** |
| Precision | 95.44% |
| Recall | 90.05% |
| F1-score | 92.67% |

This ensures the system is **highly reliable** in filtering travel images.

---

### 2️⃣ Image Caption Generator
Used only after images pass the travel classifier.

- **Base Model:** `Salesforce/blip-image-captioning-base`
- **Fine-tuning:** LoRA (Low-Rank Adaptation)
- **Framework:** Hugging Face Transformers + PEFT
- **Output:** Natural-language image captions

The model produces a **raw caption**, which is later refined into human-friendly content.

---

## ✍️ Caption Generation Logic

Once travel images are validated:

1. Generate **raw captions** for each valid image
2. Merge captions into a single semantic context
3. Extract keywords (NLP-based)
4. Generate:
   - **Title** → short & readable
   - **Description** → story-like, travel-focused
   - **Hashtags** → relevant + base travel tags

### Example Output
```json
{
  "valid": true,
  "title": "Small Cabin Middle Fjord",
  "description": "A journey through small, cabin, and middle. ✨",
  "hashtags": [
    "#cabin",
    "#fjord",
    "#travel",
    "#nature",
    "#photography"
  ]
}
```
### 🚫 Non-Travel Image Handling
If uploaded images are not travel-related, the API returns:
```json
{
  "valid": false,
  "message": "This doesn't look like travel photos.",
  "suggestions": [
    "Try outdoor scenery (mountains, beaches, forests, roads).",
    "Avoid screenshots, documents, or indoor-only photos.",
    "Landmarks and wide landscape shots work best."
  ]
}
```
This prevents incorrect or misleading captions.

### 🔁 Regeneration & Tone Support
Captions can be regenerated without re-uploading images by changing the tone:

1. fun

2. formal

3. romantic

This makes the API ideal for content creation apps where users want multiple caption styles.

### 🌐 API Overview
Health Check
```json
GET /health
```
Generate Caption
```json
POST /generate-caption
```
Request Type: multipart/form-data

Fields:

images → one or more image files

platform → instagram / facebook / linkedin

tone → fun / formal / romantic

context → optional user context

userTitle → optional manual title

userDescription → optional manual description

---

### ▶️ Run the Project Locally
```text
Follow these steps to run the backend locally.
1️1️⃣ Create Virtual Environment
bash
Copy code
python -m venv venv
2️⃣ Activate Virtual Environment (Windows)
bash
Copy code
venv\Scripts\activate
2️⃣ Activate Virtual Environment (macOS / Linux)
bash
Copy code
source venv/bin/activate
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Start Backend Server
bash
Copy code
uvicorn backend.app.main:app --reload
5️⃣ Open API Docs
text
Copy code
http://127.0.0.1:8000/docs
🧪 Evaluate Travel Classifier
bash
Copy code
python training/classifier/evaluate_travel_classifier.py
```

### 📊 Evaluation Metrics Output
Accuracy
Precision
Recall
F1-score
Confusion Matrix

### 🧩 Designed for App Integration
VoyageCaption AI is a backend-only AI engine designed for:

- Flutter apps
- Web apps
- Content creation platforms
- Social media tools

### 🎯 Project Focus
- Travel photos only

### ✅ Project Status
- Stable & production-ready

### 🚀 Future Improvements
- General image captioning
- Emotional storytelling
- Multi-language captions
- User personalization

### 👤 Author
MD. Abdul Hamim
AI & Flutter Developer

### 🧠 Project Scope
- This project demonstrates a full AI pipeline:

- model training → evaluation → inference → API → app integration

### 📄 License
This project is provided for learning, experimentation, and integration purposes.
Commercial usage depends on model and dataset licenses.


### 🔜 Next Possible Additions
- Flutter API integration example
- Postman collection
- Docker setup
- Production deployment (Render / Railway)

### Screnshots
<img width="3375" height="3375" alt="5" src="https://github.com/user-attachments/assets/717398e2-df3e-4f4a-a152-bf2eb5b8c01b" />
<img width="3375" height="3375" alt="4" src="https://github.com/user-attachments/assets/9a3118f0-ea25-42bd-b6c1-6c63ec6bf0ab" />
<img width="3375" height="3375" alt="3" src="https://github.com/user-attachments/assets/0afebdcd-92b7-4a4d-999f-7d631d0e8ced" />
<img width="3375" height="3375" alt="2" src="https://github.com/user-attachments/assets/3d6f913b-2184-4a49-acab-a9ae85a7099a" />
<img width="3375" height="3375" alt="1" src="https://github.com/user-attachments/assets/bff8a3d4-83e7-4390-a2ab-55ed144e8c98" />
<img width="1080" height="2400" alt="Screenshot_20260103_160323" src="https://github.com/user-attachments/assets/081c366f-94de-4abf-a820-43fbb84e4fa8" />
<img width="1080" height="2400" alt="Screenshot_20260103_160312" src="https://github.com/user-attachments/assets/786b0902-1cdf-4f09-83f9-b270a7aa4757" />
<img width="1080" height="2400" alt="Screenshot_20260103_160304" src="https://github.com/user-attachments/assets/2e0460d2-5e03-4944-a08f-4ff8cfa8472a" />
<img width="1080" height="2400" alt="Screenshot_20260103_160225" src="https://github.com/user-attachments/assets/2f35c846-a1e5-421c-bdd4-03dcb697bc3e" />
<img width="1080" height="2400" alt="Screenshot_20260103_155059" src="https://github.com/user-attachments/assets/bdb0b5db-45de-487f-952a-4f4c15a97946" />
<img width="1080" height="2400" alt="Screenshot_20260103_154644" src="https://github.com/user-attachments/assets/e5687030-4c72-4a25-8b09-b747131e8898" />






