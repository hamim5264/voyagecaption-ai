import re
from collections import Counter
from typing import List

STOPWORDS = {
    "a","an","the","and","or","to","of","in","on","at","by","for","with","from",
    "this","that","it","is","are","was","were","be","been","being",
    "through","into","over","under","near","around","during",
    "photo","image","picture","view","scene","looking","standing","sitting"
}

KEYWORD_TAGS = [
    "travel","adventure","hiking","mountain","beach","nature","trip",
    "journey","vacation","explore","wanderlust","sunset","landscape",
    "camping","outdoors","photography","road","forest","lake","ocean","river"
]

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return s

def extract_keywords(captions: List[str], top_k: int = 6) -> List[str]:
    text = " ".join(captions).lower()
    words = re.findall(r"[a-z]+", text)
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_k)]

def make_title_from_keywords(keywords: List[str]) -> str:
    if not keywords:
        return "Travel Moment"
    # short phrase 2–4 words
    phrase = keywords[:4]
    return " ".join(phrase).title()

def make_description(captions: List[str], keywords: List[str], tone: str = "fun", context: str = "", user_desc: str = "") -> str:
    # build a single “story-like” description
    base = ""
    if keywords:
        # pick 3–5 keywords
        k = keywords[:5]
        if len(k) >= 3:
            base = f"A journey through {k[0]}, {k[1]}, and {k[2]}."
        elif len(k) == 2:
            base = f"A journey through {k[0]} and {k[1]}."
        else:
            base = f"A journey through {k[0]}."
    else:
        # fallback: use the first caption
        base = captions[0].strip().capitalize()
        if not base.endswith("."):
            base += "."

    tone = (tone or "").lower().strip()
    if tone == "fun":
        base += " ✨"
    elif tone == "formal":
        base = base.replace("A journey", "A memorable journey")
    elif tone == "romantic":
        base += " ❤️"

    if context.strip():
        base += f" {clean_text(context)}"
        if not base.endswith("."):
            base += "."

    if user_desc.strip():
        base += f" {clean_text(user_desc)}"
        if not base.endswith("."):
            base += "."

    return clean_text(base)

def make_hashtags(captions: List[str], keywords: List[str], limit: int = 15) -> List[str]:
    joined = " ".join(captions).lower()
    tags = []

    # keyword-based tags
    for k in keywords:
        if k.isalpha():
            tags.append(f"#{k}")

    # add known travel tags if present in caption text
    for k in KEYWORD_TAGS:
        if k in joined and f"#{k}" not in tags:
            tags.append(f"#{k}")

    # ensure base tags
    for t in ["#travel", "#nature", "#photography"]:
        if t not in tags:
            tags.append(t)

    # de-dup preserve order
    seen = set()
    out = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)

    return out[:limit]


### NEW CODE
# import re
# from typing import List

# # -----------------------------
# # CLEAN TEXT
# # -----------------------------
# def clean_sentence(text: str) -> str:
#     text = text.strip()
#     if not text:
#         return ""
#     text = text[0].upper() + text[1:]
#     if not text.endswith("."):
#         text += "."
#     return text


# # -----------------------------
# # SCENE INFERENCE (LIGHTWEIGHT)
# # -----------------------------
# def infer_scene(caption: str) -> str:
#     c = caption.lower()

#     if any(k in c for k in ["hospital", "patient", "medical", "oxygen", "bed"]):
#         return "medical"
#     if any(k in c for k in ["mountain", "beach", "desert", "forest", "hiking", "sunset"]):
#         return "travel"
#     if any(k in c for k in ["people", "friends", "group", "crowd"]):
#         return "social"
#     if any(k in c for k in ["sad", "cry", "emotional", "tear"]):
#         return "emotional"

#     return "general"


# # -----------------------------
# # TITLE FROM RAW CAPTION
# # -----------------------------
# def make_title_from_caption(raw_caption: str) -> str:
#     """
#     Converts raw caption into a short, human-readable title.
#     """
#     words = re.findall(r"[a-zA-Z]+", raw_caption)
#     if not words:
#         return "A Captured Moment"

#     # Keep it short: 3–5 words
#     title_words = words[:5]
#     return " ".join(title_words).title()


# # -----------------------------
# # DESCRIPTION FROM RAW CAPTION
# # -----------------------------
# def make_description_from_caption(raw_caption: str, mode: str = "general") -> str:
#     base = clean_sentence(raw_caption)
#     mode = mode.lower()

#     if mode == "travel":
#         return base + " It captures the calm, freedom, and beauty that travel often brings."
#     if mode == "emotional":
#         return base + " There is a quiet emotion in this moment that feels deeply personal."
#     if mode == "social":
#         return base + " Moments like these are best when they are shared."
#     if mode == "medical":
#         return base + " A moment of strength, patience, and quiet resilience."

#     # general
#     return base + " A moment that feels natural, unposed, and real."


# # -----------------------------
# # HASHTAGS FROM SCENE
# # -----------------------------
# def make_hashtags(scene: str, limit: int = 12) -> List[str]:
#     base = ["#photography", "#moments", "#storytelling"]

#     scene_tags = {
#         "travel": ["#travel", "#wanderlust", "#nature"],
#         "emotional": ["#emotions", "#feelings"],
#         "social": ["#people", "#life"],
#         "medical": ["#strength", "#healing"],
#         "general": ["#life", "#capture"],
#     }

#     tags = base + scene_tags.get(scene, [])
#     return tags[:limit]
