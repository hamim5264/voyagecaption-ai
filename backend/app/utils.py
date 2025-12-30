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
