# import re
# from collections import Counter
# from typing import List

# STOPWORDS = {
#     "a","an","the","and","or","to","of","in","on","at","by","for","with","from",
#     "this","that","it","is","are","was","were","be","been","being",
#     "through","into","over","under","near","around","during",
#     "photo","image","picture","view","scene","looking","standing","sitting"
# }

# KEYWORD_TAGS = [
#     "travel","adventure","hiking","mountain","beach","nature","trip",
#     "journey","vacation","explore","wanderlust","sunset","landscape",
#     "camping","outdoors","photography","road","forest","lake","ocean","river"
# ]

# def clean_text(s: str) -> str:
#     s = re.sub(r"\s+", " ", s.strip())
#     return s

# def extract_keywords(captions: List[str], top_k: int = 6) -> List[str]:
#     text = " ".join(captions).lower()
#     words = re.findall(r"[a-z]+", text)
#     words = [w for w in words if w not in STOPWORDS and len(w) > 2]
#     freq = Counter(words)
#     return [w for w, _ in freq.most_common(top_k)]

# def make_title_from_keywords(keywords: List[str]) -> str:
#     if not keywords:
#         return "Travel Moment"
#     # short phrase 2–4 words
#     phrase = keywords[:4]
#     return " ".join(phrase).title()

# def make_description(captions: List[str], keywords: List[str], tone: str = "fun", context: str = "", user_desc: str = "") -> str:
#     # build a single “story-like” description
#     base = ""
#     if keywords:
#         # pick 3–5 keywords
#         k = keywords[:5]
#         if len(k) >= 3:
#             base = f"A journey through {k[0]}, {k[1]}, and {k[2]}."
#         elif len(k) == 2:
#             base = f"A journey through {k[0]} and {k[1]}."
#         else:
#             base = f"A journey through {k[0]}."
#     else:
#         # fallback: use the first caption
#         base = captions[0].strip().capitalize()
#         if not base.endswith("."):
#             base += "."

#     tone = (tone or "").lower().strip()
#     if tone == "fun":
#         base += " ✨"
#     elif tone == "formal":
#         base = base.replace("A journey", "A memorable journey")
#     elif tone == "romantic":
#         base += " ❤️"

#     if context.strip():
#         base += f" {clean_text(context)}"
#         if not base.endswith("."):
#             base += "."

#     if user_desc.strip():
#         base += f" {clean_text(user_desc)}"
#         if not base.endswith("."):
#             base += "."

#     return clean_text(base)

# def make_hashtags(captions: List[str], keywords: List[str], limit: int = 15) -> List[str]:
#     joined = " ".join(captions).lower()
#     tags = []

#     # keyword-based tags
#     for k in keywords:
#         if k.isalpha():
#             tags.append(f"#{k}")

#     # add known travel tags if present in caption text
#     for k in KEYWORD_TAGS:
#         if k in joined and f"#{k}" not in tags:
#             tags.append(f"#{k}")

#     # ensure base tags
#     for t in ["#travel", "#nature", "#photography"]:
#         if t not in tags:
#             tags.append(t)

#     # de-dup preserve order
#     seen = set()
#     out = []
#     for t in tags:
#         if t not in seen:
#             seen.add(t)
#             out.append(t)

#     return out[:limit]

import re
import random
from collections import Counter
from typing import List

# -------------------------
# Config
# -------------------------

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

# -------------------------
# Text helpers
# -------------------------

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return s


# -------------------------
# Keyword extraction
# -------------------------

def extract_keywords(captions: List[str], top_k: int = 6) -> List[str]:
    text = " ".join(captions).lower()
    words = re.findall(r"[a-z]+", text)
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_k)]


# -------------------------
# Title generation
# -------------------------

def make_title_from_keywords(keywords: List[str]) -> str:
    if not keywords:
        return "Travel Moment"

    phrase = keywords[:4]  # 2–4 words
    return " ".join(phrase).title()


# -------------------------
# Description generation
# -------------------------

DESCRIPTION_OPENERS = [
    "This moment captures",
    "Captured during a quiet moment of",
    "A glimpse into",
    "An unforgettable scene of",
    "A striking view showcasing",
    "This frame tells the story of",
    "A timeless capture featuring",
    "An authentic moment drawn from",
    "A powerful visual representing",
    "A carefully framed moment highlighting",
    "This image reflects",
    "A scenic portrayal of",
    "An expressive snapshot of",
    "A natural composition revealing",
    "A visually rich moment centered around",
]

DESCRIPTION_MIDDLES = [
    "where {k1} meets {k2}",
    "highlighting the harmony between {k1} and {k2}",
    "shaped by {k1}, {k2}, and the surrounding landscape",
    "defined by {k1} and the raw presence of {k2}",
    "surrounded by {k1} with a strong sense of {k2}",
    "framed by {k1} while embracing the essence of {k2}",
    "marked by {k1} and influenced by {k2}",
    "built around the contrast between {k1} and {k2}",
]

DESCRIPTION_EXPANSIONS = [
    "The environment adds depth, scale, and a sense of direction to the scene.",
    "Subtle details throughout the frame strengthen the visual narrative.",
    "Natural elements combine to create a balanced and immersive atmosphere.",
    "The setting enhances the feeling of movement and exploration.",
    "Light, texture, and composition work together seamlessly.",
    "The surrounding landscape plays a key role in defining the mood.",
]

DESCRIPTION_ENDINGS = [
    "It reflects the spirit of exploration and travel.",
    "The scene leaves a lasting impression of freedom and calm.",
    "It captures the essence of adventure in its purest form.",
    "The moment feels authentic, grounded, and visually compelling.",
    "It represents a meaningful connection with nature.",
]

def make_description(
    captions: List[str],
    keywords: List[str],
    tone: str = "fun",
    context: str = "",
    user_desc: str = "",
    title: str = "",
) -> str:
    keywords = keywords[:5] if keywords else []

    k1 = keywords[0] if len(keywords) > 0 else "nature"
    k2 = keywords[1] if len(keywords) > 1 else "travel"

    opener = random.choice(DESCRIPTION_OPENERS)
    middle = random.choice(DESCRIPTION_MIDDLES).format(k1=k1, k2=k2)
    expansion = random.choice(DESCRIPTION_EXPANSIONS)
    ending = random.choice(DESCRIPTION_ENDINGS)

    # Sentence 1 — visual meaning
    sentence1 = f"{opener} {k1} and {k2}, {middle}."

    # Sentence 2 — depth & environment
    sentence2 = expansion

    # Sentence 3 — title alignment
    if title:
        sentence3 = (
            f"It naturally aligns with the theme of “{title}”, "
            "reinforcing the overall atmosphere of the moment."
        )
    else:
        sentence3 = ending

    # Tone handling
    tone = (tone or "").lower().strip()
    if tone == "fun":
        sentence3 += " The scene feels lively, engaging, and full of character."
    elif tone == "formal":
        sentence3 = sentence3.replace(
            "The scene",
            "The composition"
        ) + " The visual structure remains clean and refined."
    elif tone == "romantic":
        sentence3 += " Soft light and natural textures add emotional warmth."

    description = " ".join([sentence1, sentence2, sentence3])

    if context.strip():
        description += " " + clean_text(context)

    if user_desc.strip():
        description += " " + clean_text(user_desc)

    return clean_text(description)


# -------------------------
# Hashtag generation
# -------------------------

def make_hashtags(captions: List[str], keywords: List[str], limit: int = 15) -> List[str]:
    joined = " ".join(captions).lower()
    tags = []

    # keyword-based hashtags
    for k in keywords:
        if k.isalpha():
            tags.append(f"#{k}")

    # known travel tags
    for k in KEYWORD_TAGS:
        if k in joined and f"#{k}" not in tags:
            tags.append(f"#{k}")

    # base tags
    for t in ["#travel", "#nature", "#photography"]:
        if t not in tags:
            tags.append(t)

    # dedupe while preserving order
    seen = set()
    out = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)

    return out[:limit]
