from typing import List, Dict, Any, Tuple
from PIL import Image

from .utils import extract_keywords, make_title_from_keywords, make_description, make_hashtags

SUGGESTIONS = [
    "Try outdoor scenery (mountains, beaches, forests, roads).",
    "Avoid screenshots, documents, or indoor-only photos.",
    "Landmarks and wide landscape shots work best.",
    "Upload clearer images with good lighting."
]

def run_multi_image_pipeline(
    images: List[Image.Image],
    platform: str,
    tone: str,
    context: str,
    user_title: str,
    user_description: str,
    travel_probs: List[float],
    caption_fn,
    travel_threshold: float = 0.60
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    caption_fn: function(Image)->str from caption model
    Returns (response_dict, debug_dict)
    """
    per_img = []
    is_travel_flags = []
    for i, p in enumerate(travel_probs):
        is_travel = p >= 0.5
        is_travel_flags.append(is_travel)
        per_img.append({"index": i, "travel_prob": float(p), "is_travel": bool(is_travel)})

    travel_ratio = sum(is_travel_flags) / max(len(is_travel_flags), 1)

    debug = {
        "per_image": per_img,
        "travel_ratio": float(travel_ratio),
    }

    if travel_ratio < travel_threshold:
        return (
            {
                "valid": False,
                "message": "This doesn't look like travel photos.",
                "suggestions": SUGGESTIONS
            },
            debug
        )

    # Generate per-image captions for travel images only
    captions = []
    for img, ok in zip(images, is_travel_flags):
        if ok:
            captions.append(caption_fn(img))

    if not captions:
        return (
            {
                "valid": False,
                "message": "Could not generate captions from the provided photos.",
                "suggestions": SUGGESTIONS
            },
            debug
        )

    # Merge into one final output
    keywords = extract_keywords(captions, top_k=8)

    # If user_title is provided, respect it; else auto title
    title = user_title.strip() if user_title.strip() else make_title_from_keywords(keywords)

    # Description: merged + optional context/user_description
    desc = make_description(captions, keywords, tone=tone, context=context, user_desc=user_description)

    # Hashtags: from captions + keywords
    hashtags = make_hashtags(captions, keywords, limit=15)

    # Small platform tweak (optional)
    platform = (platform or "").lower().strip()
    if platform == "linkedin":
        # LinkedIn: fewer hashtags usually
        hashtags = hashtags[:7]
    elif platform == "facebook":
        hashtags = hashtags[:10]

    return (
        {
            "valid": True,
            "title": title,
            "description": desc,
            "hashtags": hashtags
        },
        debug
    )
