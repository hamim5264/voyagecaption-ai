# from typing import List, Dict, Any, Tuple
# from PIL import Image

# from .utils import extract_keywords, make_title_from_keywords, make_description, make_hashtags

# SUGGESTIONS = [
#     "Try outdoor scenery (mountains, beaches, forests, roads).",
#     "Avoid screenshots, documents, or indoor-only photos.",
#     "Landmarks and wide landscape shots work best.",
#     "Upload clearer images with good lighting."
# ]

# def run_multi_image_pipeline(
#     images: List[Image.Image],
#     platform: str,
#     tone: str,
#     context: str,
#     user_title: str,
#     user_description: str,
#     travel_probs: List[float],
#     caption_fn,
#     travel_threshold: float = 0.60
# ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#     """
#     caption_fn: function(Image)->str from caption model
#     Returns (response_dict, debug_dict)
#     """
#     per_img = []
#     is_travel_flags = []
#     for i, p in enumerate(travel_probs):
#         is_travel = p >= 0.5
#         is_travel_flags.append(is_travel)
#         per_img.append({"index": i, "travel_prob": float(p), "is_travel": bool(is_travel)})

#     travel_ratio = sum(is_travel_flags) / max(len(is_travel_flags), 1)

#     debug = {
#         "per_image": per_img,
#         "travel_ratio": float(travel_ratio),
#     }

#     if travel_ratio < travel_threshold:
#         return (
#             {
#                 "valid": False,
#                 "message": "This doesn't look like travel photos.",
#                 "suggestions": SUGGESTIONS
#             },
#             debug
#         )

#     # Generate per-image captions for travel images only
#     captions = []
#     for img, ok in zip(images, is_travel_flags):
#         if ok:
#             captions.append(caption_fn(img))

#     if not captions:
#         return (
#             {
#                 "valid": False,
#                 "message": "Could not generate captions from the provided photos.",
#                 "suggestions": SUGGESTIONS
#             },
#             debug
#         )

#     # Merge into one final output
#     keywords = extract_keywords(captions, top_k=8)

#     # If user_title is provided, respect it; else auto title
#     title = user_title.strip() if user_title.strip() else make_title_from_keywords(keywords)

#     # Description: merged + optional context/user_description
#     desc = make_description(captions, keywords, tone=tone, context=context, user_desc=user_description)

#     # Hashtags: from captions + keywords
#     hashtags = make_hashtags(captions, keywords, limit=15)

#     # Small platform tweak (optional)
#     platform = (platform or "").lower().strip()
#     if platform == "linkedin":
#         # LinkedIn: fewer hashtags usually
#         hashtags = hashtags[:7]
#     elif platform == "facebook":
#         hashtags = hashtags[:10]

#     return (
#         {
#             "valid": True,
#             "title": title,
#             "description": desc,
#             "hashtags": hashtags
#         },
#         debug
#     )


from typing import List, Dict, Any, Tuple
from PIL import Image

from .utils import (
    extract_keywords,
    make_title_from_keywords,
    make_description,
    make_hashtags,
)

def run_multi_image_pipeline(
    images: List[Image.Image],
    platform: str,
    tone: str,
    context: str,
    user_title: str,
    user_description: str,
    travel_probs: List[float],  # kept for future use
    caption_fn,
    travel_threshold: float = 0.0,  # unused now
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Caption-first pipeline.
    No travel validation.
    Always generate captions.
    """

    # -------------------------
    # 1️⃣ GENERATE RAW CAPTIONS
    # -------------------------
    captions: List[str] = []

    for img in images:
        try:
            captions.append(caption_fn(img))
        except Exception:
            continue

    if not captions:
        return (
            {
                "valid": False,
                "message": "Could not generate captions from images.",
                "suggestions": ["Try clearer images with better lighting."],
            },
            {},
        )

    # -------------------------
    # 2️⃣ KEYWORD EXTRACTION
    # -------------------------
    keywords = extract_keywords(captions, top_k=8)

    # -------------------------
    # 3️⃣ TITLE
    # -------------------------
    title = (
        user_title.strip()
        if user_title.strip()
        else make_title_from_keywords(keywords)
    )

    # -------------------------
    # 4️⃣ DESCRIPTION
    # -------------------------
    description = make_description(
        captions=captions,
        keywords=keywords,
        tone=tone,
        context=context,
        user_desc=user_description,
    )

    # -------------------------
    # 5️⃣ HASHTAGS
    # -------------------------
    hashtags = make_hashtags(captions, keywords, limit=15)

    # Platform tuning
    platform = (platform or "").lower()
    if platform == "linkedin":
        hashtags = hashtags[:6]
    elif platform == "facebook":
        hashtags = hashtags[:10]

    # -------------------------
    # 6️⃣ DEBUG (OPTIONAL)
    # -------------------------
    debug = {
        "images_used": len(images),
        "captions_generated": len(captions),
        "keywords": keywords,
    }

    # -------------------------
    # 7️⃣ FINAL RESPONSE
    # -------------------------
    return (
        {
            "valid": True,
            "title": title,
            "description": description,
            "hashtags": hashtags,
            "debug": debug,
        },
        debug,
    )

### NEW CODE
# from typing import List, Dict, Any, Tuple, Optional
# from PIL import Image

# from .utils import (
#     infer_scene,
#     make_title_from_caption,
#     make_description_from_caption,
#     make_hashtags
# )

# SUGGESTIONS = [
#     "Try clearer images with better lighting.",
#     "Avoid screenshots or text-heavy images.",
#     "Natural moments work best."
# ]

# def run_multi_image_pipeline(
#     images: List[Image.Image],
#     caption_fn,
#     mode: str = "general",
#     travel_probs: Optional[List[float]] = None,
#     travel_threshold: float = 0.60,
#     platform: str = ""
# ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#     """
#     Final caption pipeline.
#     """

#     if travel_probs is None:
#         travel_probs = [1.0] * len(images)

#     # -----------------------------
#     # RAW CAPTIONS (AI OUTPUT)
#     # -----------------------------
#     raw_captions: List[str] = []
#     for img in images:
#         try:
#             raw_captions.append(caption_fn(img))
#         except Exception:
#             continue

#     if not raw_captions:
#         return (
#             {
#                 "valid": False,
#                 "message": "Could not understand the images.",
#                 "suggestions": SUGGESTIONS
#             },
#             {}
#         )

#     merged_caption = " ".join(raw_captions)

#     # -----------------------------
#     # OPTIONAL TRAVEL VALIDATION
#     # -----------------------------
#     if mode == "travel":
#         valid_count = sum(p >= travel_threshold for p in travel_probs)
#         travel_ratio = valid_count / max(len(travel_probs), 1)

#         if travel_ratio < travel_threshold:
#             return (
#                 {
#                     "valid": False,
#                     "mode": mode,
#                     "message": "These images do not appear to be travel related.",
#                     "suggestions": SUGGESTIONS
#                 },
#                 {
#                     "travel_ratio": travel_ratio
#                 }
#             )

#     # -----------------------------
#     # SEMANTIC ANALYSIS
#     # -----------------------------
#     scene = infer_scene(merged_caption)

#     analysis = {
#         "scene": scene,
#         "images_used": len(images),
#     }

#     # -----------------------------
#     # HUMAN OUTPUT (RAW-CAPTION DRIVEN)
#     # -----------------------------
#     title = make_title_from_caption(merged_caption)
#     description = make_description_from_caption(merged_caption, mode)
#     hashtags = make_hashtags(scene)

#     # Platform tweaks
#     platform = (platform or "").lower().strip()
#     if platform == "linkedin":
#         hashtags = hashtags[:5]
#     elif platform == "facebook":
#         hashtags = hashtags[:8]

#     return (
#         {
#             "valid": True,
#             "mode": mode,
#             "title": title,
#             "description": description,
#             "hashtags": hashtags,
#             "raw_caption": merged_caption,
#             "analysis": analysis,
#         },
#         analysis
#     )
