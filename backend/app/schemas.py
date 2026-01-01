from pydantic import BaseModel
from typing import List, Optional

class CaptionResponse(BaseModel):
    valid: bool
    title: Optional[str] = None
    description: Optional[str] = None
    hashtags: Optional[List[str]] = None
    message: Optional[str] = None
    suggestions: Optional[List[str]] = None

class DebugImageResult(BaseModel):
    index: int
    travel_prob: float
    is_travel: bool

class DebugInfo(BaseModel):
    per_image: List[DebugImageResult]
    travel_ratio: float


### NEW CODE
# from pydantic import BaseModel
# from typing import List, Optional, Dict, Any

# class CaptionResponse(BaseModel):
#     valid: bool
#     mode: Optional[str] = "general"

#     title: Optional[str] = None
#     description: Optional[str] = None
#     hashtags: Optional[List[str]] = None

#     # NEW (important for regenerate)
#     raw_caption: Optional[str] = None
#     analysis: Optional[Dict[str, Any]] = None

#     message: Optional[str] = None
#     suggestions: Optional[List[str]] = None
