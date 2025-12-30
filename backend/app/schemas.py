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
