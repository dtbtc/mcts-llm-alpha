"""用于alpha生成的LLM集成模块。"""

from .client import LLMClient
from .prompts import FIELDS, OPS, DIMENSION_GUIDANCE

__all__ = [
    "LLMClient",
    "FIELDS",
    "OPS", 
    "DIMENSION_GUIDANCE"
]