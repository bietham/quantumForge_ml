import re
from typing import List

from .retriever import RetrievedChunk

SUSPICIOUS_PATTERNS = [
    r"ignore\s+all\s+instructions",
    r"output\s*:",
    r"swordfish",
    r"superpassword",
    r"root\s*:",
]

SUSPICIOUS_RE = re.compile("|".join(SUSPICIOUS_PATTERNS), re.IGNORECASE)

def is_suspicious_text(text: str) -> bool:
    return bool(SUSPICIOUS_RE.search(text))

def filter_suspicious(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    return [c for c in chunks if not is_suspicious_text(c.text)]

def sanitize_context(text: str) -> str:
    text = re.sub(r"(?i)ignore\s+all\s+instructions\.?\s*", "", text)
    text = re.sub(r"(?i)output\s*:\s*", "", text)
    return text
