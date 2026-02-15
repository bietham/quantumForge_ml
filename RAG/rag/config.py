import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    store_dir: str = os.getenv("RAG_STORE_DIR", "artifacts/faiss_store")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
    top_k: int = int(os.getenv("TOP_K", "6"))
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", "0.22"))

settings = Settings()
