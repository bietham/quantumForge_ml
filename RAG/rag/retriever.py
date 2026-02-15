import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return v / norms


@dataclass
class RetrievedChunk:
    chunk_id: int
    score: float
    text: str
    metadata: Dict


class FaissRetriever:
    def __init__(self, store_dir: str, embed_model: str):
        self.store = Path(store_dir)
        self.index_path = self.store / "index.faiss"
        self.chunks_path = self.store / "chunks.jsonl"

        if not self.index_path.exists() or not self.chunks_path.exists():
            raise FileNotFoundError(f"FAISS store not found in {self.store}. Build index first.")

        self.index = faiss.read_index(str(self.index_path))
        self.chunks = self._load_chunks(self.chunks_path)
        self.model = SentenceTransformer(embed_model)

    @staticmethod
    def _load_chunks(path: Path) -> List[Dict]:
        rows: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        q_emb = l2_normalize(q_emb)
        scores, ids = self.index.search(q_emb, top_k)

        out: List[RetrievedChunk] = []
        for cid, score in zip(ids[0].tolist(), scores[0].tolist()):
            if cid < 0 or cid >= len(self.chunks):
                continue
            item = self.chunks[cid]
            out.append(RetrievedChunk(
                chunk_id=cid,
                score=float(score),
                text=item["text"],
                metadata=item["metadata"],
            ))
        return out
