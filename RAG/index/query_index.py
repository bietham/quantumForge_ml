import argparse
import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return v / norms


def load_chunks(chunks_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="artifacts/faiss_store", help="Path to faiss_store directory")
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2", help="Embedding model name")
    parser.add_argument("--q", required=True, help="Query text")
    parser.add_argument("--k", type=int, default=5, help="Top-k")
    args = parser.parse_args()

    store = Path(args.store)
    index_path = store / "index.faiss"
    chunks_path = store / "chunks.jsonl"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("FAISS store not found. Build index first (docker compose up indexer).")

    index = faiss.read_index(str(index_path))
    chunks = load_chunks(chunks_path)

    model = SentenceTransformer(args.model)
    q_emb = model.encode([args.q], convert_to_numpy=True).astype("float32")
    q_emb = l2_normalize(q_emb)

    scores, ids = index.search(q_emb, args.k)

    print("\nQUERY:")
    print(args.q)
    print("\nRESULTS:")
    for rank, (cid, score) in enumerate(zip(ids[0].tolist(), scores[0].tolist()), start=1):
        if cid < 0 or cid >= len(chunks):
            continue
        item = chunks[cid]
        meta = item["metadata"]
        snippet = item["text"].replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        print(f"\n#{rank}  score={score:.4f}  chunk_id={cid}")
        print(f"source_path: {meta.get('source_path')}")
        print(f"title:       {meta.get('title')}")
        print(f"doc_id:      {meta.get('doc_id')}")
        print(f"chunk_index: {meta.get('chunk_index')}  chars={meta.get('start_char')}-{meta.get('end_char')}")
        print(f"snippet:     {snippet}")


if __name__ == "__main__":
    main()
