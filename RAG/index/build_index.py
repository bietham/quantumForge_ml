import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    source_path: str
    title: str
    lang: str
    chunk_index: int
    start_char: int
    end_char: int
    text: str


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSONL at line {line_no}: {e}") from e
    return rows


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(ln.rstrip() for ln in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def build_chunks(rows: List[Dict], chunk_size: int, overlap: int) -> List[Chunk]:
    out: List[Chunk] = []
    global_chunk_id = 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for row in rows:
        doc_id = str(row.get("id", "")).strip()
        source_path = str(row.get("path", "")).strip()
        title = str(row.get("title") or doc_id or source_path).strip() or "untitled"
        lang = str(row.get("lang") or "en").strip()
        text = normalize_text(str(row.get("text", "")))

        if not doc_id or not source_path or not text:
            continue

        pieces = splitter.split_text(text)
        cursor = 0

        for chunk_index, ctext in enumerate(pieces):
            ctext = ctext.strip()
            if not ctext:
                continue

            pos = text.find(ctext, cursor)
            if pos == -1:
                pos = cursor
            start_char = pos
            end_char = min(pos + len(ctext), len(text))
            cursor = max(end_char - overlap, 0)

            out.append(
                Chunk(
                    chunk_id=global_chunk_id,
                    doc_id=doc_id,
                    source_path=source_path,
                    title=title,
                    lang=lang,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    text=ctext,
                )
            )
            global_chunk_id += 1

    return out


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return v / norms


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_chunks_jsonl(chunks: List[Chunk], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": ch.chunk_id,
                        "text": ch.text,
                        "metadata": {
                            "doc_id": ch.doc_id,
                            "source_path": ch.source_path,
                            "title": ch.title,
                            "lang": ch.lang,
                            "chunk_index": ch.chunk_index,
                            "start_char": ch.start_char,
                            "end_char": ch.end_char,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", default="knowledge_base/kb_corpus.jsonl", help="Path to kb_corpus.jsonl")
    parser.add_argument("--out", default="artifacts/faiss_store", help="Output directory for FAISS store")
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2", help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=1600, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=250, help="Overlap in characters")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    kb_path = Path(args.kb)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    if not kb_path.exists():
        raise FileNotFoundError(f"KB file not found: {kb_path}")

    t0 = time.time()
    rows = load_jsonl(kb_path)
    t_load = time.time()

    chunks = build_chunks(rows, chunk_size=args.chunk_size, overlap=args.overlap)
    if not chunks:
        raise RuntimeError("No chunks were produced. Check kb_corpus.jsonl content.")
    t_chunk = time.time()

    model = SentenceTransformer(args.model)
    texts = [c.text for c in chunks]

    embeddings_list = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Embedding"):
        batch = texts[i : i + args.batch_size]
        emb = model.encode(
            batch,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embeddings_list.append(emb)

    emb = np.vstack(embeddings_list).astype("float32")
    emb = l2_normalize(emb)
    t_embed = time.time()

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    t_index = time.time()

    faiss_path = out_dir / "index.faiss"
    chunks_path = out_dir / "chunks.jsonl"
    manifest_path = out_dir / "manifest.json"

    faiss.write_index(index, str(faiss_path))
    save_chunks_jsonl(chunks, chunks_path)

    manifest = {
        "kb": str(kb_path),
        "model": args.model,
        "embedding_dim": dim,
        "chunk_size_chars": args.chunk_size,
        "overlap_chars": args.overlap,
        "batch_size": args.batch_size,
        "docs_in_kb": len(rows),
        "chunks_total": len(chunks),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timing_sec": {
            "load_kb": round(t_load - t0, 3),
            "chunking": round(t_chunk - t_load, 3),
            "embeddings": round(t_embed - t_chunk, 3),
            "faiss_index": round(t_index - t_embed, 3),
            "total": round(time.time() - t0, 3),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDONE")
    print(f"Docs read:        {len(rows)}")
    print(f"Chunks created:   {len(chunks)}")
    print(f"Embedding dim:    {dim}")
    print(f"Saved index:      {faiss_path}")
    print(f"Saved chunks:     {chunks_path}")
    print(f"Saved manifest:   {manifest_path}")
    print(f"Total time (sec): {manifest['timing_sec']['total']}")


if __name__ == "__main__":
    main()
