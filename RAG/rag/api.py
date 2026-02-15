from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import settings
from .retriever import FaissRetriever, RetrievedChunk
from .safety import filter_suspicious
from .prompting import build_prompts
from .llm import chat_completion

app = FastAPI(title="QuantumForge RAG Bot (demo)")

retriever: Optional[FaissRetriever] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = None
    protection: Literal["none", "pre", "post", "both"] = "both"


class AskResponse(BaseModel):
    answer: str
    used_chunks: list
    note: str


@app.on_event("startup")
def _startup():
    global retriever
    retriever = FaissRetriever(settings.store_dir, settings.embed_model)


def _should_say_idk(chunks: list[RetrievedChunk], threshold: float) -> bool:
    if not chunks:
        return True
    return chunks[0].score < threshold


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    assert retriever is not None

    top_k = req.top_k or settings.top_k
    chunks = retriever.search(req.question, top_k=top_k)

    if req.protection in ("post", "both"):
        chunks = filter_suspicious(chunks)

    if _should_say_idk(chunks, settings.score_threshold):
        return AskResponse(
            answer="I don't know.",
            used_chunks=[],
            note="No relevant context found in the knowledge base (or it was filtered).",
        )

    prompts = build_prompts(req.question, chunks, protection=req.protection)

    text = chat_completion(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        system=prompts.system,
        user=prompts.user,
    )

    used = []
    for c in chunks:
        used.append({
            "chunk_id": c.chunk_id,
            "score": c.score,
            "source_path": c.metadata.get("source_path"),
            "title": c.metadata.get("title"),
            "chunk_index": c.metadata.get("chunk_index"),
        })

    return AskResponse(
        answer=text.strip(),
        used_chunks=used,
        note="Answer generated with retrieval + prompting (few-shot + short steps).",
    )


@app.get("/health")
def health():
    return {"ok": True}
