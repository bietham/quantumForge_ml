"""02_apply_terms_map.py

Назначение:
  Подменить LOTR-термины на вымышленные, чтобы модель не могла отвечать «по памяти»
  и была вынуждена опираться на ваш RAG-индекс.

Как работает:
  - загружает terms_map.json (исходный термин -> вымышленный термин)
  - проходит по файлам в knowledge_base_raw/
  - применяет замены (от длинных терминов к коротким, без учёта регистра)
  - сохраняет результат в knowledge_base/
  - дополнительно собирает knowledge_base/kb_corpus.jsonl

Использование:
  python scripts/02_apply_terms_map.py
"""

import json, re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "knowledge_base_raw"
OUT = BASE / "knowledge_base"
OUT.mkdir(exist_ok=True)

TERMS = json.loads((BASE / "terms_map.json").read_text(encoding="utf-8"))

def apply_map(text: str, mapping: dict) -> str:
    # Сначала длинные термины, чтобы не ломать составные сущности
    items = sorted(mapping.items(), key=lambda kv: (-len(kv[0]), kv[0]))
    out = text
    for src, dst in items:
        out = re.compile(re.escape(src), re.IGNORECASE).sub(dst, out)
    return out

def main():
    for p in RAW.glob("*.md"):
        new_text = apply_map(p.read_text(encoding="utf-8"), TERMS)
        (OUT / p.name).write_text(new_text, encoding="utf-8")

    # JSONL корпус для загрузки в пайплайн
    jsonl = OUT / "kb_corpus.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for p in sorted(OUT.glob("*.md")):
            obj = {"id": p.stem, "path": str(p.relative_to(BASE)), "text": p.read_text(encoding="utf-8")}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
