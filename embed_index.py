"""检索最小实现:建索引+查询两层.

建索引:parsed JSON->切块->embed向量->vectors.npy+meta.json
查询:问题->embed->与矩阵点积->top-k->行号回meta取原文
"""
import json
from pathlib import Path


import chunking

from datetime import datetime
import numpy as np
import llm_client

PARSED_DIR = Path("data/parsed")
EXPECTED_DOCS = 43  # 实测语料份数,防静默少读
INDEX_DIR = Path("data/index")
EMBED_MODEL = "qwen3.7-text-embedding-flash"
BATCH_SIZE = 20  # 实测上限:25 直接400 batch size invalid


def load_chunks():
    """读全部parse JSON,逐份切块,按文档顺序返回所有chunk."""
    chunks = []
    doc_count = 0
    for path in sorted(PARSED_DIR.glob("*.json")):
        if path.name.startswith("~$"):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            chunks.extend(chunking.chunk_fixed_window(blocks=data["blocks"], source=data["source"]))
        doc_count += 1
    assert doc_count == EXPECTED_DOCS
    return chunks


def build_index():
    chunks = load_chunks()
    client = llm_client.LLMClient()
    vectors = []
    metas = []
    for i in range(0,len(chunks),BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        vecs = client.embed([c["text"] for c in batch],model=EMBED_MODEL)
        vectors.extend(vecs)
        metas.extend(batch)#同一循环双写:vectors第i行<->metas第i行
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norm
    assert len(vectors) == len(metas) ==len(chunks)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy",np.asarray(vectors,dtype=np.float32))
    entries=[]
    for c in metas:
        entries.append({
            "text": c["text"],
            "source": c["source"],
            "start": c["start"],
            "end": c["end"],
            "len": len(c["text"]),
        })
    usage = client.get_cost()[EMBED_MODEL]
    meta = {
        "model": EMBED_MODEL,
        "dim": len(vectors[0]),
        "count": len(metas),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "chunk_params": {"window_chars": 500, "overlap_chars": 100},
        "total_tokens":usage["input_tokens"],
        "cost_yuan": usage["cost"],
        "chunks":entries,
    }
    with open(INDEX_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f,ensure_ascii=False, indent=2)
    print(f"indexed={len(metas)},tokens={usage['input_tokens']},cost_yuan={usage['cost']:.4f}")






if __name__ == "__main__":
    build_index()
