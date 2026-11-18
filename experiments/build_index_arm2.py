"""
建索引: 递归分隔符切块法。

流程: 读 parsed JSON -> 整篇递归分隔符切块(500, 不加 overlap)
-> embed -> 落 data/index_arm2/(vectors.npy+meta.json)。

start:end 由偏移表反查原始 block 下标; 超窗大表被切中间时
记整表区间, 比块实际内容宽, 评测命中判断偏松。
产物形状与 data/index 完全一致, exp_009 的 evaluate 直接指过来就能评。

覆写保护: 块向量也来自非确定 API, 重建=换签, 与已入册结果不可比;
真要重建须先手动移走旧目录。
运行: PyCharm 右键 Run(路径全锚定, 不依赖工作目录)。
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 才能 import 根目录的模块
os.chdir(ROOT)
import numpy as np
import chunking
import llm_client

PARSED_DIR = ROOT / "data/parsed"
EXPECTED_DOCS = 43  # 实测语料份数, 防静默少读
OUT_DIR = ROOT / "data/index_arm2"
EMBED_MODEL = "qwen3.7-text-embedding-flash"  # 与另两臂索引同模型, 不同切块法之间才可比
BATCH_SIZE = 20  # 实测上限: 25 直接 400 batch size invalid
WINDOW_CHARS = 500


def load_chunks():
    """逐份: 整篇递归切块, start:end 已是原始 block 下标, 无需映射."""
    chunks = []
    doc_count = 0
    for path in sorted(PARSED_DIR.glob("*.json")):
        if path.name.startswith("~$"):
            continue  # Word 瞬态锁文件, 二进制垃圾
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        doc_chunks = chunking.chunk_recursive(data["blocks"], window_chars=WINDOW_CHARS,
                                              source=data["source"])
        chunks.extend(doc_chunks)
        doc_count += 1
    assert doc_count == EXPECTED_DOCS
    return chunks


def main():
    if (OUT_DIR / "vectors.npy").exists():
        raise FileExistsError(f"{OUT_DIR} 已有索引, 拒绝覆盖(重建=换签); 真要重建先手动移走旧目录")
    chunks = load_chunks()
    client = llm_client.LLMClient()
    vectors = []
    metas = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vecs = client.embed([c["text"] for c in batch], model=EMBED_MODEL)
        vectors.extend(vecs)
        metas.extend(batch)  # 同一循环双写: vectors 第 i 行 <-> metas 第 i 行, 行序即身份
        print(f"batch {i // BATCH_SIZE + 1}: {len(metas)}/{len(chunks)}")
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norm  # L2 归一化后点积即余弦, 查询不必再算模长
    assert len(vectors) == len(metas) == len(chunks)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "vectors.npy", np.asarray(vectors, dtype=np.float32))
    entries = []
    for c in metas:
        entries.append({
            "text": c["text"],
            "source": c["source"],
            "start": c["start"],
            "end": c["end"],
            "len": len(c["text"]),
        })
    usage = client.get_cost()[EMBED_MODEL]
    over = [e for e in entries if e["len"] > WINDOW_CHARS]
    meta = {
        "model": EMBED_MODEL,
        "dim": len(vectors[0]),
        "count": len(metas),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "chunk_params": {"window_chars": WINDOW_CHARS, "overlap_chars": 0,
                         "preprocess": "整篇递归分隔符切块＋偏移表反查, 超窗大表按整表区间标注"},
        "total_tokens": usage["input_tokens"],
        "cost_yuan": usage["cost"],
        "chunks": entries,
    }
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"indexed={len(metas)},tokens={usage['input_tokens']},cost_yuan={usage['cost']:.4f}")
    print(f"over_window={len(over)},max_len={max(e['len'] for e in entries)}")
    print(f"saved={OUT_DIR}")


if __name__ == "__main__":
    main()