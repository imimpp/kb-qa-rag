"""[013] 入册探针: 复算 #31 答案相交块在 [012] 融合候选池里的名次(零 API).

背景: [012] 记录原句「标题块被挤出候选池」待核实。本探针用冻结问题向量＋
meta 现建 BM25＋生产 rrf_merge 重算固定窗臂候选池, 看标题块到底在不在池里、
排第几; 再对照 [012] 落盘 top5 与答案区间。运行: .venv/Scripts/python.exe experiments/probe_013_title_pool.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import jieba
import numpy as np
import embed_index

INDEX_DIR = ROOT / "data/index"
meta = json.load(open(INDEX_DIR / "meta.json", encoding="utf-8"))
vectors = np.load(INDEX_DIR / "vectors.npy")
questions = json.load(open(ROOT / "experiments" / "eval_set.json", encoding="utf-8"))["questions"]
frozen = np.load(ROOT / "experiments" / "query_vectors.npy")
q = next(q for q in questions if q["id"] == 31)
q_vec = frozen[questions.index(q)]
q_vec = q_vec / np.linalg.norm(q_vec)

scores = vectors.dot(q_vec)
bm25 = embed_index.build_bm25(meta)
bm25_scores = bm25.get_scores(list(jieba.cut(q["question"])))
bm25_idx = np.argsort(bm25_scores)[::-1][:embed_index.POOL_N]
vec_idx = np.argsort(scores)[::-1][:embed_index.POOL_N]
fused = embed_index.rrf_merge(vec_idx, bm25_idx)
pool = [idx for idx, _ in fused[:embed_index.POOL_N]]

answer = q["answers"][0]
overlap_rows = [
    i for i, c in enumerate(meta["chunks"])
    if c["source"] == answer["source"]
    and c["start"] < answer["end"] and c["end"] > answer["start"]
]
print("answer:", answer["source"], f"[{answer['start']}:{answer['end']}]")
for i in overlap_rows:
    c = meta["chunks"][i]
    rank = pool.index(i) + 1 if i in pool else None
    print(f"row {i} blocks[{c['start']}:{c['end']}] in_pool={i in pool} pool_rank={rank}")

old = json.load(open(ROOT / "experiments" / "eval_results_012_hybrid.json", encoding="utf-8"))
rec = next(r for r in old["questions"] if r["id"] == 31)
print("[012] #31 top5:", [(t["source"], t["start"], t["end"], round(t["score"], 3)) for t in rec["top5"]])
print("[012] #31 hit:", rec["hit"])