"""
[010a] 语义切块前置扫描: 全库相邻句相似度分布.

口径(2026-11-16 定): 句=段落按 。！？;换行切、表格按行切; <8字碎片丢弃、
纯分隔行丢弃; 相邻对=同文档内连续句, 不跨文档; 分层=段对段(p_p) vs
含表格行(with_table); 抽检=p_p 最低10对+最高5对, 人工判是否真话题断点。
分布双峰可分->做语义臂并定阈值; 单峰糊住->语义臂放弃。
只读 data/parsed/ 并调 embed, 不改 chunking.py / embed_index.py。
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根的 llm_client 不在 experiments/ 下
os.chdir(ROOT)  # PyCharm Run 工作目录默认脚本目录, 锚定仓库根
import llm_client

PARSED_DIR = ROOT / "data" / "parsed"
EXPECTED_DOCS = 43  # 断言防静默少读
MIN_SENT_LEN = 8  # 「（一）」这类切分残渣不算句单元
EMBED_MODEL = "qwen3.7-text-embedding-flash"  # 与索引同模型, 换模型相似度不可比
BATCH_SIZE = 20  # API 批上限实测 20, 25 报 400
SEP_ROW = re.compile(r"^[|\-:\s]+$")  # |---| 纯表格排版行, 不含文字
SENT_SPLIT = re.compile(r"[。！？；\n]")


def block_units(block):
    """block -> (kind, 句列表). paragraph 按 SENT_SPLIT 切、table 按行切;
    两者均丢 <MIN_SENT_LEN 碎片, table 另丢纯分隔行."""
    if block["type"] == "paragraph":
        kind = "paragraph"
        texts = [t.strip() for t in SENT_SPLIT.split(block["text"]) if len(t) >= MIN_SENT_LEN]
    else:
        kind = "table"
        texts = [line.strip() for line in block["markdown"].split("\n") if len(line) >= MIN_SENT_LEN and not SEP_ROW.match(line)]
    return kind, texts


def load_units():
    units = []
    doc_count = 0
    for path in sorted(PARSED_DIR.glob("*.json")):  # glob 顺序不定, sorted 保两次运行一致
        if path.name.startswith("~$"):
            continue  # Word 瞬态锁文件, 二进制垃圾
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for block in data["blocks"]:  # 列表顺序=原文档顺序(解析层契约)
            kind, texts = block_units(block)
            units.extend((data["source"], kind, t) for t in texts)
        doc_count += 1
    assert doc_count == EXPECTED_DOCS
    return units


def embed_all(texts):
    client = llm_client.LLMClient()
    vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        vecs.extend(
            client.embed(texts[i:i + BATCH_SIZE], model=EMBED_MODEL))
    arr = np.asarray(vecs, dtype=np.float32)  # float32 省一半内存
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)
    # L2 归一化后行点积=余弦相似度, 全脚本「相似度」的算法含义


def make_pairs(units, vecs):
    pairs = []
    for i in range(len(units) - 1):
        (s1, k1, t1), (s2, k2, t2) = units[i], units[i + 1]
        if s1 != s2:
            continue
        stratum = "p_p" if k1 == k2=="paragraph" else "with_table"
        pairs.append({"source": s1, "stratum": stratum,
                      "sim": float(vecs[i] @ vecs[i + 1]), "left": t1, "right": t2})
    return pairs


def print_stratum(name, sims):
    qs = np.percentile(sims, [10, 25, 50, 75, 90])
    print(f"{name}: n={len(sims)}, mean={sims.mean():.3f}, std={sims.std():.3f}")
    print("  P10/25/50/75/90 = " + "/".join(f"{q:.3f}" for q in qs))
    hist, edges = np.histogram(sims, bins=np.arange(-1, 1.01, 0.05))  # 含负值区, 防越界静默丢弃
    for h, e in zip(hist, edges):
        if h:
            print(f"  [{e:.2f}] {'#' * min(int(h), 60)} {h}")  # 封顶 60 防刷屏


def pick_samples(pairs):
    """抽检: p_p 层按 sim 升序, 取最低 10 对 + 最高 5 对."""
    s = sorted([pair  for pair in pairs if pair["stratum"]=="p_p"], key=lambda pair : pair["sim"])
    samples = s[:10]+s[-5:]
    return samples


def main():
    units = load_units()
    assert len(units) >= 1000, f"句数 {len(units)} 异常, 切分逻辑可能静默失效"  # 量级不对当场崩, 不烧 token 算垃圾
    print(f"units={len(units)}")
    vecs = embed_all([u[2] for u in units])
    pairs = make_pairs(units, vecs)
    for name in ("p_p", "with_table"):
        sims = np.array([p["sim"] for p in pairs if p["stratum"] == name],
                        dtype=np.float32)
        if len(sims):
            print_stratum(name, sims)
    print("\n抽检: p_p 相似度最低10对+最高5对(人工判是否真话题断点)")
    samples = pick_samples(pairs)
    for p in samples:
        print(f"  {p['sim']:.3f}  [{p['source']}]")
        print(f"    前: {p['left'][:50]}")
        print(f"    后: {p['right'][:50]}")
    out = ROOT / "experiments" / "scan_results_010a.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": EMBED_MODEL, "units": len(units),
                   "pairs": pairs, "samples": samples}, f, ensure_ascii=False,
                  indent=2)
    print(f"\n全量句对已落盘 {out.name}")


if __name__ == "__main__":
    main()