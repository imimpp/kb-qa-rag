"""
[009] baseline Recall@5: 20 条互联网真实提问评测固定窗切块+embedding top-5.

口径: k=5, 命中=top-5 存在同文档 chunk 其 [start,end) 与答案区间相交,
Recall@5=命中题数/20。不改 embed_index.py / chunking.py。
读评测集+逐题 search+命中判断(for-else 区分未命中),打印逐题 hit/rank 与 top-5。
评测逻辑抽成 evaluate(index_dir, out_path) 函数: 传哪个索引目录就测哪种
切块法, 命中判断只写一份, 三种切块法(固定窗/递归分隔符/固定窗+大表按行拆)共用。
问题向量不再现场调 API, 改读冻结文件(freeze_query_vectors.py 生成),
三种切块法用同一份问题向量, 分数差异才只来自切块法本身。
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import embed_index
import llm_client  # 运行时临时换掉它的 LLMClient.embed, 让 search 读冻结向量
import numpy as np

EVAL_SET_PATH = ROOT / "experiments" / "eval_set.json"
EXPECTED_QUESTIONS = 20  # 断言防评测集被误改题数
K = 5  # 与生产 search 实际返回条数一致, 评测才有代表性
QUERY_VECTORS_PATH = ROOT / "experiments" / "query_vectors.npy"  # 冻结问题向量, freeze_query_vectors.py 生成, 行序=题序


def load_questions(eval_set_path=None, expected_questions=None):
    path = EVAL_SET_PATH if eval_set_path is None else Path(eval_set_path)
    expected = EXPECTED_QUESTIONS if expected_questions is None else expected_questions
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    assert len(questions) == expected
    return questions


def load_doc_chunk_counts(index_dir):
    """读索引 meta, 统计每份文档被切成几块, 供后面按块数分层."""
    with open(Path(index_dir) / "meta.json", encoding="utf-8") as f:  # meta 跟着传入的索引目录走, 每种切块法各读各的
        meta = json.load(f)
    counts = {}
    for chunk in meta["chunks"]:
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1
    return counts


def print_report(report):
    print(f"Recall@{K}={report['hits']}/{report['total']}={report['recall']:.2f}")
    for name in ("<=5", ">5"):
        s = report["strata"][name]
        recall = f"{s['recall']:.2f}" if s['recall'] is not None else "N/A"
        print(f" stratum{name}:{s['hits']}/{s['total']}={recall}")


def evaluate(index_dir, out_path, eval_set_path=None, vectors_path=None, expected_questions=None):
    """三种切块法共用的评测: 命中判断只写这一份, 只换索引目录与输出路径。

    问题向量读冻结文件, 运行时临时替换 embed 不调 API:
    排除 API 每次返回向量的微小抖动, 不同切块法的分数才能公平对比。

    可选参数默认 None=沿用 20 题旧评测集与旧冻结向量(行为与原先完全一致);
    传入新评测集/新冻结向量/新题数即可评测扩充集, 命中判断逻辑不变。
    """
    embed_index.INDEX_DIR = Path(index_dir)  # search 在调用时读这个模块全局, 指到臂的索引目录即可, 不改 embed_index.py
    counts = load_doc_chunk_counts(index_dir)
    records = []
    questions = load_questions(eval_set_path, expected_questions)
    vec_path = QUERY_VECTORS_PATH if vectors_path is None else Path(vectors_path)
    frozen = np.load(vec_path)  # 行序=eval_set 题序
    expected = EXPECTED_QUESTIONS if expected_questions is None else expected_questions
    assert frozen.shape[0] == expected  # 防冻结文件与评测集错位
    frozen_by_text = {q["question"]: frozen[i] for i, q in enumerate(questions)}  # 题干->向量字典, 供离线 embed 查
    original_embed = llm_client.LLMClient.embed  # 记住真方法, 用完还原
    def offline_embed(self, texts, model=None, base_url=None):
        return [frozen_by_text[t] for t in texts]  # 查不到就 KeyError 炸出来, 绝不静默回退 live API
    llm_client.LLMClient.embed = offline_embed  # monkey-patch: search 内部调到的就是它
    try:
        hits = 0
        total = 0
        for q in questions:
            results = embed_index.search(q["question"], k=K)  # embed 已被离线替换, 不调 API
            rank = 0
            hit = 0
            total += 1
            for i, r in enumerate(results, start=1):
                for answer in q["answers"]:
                    if r["source"] != answer["source"]:
                        continue  # 不同文档, 不可能相交
                    if r["start"] >= answer["end"] or r["end"] <= answer["start"]:
                        continue  # 半开区间不相交的两种: chunk 整个在答案右边 / 整个在左边; 相邻(==)不算相交
                    break
                else:
                    continue  # 内层走完=没命中任何区间,看下一个chunk
                hit = 1
                hits += 1
                rank = i
                break  # 命中一次即可, 外层也不用再看后面的块
            min_chunk_count = min(counts[a["source"]] for a in q["answers"])  # 多个答案文档取块数最小者: 任一答案在小块文档, 这题就进 <=5 层
            stratum = "<=5" if min_chunk_count <= 5 else ">5"
            records.append({
                "id": q["id"],
                "hit": hit,
                "rank": rank,
                "stratum": stratum,
                "top5": [{
                    "source": r["source"],
                    "start": r["start"],
                    "end": r["end"],
                    "score": r["score"],
                } for r in results],
            })
            print(f"#{q['id']} hit={hit} rank={rank} {q['question']}")
            for pos, r in enumerate(results, start=1):
                print(f"  [{pos}] score={r['score']:.4f}  {r['source']}  blocks[{r['start']}:{r['end']}]")
                print(f"      {r['text'][:40]}")
        strata = {}
        for name in ("<=5", ">5"):
            layer = [record for record in records if record["stratum"] == name]
            layer_hits = sum(record["hit"] for record in layer)
            strata[name] = {
                "total": len(layer),  # 分层 recall 的分母=层内题数, 不是全局 20
                "hits": layer_hits,
                # 空层recall给None,配print_report的N/A分支
                "recall": layer_hits / len(layer) if layer else None,
            }
        report = {
            "total": total,
            "hits": hits,
            "recall": hits / total,
            "strata": strata,
        }
        print_report(report)
        with open(out_path, "w", encoding="utf-8") as f:  # 路径由调用方传入, 每种切块法各写各的文件互不覆盖
            json.dump({"report": report, "questions": records}, f, ensure_ascii=False, indent=2)
    finally:
        llm_client.LLMClient.embed = original_embed  # 无论成败都还原, 不污染同进程后续 live 调用
    return report


def main():
    evaluate(ROOT / "data/index", ROOT / "experiments" / "eval_results_009.json")  # 默认入口: 测现有固定窗切块的索引(baseline), 问题向量走冻结文件


if __name__ == "__main__":
    main()