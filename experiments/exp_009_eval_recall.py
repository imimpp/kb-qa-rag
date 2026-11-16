"""
[009] baseline Recall@5: 20 条互联网真实提问评测固定窗切块+embedding top-5.

口径见 experiments/eval_set.json 的 meta(k=5、命中=同 source 且区间相交、
Recall@5=命中题数/20)。不改 embed_index.py / chunking.py。
读评测集+逐题 search+命中判断(for-else 区分未命中),打印逐题 hit/rank 与 top-5。
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import embed_index

EVAL_SET_PATH = ROOT / "experiments" / "eval_set.json"
EXPECTED_QUESTIONS = 20
K = 5


def load_questions():
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    assert len(questions) == EXPECTED_QUESTIONS
    return questions


def load_doc_chunk_counts():
    with open(ROOT / "data/index/meta.json", encoding="utf-8") as f:
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


def main():
    counts = load_doc_chunk_counts()
    records = []
    questions = load_questions()
    hits = 0
    total = 0
    for q in questions:
        results = embed_index.search(q["question"], k=K)
        rank = 0
        hit = 0
        total += 1
        for i, r in enumerate(results, start=1):
            for answer in q["answers"]:
                if r["source"] != answer["source"]:
                    continue
                if r["start"] >= answer["end"] or r["end"] <= answer["start"]:
                    continue
                break
            else:
                continue  # 内层走完=没命中任何区间,看下一个chunk
            hit = 1
            hits += 1
            rank = i
            break
        min_chunk_count = min(counts[a["source"]] for a in q["answers"])
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
            "total": len(layer),
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
    with open(ROOT / "experiments" / "eval_results_009.json", "w", encoding="utf-8") as f:
        json.dump({"report": report, "questions": records}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
