"""生成侧冒烟测试：3 道有据代表题 + 1 道无据题。

目的：验证链路两端——「有据能答、出处可回指」与「无据能拒、出处为空」。
只跑链路、打印证据、做机械核对；不判断答案语义对不对（那归人看）。

用法：
    .venv/Scripts/python.exe experiments/smoke_generation.py

选题口径：三道代表题按机理挑（材料表 / 常规 / 多答案），均来自 40 题评测集；
无据题为自编问题，已核实 43 份指南正文 0 命中（住房公积金）。
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# 检索模块内部用相对路径 data/index 读索引；右键 Run 时 cwd 未必是仓库根，先切过去
os.chdir(ROOT)

import embed_index
from generation import CITATION_RE, REFUSAL_TEXT, build_messages, format_answer
from llm_client import LLMClient

META = json.loads((ROOT / "data/index/meta.json").read_text(encoding="utf-8"))
LOOKUP = {(c["source"], c["start"], c["end"]): c["text"] for c in META["chunks"]}

ES = json.loads((ROOT / "experiments/eval_set_014.json").read_text(encoding="utf-8"))
QMAP = {str(q["id"]): q["question"] for q in ES["questions"]}

CASES = [
    ("有据-材料表", "51", QMAP["51"]),
    ("有据-常规", "35", QMAP["35"]),
    ("有据-多答案", "46", QMAP["46"]),
    ("无据-拒答", None, "住房公积金提取需要什么材料？"),
]

FAKE_DATA_KEYS = ("填写示范", "常见错误示例", "受委托人姓名", "张三", "李四")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    client = LLMClient()
    before = client.get_cost()["qwen3.8-flash"]["cost"]

    for label, qid, question in CASES:
        results = embed_index.search(question, k=5)
        messages = build_messages(question, results)
        chat_result = client.chat(messages, model="qwen3.8-flash")
        answer, citations = format_answer(chat_result, results)

        print("=" * 72)
        print(f"{label} | #{qid or '-'} | {question}")
        print("-" * 72)
        print("检索 5 块：")
        for i, r in enumerate(results, 1):
            fake = "  <含示范/错误示例关键词>" if any(k in r["text"] for k in FAKE_DATA_KEYS) else ""
            print(f"  [{i}] {r['source']} blocks[{r['start']}:{r['end']}]{fake}")
        print("回答：")
        print(answer)
        print("出处：")
        if not citations:
            print("  （空）")
        for c in citations:
            s, e = c["block[start:end]"]
            ok = (c["source"], s, e) in LOOKUP
            print(f"  {c['n']} {c['source']} blocks[{s}:{e}]  回指原文={ok}")

        refused = REFUSAL_TEXT in answer
        body_nums = {int(x) - 1 for x in CITATION_RE.findall(answer)}
        cite_nums = {int(c["n"][1:-1]) - 1 for c in citations}
        all_back = all(
            (c["source"], c["block[start:end]"][0], c["block[start:end]"][1]) in LOOKUP
            for c in citations
        )
        print("机械核对：")
        print(f"  拒答文案={'是' if refused else '否'}  出处条数={len(citations)}")
        if label.startswith("无据"):
            passed = refused and not citations
            print(f"  无据题判据（拒答且出处为空）={'通过' if passed else '未通过'}")
        else:
            print(
                f"  正文标引={sorted(n + 1 for n in body_nums)} "
                f"出处编号={sorted(n + 1 for n in cite_nums)} "
                f"一致={body_nums == cite_nums}"
            )
            print(f"  出处坐标全部回指 meta.json 原文={all_back}")

    after = client.get_cost()["qwen3.8-flash"]["cost"]
    print("=" * 72)
    print(f"成本增量（仅 chat，embed/rerank 记在检索模块自己的账本实例）：+{after - before:.6f} 元")


if __name__ == "__main__":
    main()
