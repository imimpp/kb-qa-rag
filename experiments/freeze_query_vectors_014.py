"""扩评测集(40题)的问题向量冻结入口。

旧 20 题的向量直接从旧冻结文件复制(不重新调 API, 保住已入册结果的测量基准),
只对新增 20 题逐题调一次 embed, 拼成 40 行新冻结文件。
行序 = eval_set_014.json questions 顺序; float64 原样存。
前 20 行与旧文件逐位相等是回归锚: 旧题的分数差异只能来自切块法, 不来自向量抖动。
覆写保护与旧冻结脚本同款: 文件已存在即拒跑, 重冻等于换测量基准。
运行: PyCharm 右键 Run(路径全锚定, 不依赖工作目录)。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 才能 import 根目录的 llm_client
import numpy as np
import llm_client

EVAL_SET_PATH = ROOT / "experiments" / "eval_set_014.json"
OLD_SET_PATH = ROOT / "experiments" / "eval_set.json"
OLD_VECTORS_PATH = ROOT / "experiments" / "query_vectors.npy"
OUT_PATH = ROOT / "experiments" / "query_vectors_014.npy"
EXPECTED_QUESTIONS = 40
OLD_QUESTIONS = 20
EMBED_MODEL = "qwen3.7-text-embedding-flash"  # 与索引同模型
EXPECTED_DIM = 1024  # 出处: data/index/meta.json dim


def main():
    if OUT_PATH.exists():  # 冻结文件不可覆盖: 真要重冻须先手动移走旧文件
        raise FileExistsError(f"{OUT_PATH} 已存在, 拒绝覆盖")
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    assert len(questions) == EXPECTED_QUESTIONS  # 防评测集被误改题数
    with open(OLD_SET_PATH, encoding="utf-8") as f:
        old_questions = json.load(f)["questions"]
    assert len(old_questions) == OLD_QUESTIONS
    # 新集前 20 题必须与旧集逐字同序: 旧向量按行号复制才对得上题
    assert [q["question"] for q in questions[:OLD_QUESTIONS]] == [q["question"] for q in old_questions]
    old = np.load(OLD_VECTORS_PATH)
    assert old.shape == (OLD_QUESTIONS, EXPECTED_DIM)
    client = llm_client.LLMClient()
    vectors = [old[i] for i in range(OLD_QUESTIONS)]  # 旧行原样复制, 不重调 API
    for q in questions[OLD_QUESTIONS:]:
        vec = client.embed([q["question"]], model=EMBED_MODEL)[0]  # 逐题单调用, 与生产 search 同形态
        vectors.append(vec)
    arr = np.asarray(vectors)  # float64 原样存, 不降精度
    assert arr.shape == (EXPECTED_QUESTIONS, EXPECTED_DIM)  # 形状不对当场炸, 不落坏文件
    assert np.array_equal(arr[:OLD_QUESTIONS], old)  # 回归锚: 前 20 行逐位相等
    np.save(OUT_PATH, arr)
    usage = client.get_cost()[EMBED_MODEL]
    print(f"frozen={arr.shape[0]}x{arr.shape[1]}, new_live={EXPECTED_QUESTIONS - OLD_QUESTIONS}, tokens={usage['input_tokens']}, cost_yuan={usage['cost']:.4f}")
    print(f"saved={OUT_PATH}")


if __name__ == "__main__":
    main()