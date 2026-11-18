"""
评测前置步骤: 把 20 道评测题的问题向量调一次 API 算出来, 存成文件(冻结)。
之后三种切块法的评测都直接读这份文件, 不再现场调 API。

Why: 千问 embedding 接口每次返回的向量有微小差别(同一句话调两次,
余弦相似度≈0.9986, 不是完全相同)。评测时若现场调 API, 两种切块法的
分数差就分不清是切块法本身造成的, 还是接口抖动造成的; 冻结问题向量后,
分数差异只能来自切块法。
行序 = eval_set.json questions 顺序; float64 原样存(API 返回精度).
运行: PyCharm 右键 Run(路径全锚定, 不依赖工作目录).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 才能 import 根目录的 llm_client
import numpy as np
import llm_client

EVAL_SET_PATH = ROOT / "experiments" / "eval_set.json"
OUT_PATH = ROOT / "experiments" / "query_vectors.npy"
EXPECTED_QUESTIONS = 20
EMBED_MODEL = "qwen3.7-text-embedding-flash"  # 与索引同模型
EXPECTED_DIM = 1024  # 出处: data/index/meta.json dim


def main():
    if OUT_PATH.exists():  # 冻结文件不可覆盖: 重冻等于换了测量基准, 已入册的评测结果作废; 真要重冻须先手动移走旧文件
        raise FileExistsError(f"{OUT_PATH} 已存在, 拒绝覆盖")
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    assert len(questions) == EXPECTED_QUESTIONS  # 防评测集被误改题数
    client = llm_client.LLMClient()
    vectors = []
    for q in questions:
        vec = client.embed([q["question"]], model=EMBED_MODEL)[0]  # 逐题单调用, 与生产 search 同形态
        vectors.append(vec)
    arr = np.asarray(vectors)  # float64 原样存, 不降精度
    assert arr.shape == (EXPECTED_QUESTIONS, EXPECTED_DIM)  # 形状不对当场炸, 不落坏文件
    np.save(OUT_PATH, arr)
    usage = client.get_cost()[EMBED_MODEL]
    print(f"frozen={arr.shape[0]}x{arr.shape[1]}, tokens={usage['input_tokens']}, cost_yuan={usage['cost']:.4f}")
    print(f"saved={OUT_PATH}")


if __name__ == "__main__":
    main()