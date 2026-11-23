"""扩评测集(40题)回归: 定型检索链路(两路召回→RRF圈池→rerank精排→MMR)在三种切块索引上各测一遍。

复用 exp_009 evaluate 的命中判断, 只换评测集/冻结向量/索引目录/输出路径。
旧 20 题在新集里逐字保留, 其命中/名次可与 eval_results_013_rerank*.json 对账(rerank 是 live API,
score 为单请求内相对分, 跨请求漂移属预期, 对账只看命中/名次/出处区间)。
rerank 走 live API: 每题一次请求排候选池 20 块, 三种切块共 120 次请求。
输出全部新文件名, 不复用任何旧证据名。
运行: PyCharm 右键 Run。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让同目录的 exp_009 可 import
import exp_009_eval_recall as ev

EVAL_SET = ev.ROOT / "experiments" / "eval_set_014.json"
VECTORS = ev.ROOT / "experiments" / "query_vectors_014.npy"
N_QUESTIONS = 40

ARMS = [
    ("data/index", "eval_results_014_arm1.json"),
    ("data/index_arm2", "eval_results_014_arm2.json"),
    ("data/index_arm3", "eval_results_014_arm3.json"),
]

for index_dir, out_name in ARMS:
    print(f"\n===== {index_dir} + RRF 圈池 + rerank 精排 + MMR (40题) =====")
    ev.evaluate(
        ev.ROOT / index_dir,
        ev.ROOT / "experiments" / out_name,
        eval_set_path=EVAL_SET,
        vectors_path=VECTORS,
        expected_questions=N_QUESTIONS,
    )