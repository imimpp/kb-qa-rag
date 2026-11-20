"""[013] 精排回归: 两路 RRF 融合圈候选池后加 rerank 逐块精打分喂 MMR, 三种切块方法各测一遍。

复用 exp_009 evaluate, 只换索引目录与输出路径; 问题向量走冻结文件。
rerank 走 live API: 每题一次请求排候选池 20 块, 三种切块方法共 60 次请求。
top5 的 score 字段自本实验起是 rerank 相关分(此前实验是归一化融合分),
跨实验比分数值无意义, 只比命中与名次。
运行: PyCharm 右键 Run。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让同目录的 exp_009 可 import
import exp_009_eval_recall as ev

ARMS = [
    ("data/index", "eval_results_013_rerank.json"),
    ("data/index_arm2", "eval_results_013_rerank_arm2.json"),
    ("data/index_arm3", "eval_results_013_rerank_arm3.json"),
]

for index_dir, out_name in ARMS:
    print(f"\n===== {index_dir} + RRF 圈池 + rerank 精排 + MMR =====")
    ev.evaluate(ev.ROOT / index_dir, ev.ROOT / "experiments" / out_name)