"""[012] 混合检索回归: BM25+向量 RRF 融合(归一化)接 MMR, 三种切块方法各测一遍。

复用 exp_009 evaluate, 只换索引目录与输出路径; 问题向量走冻结文件, 全程零 API。
BM25 从各索引 meta 原文现建(jieba 分词), 不碰 vectors.npy。
运行: PyCharm 右键 Run。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让同目录的 exp_009 可 import
import exp_009_eval_recall as ev

ARMS = [
    ("data/index", "eval_results_012_hybrid.json"),
    ("data/index_arm2", "eval_results_012_hybrid_arm2.json"),
    ("data/index_arm3", "eval_results_012_hybrid_arm3.json"),
]

for index_dir, out_name in ARMS:
    print(f"\n===== {index_dir} + BM25/向量 RRF + MMR =====")
    ev.evaluate(ev.ROOT / index_dir, ev.ROOT / "experiments" / out_name)