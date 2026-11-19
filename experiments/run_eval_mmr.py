"""[011] MMR 多样性选择层回归: 三种切块方法各测一遍, 复用 exp_009 evaluate, 只换索引目录与输出路径。

search 已内置 MMR(lam=0.7, 候选池 N=20), 本脚本零改动调用;
问题向量走冻结文件, 全程零 API。行拆法复跑应逐位复现首跑 18/20(确定性验证)。
运行: PyCharm 右键 Run。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让同目录的 exp_009 可 import
import exp_009_eval_recall as ev

ARMS = [
    ("data/index", "eval_results_011_mmr.json"),
    ("data/index_arm2", "eval_results_011_mmr_arm2.json"),
    ("data/index_arm3", "eval_results_011_mmr_arm3.json"),
]

for index_dir, out_name in ARMS:
    print(f"\n===== {index_dir} + MMR =====")
    ev.evaluate(ev.ROOT / index_dir, ev.ROOT / "experiments" / out_name)