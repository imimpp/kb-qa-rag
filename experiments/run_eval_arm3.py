"""[010] 臂3 评测入口: 复用 exp_009 的 evaluate, 只换索引目录与输出路径。

问题向量走冻结文件, 全程零 API 调用; 命中判断与臂1 同一份。
运行: PyCharm 右键 Run。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让同目录的 exp_009 可 import
import exp_009_eval_recall as ev

ev.evaluate(ev.ROOT / "data/index_arm3", ev.ROOT / "experiments" / "eval_results_010_arm3.json")