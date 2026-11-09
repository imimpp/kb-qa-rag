"""
[008] 占位符标记隔离实验。

目的：验证 parse_docs 的「四条精确骨架子串匹配」在全语料把模板空位从正文
blocks 里摘干净、且不误伤合法内容；并与裸 X{2,} 正则对照，量出通用规则的
误伤率——这条对照是「为什么不用正则」这个设计取舍的数字依据。

口径（写进实验记录时必须与此一致）：
- 样本 = data/raw_doc 下全部 *.docx。29 个 .doc 是已转换的留存原件，不计入。
- 真占位符 ground truth = 每份 4 段、全语料 4×份数，全部在段落里，表格单元格 0 个。
- 「单元」= 非空段落 + 非空表格单元格。单元格按 _tc 对象身份去重：横向合并
  (gridSpan) 会重复 yield 同一对象。去重集合存对象引用而非 id()——id 会被 GC
  回收后复用，导致静默少计。
- 「隔离数」= parse_one(doc)["placeholders"] 的条目数。
- 「残留」= blocks 中仍能被 match_placeholder 命中的 block 数，应为 0。
  paragraph 块查 text 字段，table 块查 markdown 字段。
- 「表格回归」= table 类型 block 数应等于 len(doc.tables)；table_to_block
  本轮未改动，此值不变即证明隔离逻辑没有波及表格链路。
- 「裸正则命中」= 单元文本（strip 后，不做空白归一化）能被 re.search(r"X{2,}")
  命中的单元数，大小写敏感。
- 「误伤」= 裸正则命中数 − 其中被 match_placeholder 判为真占位符的数。
- 「误伤率」= 误伤 / 裸正则命中 × 100%。

纯本地解析，不调用任何 API，零成本。
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from parse_docs import match_placeholder, parse_one

RAW_DIR = ROOT / "data/raw_doc"
NAIVE_PATTERN = re.compile(r"X{2,}")


def iter_units(doc):
    """按原文顺序吐出 (kind, text)，kind 为 'paragraph' 或 'cell'，只吐非空。"""
    for item in doc.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if text:
                yield "paragraph", text
        elif isinstance(item, Table):
            seen = set()
            for row in item.rows:
                for cell in row.cells:
                    # 存 _tc 对象本身，既去重又保住引用防止 id 复用
                    if cell._tc in seen:
                        continue
                    seen.add(cell._tc)
                    text = cell.text.strip()
                    if text:
                        yield "cell", text


def block_text(block):
    """取 block 的可检索文本：段落用 text，表格用 markdown。"""
    return block["text"] if block["type"] == "paragraph" else block["markdown"]


doc_paths = sorted(RAW_DIR.glob("*.docx"))

print("=== 语料清点 ===")
print("docx 份数:", len(doc_paths))
print()

print("=== 逐份隔离结果 ===")
total_isolated = 0
total_residual = 0
total_blocks = 0
total_table_blocks = 0
total_doc_tables = 0
drift = []

for i, path in enumerate(doc_paths, 1):
    doc = Document(str(path))
    parsed = parse_one(doc)
    blocks = parsed["blocks"]
    placeholders = parsed["placeholders"]

    n_table_blocks = sum(1 for b in blocks if b["type"] == "table")
    n_residual = sum(1 for b in blocks if match_placeholder(block_text(b)) is not None)

    total_isolated += len(placeholders)
    total_residual += n_residual
    total_blocks += len(blocks)
    total_table_blocks += n_table_blocks
    total_doc_tables += len(doc.tables)

    # 每份应恰好 4 条；不等于 4 的单独列出来，避免总数互相抵消掩盖漂移
    if len(placeholders) != 4:
        drift.append((path.stem, len(placeholders)))

    print(f"{i:>2}  隔离={len(placeholders)}  blocks={len(blocks):>3}  "
          f"残留={n_residual}  表格block={n_table_blocks}/{len(doc.tables)}  {path.stem}")

print()
print("=== 汇总 ===")
print("隔离总数:", total_isolated, " 期望:", 4 * len(doc_paths))
print("残留总数:", total_residual, " 期望: 0")
print("blocks 总数:", total_blocks)
print("表格 block 总数:", total_table_blocks, " 文档表格总数:", total_doc_tables)
print("逐份隔离数 != 4 的文档:", len(drift))
for stem, n in drift:
    print("   ", stem, "->", n)
print()

print("=== 裸 X{2,} 对照基线 ===")
naive_hits = 0
naive_true = 0
naive_by_kind = {"paragraph": 0, "cell": 0}
false_by_kind = {"paragraph": 0, "cell": 0}

for path in doc_paths:
    doc = Document(str(path))
    for kind, text in iter_units(doc):
        if NAIVE_PATTERN.search(text) is None:
            continue
        naive_hits += 1
        naive_by_kind[kind] += 1
        if match_placeholder(text) is not None:
            naive_true += 1
        else:
            false_by_kind[kind] += 1

naive_false = naive_hits - naive_true
print("裸正则命中单元:", naive_hits, " 其中段落:", naive_by_kind["paragraph"],
      " 单元格:", naive_by_kind["cell"])
print("其中真占位符:", naive_true)
print("误伤:", naive_false, " 其中段落:", false_by_kind["paragraph"],
      " 单元格:", false_by_kind["cell"])
if naive_hits:
    print(f"误伤率: {naive_false / naive_hits * 100:.1f}%")
print()

print("=== 判定 ===")
if total_isolated==4*len(doc_paths) and not drift:                          # ① 隔离总数 == 4×份数 且 drift 名单为空
    print("PASS 1 隔离总数与逐份漂移")
else:
    print("FAIL 1 隔离总数", total_isolated, "期望", 4 * len(doc_paths), "漂移:", drift)

if total_residual==0:                          # ② 残留总数 == 0
    print("PASS 2 残留")
else:
    print("FAIL 2 残留", total_residual)

if total_table_blocks==total_doc_tables:                          # ③ 表格 block 总数 == 文档表格总数
    print("PASS 3 表格回归")
else:
    print("FAIL 3 表格", total_table_blocks, "文档表格", total_doc_tables)

if naive_hits and naive_false / naive_hits * 100 > 50:                          # ④ 误伤率足够高，支撑「不用通用正则」
    print("PASS 4 对照支撑取舍")
else:
    print("FAIL 4 误伤率不足以支撑取舍")