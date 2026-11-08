"""
[007] 表格合并单元格去重实验。

目的：验证 parse_docs.table_to_markdown 的「按对象身份行内去重」能压掉
Word 横向合并（gridSpan）造成的文本重复，且不破坏无合并表格的输出。

口径（写进实验记录时必须一致）：
- 指标 = markdown 字符串的字符数 len()，不是纯文本字符数，也不是 token 数。
- 「去重前」由本脚本内的 _markdown_no_dedup() 重建（原写法已被改掉，
  磁盘上无旧版本可跑），逻辑与去重版唯一差别是不做 prev 判断。
- 「去重后」直接调用 parse_docs.table_to_markdown()，即仓库里的真实实现。
- 降幅 = (前 - 后) / 前 * 100%。
- 列数一致性 = 每行 split('|') 去掉首尾空串后的列数，全表应只有一个值。

纯本地解析，不调用任何 API，零成本。
"""

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docx import Document
from parse_docs import parse_one, table_to_markdown

DOC_PATH = ROOT / "data/raw_doc/1.企业参保登记服务指南.docx"


def _markdown_no_dedup(table):
    """去重前的写法，仅作对照基线。"""
    lines = []
    for i, row in enumerate(table.rows):
        cells = [c.text.replace('\n', ' ').strip() for c in row.cells]
        lines.append('|' + '|'.join(cells) + '|')
        if i == 0:
            lines.append('|' + '|'.join(['---'] * len(table.rows[0].cells)) + '|')
    return '\n'.join(lines)


doc = Document(str(DOC_PATH))
blocks = parse_one(doc)

print("=== blocks 构成 ===")
print("总数:", len(blocks), dict(Counter(b["type"] for b in blocks)))
print("文档规模:", len(doc.paragraphs), "段落 /", len(doc.tables), "表格")
print()

print("=== 去重前后 markdown 字符数 ===")
print("表  规格        去重前  去重后  降幅")
for i, t in enumerate(doc.tables):
    before = len(_markdown_no_dedup(t))
    after = len(table_to_markdown(t)["markdown"])
    spec = f"{len(t.rows)}行x{len(t.columns)}列"
    print(f"{i}   {spec:<11} {before:>6}  {after:>6}   {(1 - after / before) * 100:.0f}%")
print()

print("=== 列数一致性（每行 split('|') 后的列数集合）===")
for i, b in enumerate([b for b in blocks if b["type"] == "table"]):
    rows = b["markdown"].split("\n")
    counts = {len(r.split("|")) - 2 for r in rows}
    print(f"  表{i}: {len(rows)}行, 列数集合={counts}", "OK" if len(counts) == 1 else "不齐!")
print()

print("=== 表2「登记类型」行（去重后）===")
print(table_to_markdown(doc.tables[2])["markdown"].split("\n")[3])
print()

print("=== 表0 完整输出（无合并，回归对照）===")
print(table_to_markdown(doc.tables[0])["markdown"])