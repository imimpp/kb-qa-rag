from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table

# 骨架保留原文供溯源，匹配前再归一化；43 份语料实测字节级一致
PLACEHOLDER_SKELETONS = {
    "publish_date": 'XXXX年X月X日发布                   XXXX年X月X日实施',
    "publish_unit": 'XXXX（发布单位全称）  发  布',
    "address": '地址：XX市XX区（市、县）XX路XX号XX服务大厅XX-XX号窗口',
    "office_hours": '时间：周X至周X 上午XX：XX-XX:XX  下午XX：XX-XX:XX',
}


def paragraph_to_block(paragraph):
    """把 docx 段落转成一个 block，空段落返回 None。"""
    text = paragraph.text.strip()
    block = {"type": "paragraph", "text": text}
    # 判 text 而非 block：含键的 dict 恒为真，空段落会被当成有效块放过去
    return block if block["text"] else None


def table_to_block(table):
    """表格转 markdown 并包成 block，横向合并单元格只保留首次出现的内容。"""
    lines = []
    for i, row in enumerate(table.rows):
        cell_texts = []
        # Word 横向合并只存一个 <w:tc> + gridSpan 属性，python-docx 为保证「下标=列号」
        # 会把同一个 _Cell 对象连续 yield 多次（实测 12 列里 11 个是同一对象）。
        # 必须按对象身份去重：不同单元格可能合法地内容相同，按文本去重会误删。
        # prev_cell 每行重置 —— 纵向合并跨行是不同包装对象，且 markdown 无 rowspan，跨行重复是标准写法。
        prev_cell = None
        for cell in row.cells:
            if cell is prev_cell:
                cell_texts.append('')   # 占位而非跳过，保住列数与表头对齐
            else:
                prev_cell = cell
                cell_texts.append(cell.text.replace('\n', ' ').strip())   # 单元格内裸换行会把一行 markdown 劈成两行
        line = '|' + '|'.join(cell_texts) + '|'
        lines.append(line)
        if i == 0:
            # 分隔行插在表头之后、首条数据之前；列数取表头，数据行可能因合并/缺列短于表头
            line = '|' + '|'.join(['---'] * len(table.rows[0].cells)) + '|'
            lines.append(line)
    return {"type": "table", "markdown": '\n'.join(lines)}


def match_placeholder(text: str) -> str | None:
    """未填模板空位返回命中骨架键，否则 None；子串包含、两边同时空白归一化。"""
    normalized_text = ' '.join(text.split())
    for key, skeleton in PLACEHOLDER_SKELETONS.items():
        normalized_skeleton = ' '.join(skeleton.split())
        if normalized_skeleton in normalized_text:
            return key
    return None

def parse_one(doc):
    """管整个文档：按顺序遍历，段落和表格分别处理，组装成 blocks,模板空位进 placeholders。"""
    # 用 iter_inner_content() 而非 doc.paragraphs + doc.tables：
    # 后者是两个独立列表，拼起来会丢失段落与表格在原文中的先后顺序
    blocks = []
    placeholders = []
    for item in doc.iter_inner_content():
        if isinstance(item, Table):
            blocks.append(table_to_block(item))
        elif isinstance(item, Paragraph):
            rule = match_placeholder(item.text)
            if rule is not None:
                placeholders.append({"text":item.text.strip(), "rule":rule})
            else:
                block = paragraph_to_block(item)
                if block:
                    blocks.append(block)
    return {"blocks": blocks, "placeholders": placeholders}


if __name__ == '__main__':
    doc = Document(r"data/raw_doc/1.企业参保登记服务指南.docx")
    blocks = parse_one(doc)["blocks"]
    print([block for block in blocks if block["type"] == "table"])
    for block in blocks:
        if block["type"] == "table":
            print(len(block["markdown"]))