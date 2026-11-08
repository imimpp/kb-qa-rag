from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table

def paragraphs_to_text(paragraph):
    """把 docx 的非空段落抽成文字列表。"""
    text=paragraph.text.strip()
    dict1={"type":"paragraph","text":text}
    # 判 text 而非 dict1：含键的 dict 恒为真，空段落会被当成有效块放过去
    return dict1 if dict1["text"] else None

def table_to_markdown(table):
    """表格转 markdown，横向合并单元格只保留首次出现的内容。"""
    lines =[]
    for i,row in enumerate(table.rows):
        cells = []
        # Word 横向合并只存一个 <w:tc> + gridSpan 属性，python-docx 为保证「下标=列号」
        # 会把同一个 _Cell 对象连续 yield 多次（实测 12 列里 11 个是同一对象）。
        # 必须按对象身份去重：不同单元格可能合法地内容相同，按文本去重会误删。
        # prev 每行重置 —— 纵向合并跨行是不同包装对象，且 markdown 无 rowspan，跨行重复是标准写法。
        prev =None
        for c in row.cells:
            if c is prev:
                cells.append('')   # 占位而非跳过，保住列数与表头对齐
            else:
                prev = c
                cells.append(c.text.replace('\n', ' ').strip())   # 单元格内裸换行会把一行 markdown 劈成两行
        text = '|'+'|'.join(cells)+'|'
        lines.append(text)
        if i==0:
            # 分隔行插在表头之后、首条数据之前；列数取表头，数据行可能因合并/缺列短于表头
            text = '|'+'|'.join(['---'] * len(table.rows[0].cells))+'|'
            lines.append(text)
    return {"type":"table","markdown":'\n'.join(lines)}

def parse_one(doc):
    """管整个文档：按顺序遍历，段落和表格分别处理，组装成 blocks。"""
    # 用 iter_inner_content() 而非 doc.paragraphs + doc.tables：
    # 后者是两个独立列表，拼起来会丢失段落与表格在原文中的先后顺序
    blocks =[]
    for item in doc.iter_inner_content():
        if isinstance(item, Table):
            blocks.append(table_to_markdown(item))
        elif isinstance(item, Paragraph):
            block= paragraphs_to_text(item)
            if block:
                blocks.append(block)
    return blocks

if __name__ == '__main__':
    doc = Document(r"data/raw_doc/1.企业参保登记服务指南.docx")
    print(list(block for block  in parse_one(doc) if block["type"]=="table"))
    parse=parse_one(doc)
    for block in parse:
        if block["type"]=="table":
            print(len(block["markdown"]))


