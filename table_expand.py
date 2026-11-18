"""
「固定窗+大表按行拆」切块法的预处理: 把超过窗口的大表格
按行拆成小「伪 block」, 再喂给同一套固定窗切块(窗口 500 字、重叠 100 字)。

拆分条件: 只拆「单个 block 自身就超 500 字」的大表(全库 94 块、全是表格);
伪 block 经并行的 orig_idx 列表映射回原 block 下标, start:end 评测标注不作废。
拆分规则: 每行伪 block=表头+分隔行+数据行, 空行丢弃(见 split_table_rows)。
"""
import chunking


def split_table_rows(block):
    """把一个超窗表格 block 按 markdown 行拆成伪 block 列表。

    规则: 按换行切行; 前两行(表头+分隔行)只组 head 不单独产出;
    每个数据行伪 block=表头+分隔行+该行——自描述, 防中段 chunk 丢列义;
    空行丢弃。形状 {"type":"table","markdown":…} 供 chunking.block_text 读取。
    已知限制: 单行自身可超窗(全库实测最长 741 字), 行原子不再劈。
    """
    fake_blocks = []
    rows = block["markdown"].split("\n")
    head = rows[0:2]
    for row in rows[2:]:
        if row:
            fake_blocks.append({"type": "table", "markdown": "\n".join(head+[row])})
    return fake_blocks


def expand_blocks(blocks, window_chars=500):
    """展开超窗大表为行伪 block, 返回 (expanded, orig_idx)。

    expanded[i] = 伪 block 或原 block; orig_idx[i] = 它的原 block 下标。
    切块后用 orig_idx 把 chunk 的 start:end 映射回原下标(下游管道做)。
    """
    expanded = []
    orig_idx = []
    for i, block in enumerate(blocks):
        need_split = True if block["type"] == "table" and len(block["markdown"]) > window_chars else False  # 自身超窗的表格才拆
        if need_split:
            pieces = split_table_rows(block)
            expanded.extend(pieces)
            orig_idx.extend([i] * len(pieces))  # 每行伪 block 的原下标都是 i
        else:
            expanded.append(block)
            orig_idx.append(i)
    return expanded, orig_idx
