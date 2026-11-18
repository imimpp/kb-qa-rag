def block_text(block):
    """block 的文本：paragraph 取 text，table 取 markdown——schema 在这，两处复用。"""
    return block["text"] if block["type"] == "paragraph" else block["markdown"]


def block_len(block):
    """block 的字符数。"""
    return len(block_text(block))


def chunk_fixed_window(blocks, window_chars=500, overlap_chars=100, source=None):
    """固定窗切块：block为原子不劈开，累计字符数到window_chars就封块。

    封块后把上一块的尾块（累计≤overlap_chars）带回新块做重叠；
    单独超窗的块独占一块、不回退尾块。
    每个 chunk 含 text / start / end / source，
    start:end 可用 blocks[start:end] 切回原文；
    source 是文档标识（通常传文件名），原样透传到每个块，供检索与评测归因。
    """
    chunks = []
    current = []  # 正在攒的块的 block 列表
    pending_start = None  # 本块第一个 block 的下标，current 由空转非空时定

    if overlap_chars >= window_chars:
        raise ValueError("overlap_chars 不能大于等于 window_chars")

    def render(block_list):
        """按最终拼接方式还原文本：长度判断与真实产物用同一个表达式。"""
        return "\n".join(block_text(b) for b in block_list)

    def seal(end):
        chunks.append({
            "text": render(current),
            "start": pending_start,
            "end": end,
            "source": source,
        })

    for i, block in enumerate(blocks):
        # 试装：量「当前块＋新块」真实拼接后的字数，不超窗就收下
        if len(render(current + [block])) <= window_chars:
            if not current:
                pending_start = i
            current.append(block)
        else:
            tail = []  # 从上一块带回的尾块，本身就是「连续装得下」的那几块
            if current:
                seal(i)  # 封旧块：end=i 不含新块（区间含头不含尾）
                if block_len(block) <= window_chars:  # 超窗块独占，不拖尾
                    # 从离新块最近的旧块往回贪心挑尾块
                    for b in reversed(current):
                        if len(render([b] + tail)) > overlap_chars:
                            break  # 尾块累计超 overlap 上限，停
                        if len(render([b] + tail + [block])) > window_chars:
                            break  # 尾块＋新块超窗，停（重叠不能把新块挤超窗）
                        tail.insert(0, b)  # 插头部保持原文顺序
            current = tail + [block]
            pending_start = i - len(tail)  # 新块起点＝新块下标往回退尾块个数
    if current:
        seal(len(blocks))  # 收尾：最后一块 end＝block 总数
    return chunks


SEPARATORS = ["\n\n", "\n", "。", " ", ""]  # 梯子从粗到细：\n\n→\n→。→空格→空串(逐字符兜底)；语料无双换行，实际从 \n 起效


def split_text(text, window_chars, separators=SEPARATORS):
    """递归切文本到每片不超 window_chars：本级切不开就降一级梯子再切；碎片首尾相接，拼回等于原文。"""
    if len(text) <= window_chars:
        return [text]

    sep = separators[0]  # 当前层用梯子第一级
    rest = separators[1:]  # 剩下的梯子，留给递归用

    if sep == "":
        raw_pieces = list(text)  # 兜底：逐字符劈，没有分隔符可补
    else:
        raw_pieces = text.split(sep)  # 注意：split 之后 sep 已经不在片里了

    pieces = []
    # 把 split 吃掉的 sep 补回每片尾巴（最后一片后面本就没有 sep，不补）——补回才拼得回原文
    for i, raw in enumerate(raw_pieces):
        if i != len(raw_pieces) - 1:
            pieces.append(raw + sep)
        else:
            pieces.append(raw)

    out = []
    for piece in pieces:
        if len(piece) <= window_chars:
            out.append(piece)
        else:
            # 本片仍超窗：降一级梯子递归再切
            out.extend(split_text(piece, window_chars, separators=rest))

    return out


def merge_pieces(pieces, window_chars):
    """贪心合并相邻碎片：相接后总长不超 window_chars 就拼进当前块，否则封块开新；直接相加不插字符（sep 已在碎片里）。"""
    merged_texts = []  # 最终返回的合并文本列表
    current = ""  # 正在攒的当前块文本

    if len(pieces) == 0:
        return merged_texts

    for piece in pieces:
        if not current:
            current = piece
        else:
            if len(current) + len(piece) <= window_chars:
                current = current + piece
            else:
                merged_texts.append(current)
                current = piece

    if current:
        merged_texts.append(current)  # 封尾块
    return merged_texts


def chunk_recursive(blocks, window_chars=500, source=None):
    """递归分隔符切块：整篇递归＋偏移表反查，不加 overlap。

    blocks 用 \n 拼成全文，同时记下每个 block 的字符区间（偏移表）；
    全文按分隔符梯子递归切到不超窗、合并相邻碎片后，
    用每块的字符范围反查偏移表，得 start:end（覆盖的原始 block 下标）。
    超窗大表被切中间时 start:end 记整表区间，比块实际内容宽，评测偏松。
    不加 overlap：递归在换行、句号等语义边界下刀，硬切断少，无需回带补偿。
    """
    # 每个 block 的纯文本：paragraph 取 text，table 取 markdown（与固定窗切块同一套 schema）
    texts = [block_text(b) for b in blocks]
    # 整篇拼成一根长字符串，胶水是 \n——与 chunk_fixed_window 的拼接方式一致
    full_text = "\n".join(texts)

    # 偏移表：spans[i] = (start, end)，full_text[start:end] 恰好等于 texts[i]
    spans = []
    cursor = 0  # 下一个 block 在全文里的起点
    for text in texts:
        spans.append((cursor, cursor + len(text)))
        cursor += len(text) + 1  # +1 跳过块间胶水 \n；末块后多跳一次无人读
    # 自检①：表长必须等于 block 数（空表会让下面循环静默通过，先堵这个洞）
    assert len(spans) == len(blocks), "spans 长度与 blocks 不一致"
    # 自检②：每个区间切回去必须逐字还原原文——偏移错一位这里就炸
    for i, (s, e) in enumerate(spans):
        assert full_text[s:e] == texts[i], f"block {i} 偏移错位"

    # 两段：先递归切到每片不超窗，再贪心合并相邻碎片减少碎块
    pieces = split_text(full_text, window_chars)
    chunk_texts = merge_pieces(pieces, window_chars)
    # 自检③：可逆——切完拼回必须逐字等于全文，丢一个胶水字符这里就炸
    assert "".join(chunk_texts) == full_text, "递归切丢字符"
    # 自检④：不超窗——刀的承诺是每片字数不超窗口
    assert all(len(t) <= window_chars for t in chunk_texts), "存在超窗碎片"

    # 碎片字符范围：自检③保证首尾相接，游标累加不会歪
    chunk_ranges = []
    cursor = 0
    for text in chunk_texts:
        chunk_ranges.append((cursor, cursor + len(text)))
        cursor += len(text)
    # 自检⑤：按范围切回必须逐字等于本碎片——范围与文字两本账对齐
    for text, (cs, ce) in zip(chunk_texts, chunk_ranges):
        assert full_text[cs:ce] == text, "碎片范围错位"

    chunks = []
    dropped = 0  # 只盖胶水、被丢弃的碎片计数
    for text, (cs, ce) in zip(chunk_texts, chunk_ranges):
        first = None
        last = None
        for i, (s, e) in enumerate(spans):
            if cs < e and s < ce:
                if first is None:
                    first = i
                last = i
        if first is None:
            dropped += 1
            continue
        chunks.append({"text": text, "start": first, "end": last + 1, "source": source})

    # 自检⑥：账目对齐——产出的 chunk 数＋丢弃数＝碎片总数
    assert len(chunks) + dropped == len(chunk_texts), "反查账目不齐"
    # 自检⑦：区间合法——start:end 必须是合法的含头不含尾段落区间
    assert all(0 <= c["start"] < c["end"] <= len(blocks) for c in chunks), "段落区间非法"
    return chunks


if __name__ == "__main__":
    import json

    with open("data/parsed/1.企业参保登记服务指南.json", encoding="utf-8") as f:
        data = json.load(f)
    chunk_recursive(data["blocks"])
