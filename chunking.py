def block_text(block):
    """block 的文本：paragraph 取 text，table 取 markdown——schema 在这，两处复用。"""
    return block["text"] if block["type"] == "paragraph" else block["markdown"]


def block_len(block):
    """block 的字符数。"""
    return len(block_text(block))


def chunk_fixed_window(blocks, window_chars=500, overlap_chars=100):
    """固定窗切块：block为原子不劈开，累计字符数到window_chars就封块。

    封块后把上一块的尾块（累计≤overlap_chars）带回新块做重叠；
    单独超窗的块独占一块、不回退尾块。
    每个 chunk 含 text / start / end，
    start:end 可用 blocks[start:end] 切回原文。
    """
    chunks = []
    current = []  # 正在攒的块的 block 列表
    current_len = 0  # 正在攒的块的累计字符数
    pending_start = None  # 本块第一个 block 的下标，current 由空转非空时定

    if overlap_chars >= window_chars:
        raise ValueError("overlap_chars 不能大于等于 window_chars")

    def seal(end):
        chunks.append({
            "text": "\n".join(block_text(b) for b in current),
            "start": pending_start,
            "end": end,
        })

    for i, block in enumerate(blocks):
        length = block_len(block)
        if current_len + length <= window_chars:
            if not current:
                pending_start = i
            current.append(block)
            current_len += length
        else:
            tail = []  # 从上一块带回的尾块，本身就是「连续装得下」的那几块
            tail_len = 0
            if current:
                seal(i)
                if length <= window_chars:  # 超窗块独占，不拖尾
                    for b in reversed(current):
                        if tail_len + block_len(b) > overlap_chars:
                            break
                        tail.insert(0, b)
                        tail_len += block_len(b)
            current = tail + [block]
            current_len = tail_len + length
            pending_start = i - len(tail)
    if current:
        seal(len(blocks))
    return chunks
