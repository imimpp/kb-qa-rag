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
        if len(render(current + [block])) <= window_chars:
            if not current:
                pending_start = i
            current.append(block)
        else:
            tail = []  # 从上一块带回的尾块，本身就是「连续装得下」的那几块
            if current:
                seal(i)
                if block_len(block) <= window_chars:  # 超窗块独占，不拖尾
                    for b in reversed(current):
                        if len(render([b] + tail)) > overlap_chars:
                            break
                        if len(render([b] + tail + [block])) > window_chars:
                            break
                        tail.insert(0, b)
            current = tail + [block]
            pending_start = i - len(tail)
    if current:
        seal(len(blocks))
    return chunks
