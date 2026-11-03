"""
[004] 流式输出延迟基线实验。

目的：测量「首 token 延迟」和「完整回答耗时」的分布（P50/P95），
用实测数字支撑「为什么要做流式输出」这一结论。

口径（写进实验记录时必须一致）：
- 首 token 延迟 = 从发起请求到「第一个非空 delta.content 到达」的时间，
  不是第一个 chunk 到达的时间（首个 chunk 通常只有 role，content 为 None）。
- 完整耗时 = 从发起请求到流式循环全部结束的时间。
- 同一问题测 N=10 次，先跑 1 次「热身」丢弃（含 TCP + TLS 握手，会虚高）。
- n=10 时 P95 接近最大值，仅作参考，不具强统计意义。
"""

import os
import time

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
    timeout=60,
    max_retries=0,
)

# 固定问题，保证每次测量的 prompt 长度一致（首 token 延迟对 prompt 长度敏感）
MODEL, QUESTION, N = "qwen3.8-flash", "用一句话解释什么是向量检索", 10


def one_call():
    """发起一次流式请求，返回 (首token延迟秒, 完整耗时秒)。

    t_first 全程为 None 直到第一次拿到非空 content，
    之后不再更新，保证记录的是「首个字的到达时间」而不是最后一个字。
    """
    t0 = time.perf_counter()   # perf_counter 是单调时钟，专用于量间隔（不受系统校时影响）
    t_first = None

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": QUESTION}],
        stream=True,            # 开启流式：返回迭代器，逐 chunk 吐出增量
    )

    for chunk in resp:
        # 流式末尾有一个「usage 块」，choices 为空列表，直接取 [0] 会 IndexError，跳过它
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content   # 流式下内容在 delta.content，不是 message.content
        if content and t_first is None:            # 有真字了、且还没记过 → 记下这一刻
            t_first = time.perf_counter() - t0

    # 完整耗时：循环全部结束后再取时间，别在循环里取（否则会变成首 token 的时间）
    return t_first, time.perf_counter() - t0


first, total = [], []
for i in range(N + 1):          # 多跑 1 次做热身
    f, t = one_call()
    if i == 0:
        continue                # 丢弃第 1 次（含握手成本，数值虚高）
    first.append(f)
    total.append(t)


def pct(data, p):
    """线性插值分位数。p=0.5 即中位数，p=0.95 即 P95。"""
    d = sorted(data)
    k = (len(d) - 1) * p        # 把百分位换算成「排序后的位置」（可能是小数）
    lo = int(k)
    hi = min(lo + 1, len(d) - 1)
    return d[lo] + (d[hi] - d[lo]) * (k - lo)   # 在两个相邻值之间插值


print("首token", [round(f * 1000) for f in first], "ms")
print("总耗时", [round(t * 1000) for t in total], "ms")

print(f"首token, P50={pct(first, 0.5) * 1000:.0f}ms, P95={pct(first, 0.95) * 1000:.0f}ms")
print(f"总耗时, P50={pct(total, 0.5) * 1000:.0f}ms, P95={pct(total, 0.95) * 1000:.0f}ms")
