"""
[006] 成本基线实验。

目的：验证 llm_client 的 token/成本埋点正确——单次 chat 和 embed 的
token 用量能正确累加，并按单价换算成成本（元）。

口径（写进实验记录时必须一致）：
- 单价单位：元 / 每百万 token；成本单位：元。
- chat 成本 = (prompt - cached)/1e6*输入价 + cached/1e6*缓存价 + completion/1e6*输出价
- embed 成本 = prompt/1e6*输入价
- token 数与成本直接读 LLMClient.get_cost() 的累计账本。
"""

import json

from llm_client import LLMClient
client = LLMClient()

# 1. 调一次 chat，打印原始 usage（拿到 token 数，供手算核对）
r=client.chat(
    messages=[{
        "role": "user",
        "content": "用一句话解释什么是向量"}
    ],
    model="qwen3.8-flash"
)

print("===chat原始usage===")
print("prompt_tokens = ",r.usage.prompt_tokens)
print("completion_tokens = ",r.usage.completion_tokens)
print("cache_tokens = ",getattr(r.usage.prompt_tokens_details,"cached_tokens",0))
print()

client.embed(
    texts=["你好","世界"]
)
print("+++已调一次embed+++")
print("====get_cost()===")
print(json.dumps(client.get_cost(), ensure_ascii=False,indent=4))