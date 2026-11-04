from llm_client import LLMClient, ChatResult

c = LLMClient()
result = c.chat(
    messages=[{"role": "user", "content": "你好"}],
    model="qwen3.8-flash",
)
print("content:", result.content)
print("usage:", result.usage)
print("model:", result.model)

vecs=c.embed(
    texts=["你好","世界"],
)
print("数量:",len(vecs))
print("第一个向量维度:", len(vecs[0]))
print("第一个向量前五个数:",vecs[0][:5])

