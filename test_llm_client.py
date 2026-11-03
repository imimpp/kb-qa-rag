from llm_client import LLMClient, ChatResult

c = LLMClient()
result = c.chat(
    messages=[{"role": "user", "content": "你好"}],
    model="qwen3.8-flash",
)
print("content:", result.content)
print("usage:", result.usage)
print("model:", result.model)
