"""统一的 LLM 客户端封装。

目标：
1. model 作为参数传入，同一个 client 可以切换不同模型；
2. client 按 base_url 懒加载 + 缓存，避免重复建立连接（省 TLS 握手）；
3. chat 返回轻量结果对象 ChatResult，而不是散落的一堆字段。

用这个模块取代散落在 function_calling_v2.py / llm_agent.py 里各自建 client 的做法。
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

@dataclass
class ChatResult:
    """chat 的返回结果，把散落的字段收口成一个对象。"""
    content: str
    usage: object
    model: str

class LLMClient:
    """封装 OpenAI SDK，负责 client 的缓存和 chat 调用。"""

    # 类属性：把 base_url 映射到已经建好的 client，避免重复建连接
    _client: dict={}

    def _get_client(self,base_url:str) -> OpenAI:
        """按 base_url 拿 client，没有就新建并缓存。"""
        if base_url not in self._client:
            self._client[base_url]= OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=base_url,
                timeout=60,
                max_retries=0,
            )

        return  self._client[base_url]


    def chat(self, messages, model, base_url=os.getenv("OPENAI_API_BASE_URL")) -> ChatResult:
        """发起一次对话，返回 ChatResult。"""
        client=self._get_client(base_url)
        response=client.chat.completions.create(
            model=model,
            messages=messages,
        )
        chatResult=ChatResult(response.choices[0].message.content,response.usage,response.model)
        return chatResult

    def embed(self,texts:list[str],model="qwen3.7-text-embedding-flash", base_url=os.getenv("OPENAI_API_BASE_URL"))->list[list[float]]:
        """把一批文字转向量,供后续相似度检索使用"""
        client=self._get_client(base_url)
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        return [item.embedding for item in response.data]