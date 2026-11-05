"""统一的 LLM 客户端封装。

目标：
1. model 作为参数传入，同一个 client 可以切换不同模型；
2. client 按 base_url 懒加载 + 缓存，避免重复建立连接（省 TLS 握手）；
3. chat 返回轻量结果对象 ChatResult，而不是散落的一堆字段。

用这个模块取代散落在 function_calling_v2.py / llm_agent.py 里各自建 client 的做法。
"""
import json
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from openai import APITimeoutError, RateLimitError, APIConnectionError
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
    before_sleep_log
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # 单价：元 / 每百万 token
    PRICE = {
        "qwen3.8-flash": {
            "input": 0.8,          # 普通输入
            "cached_input": 0.1,   # 缓存命中输入
            "output": 2.7,         # 输出（已含 reasoning）
        },
        "qwen3.7-text-embedding-flash": {
            "input": 0.125,        # embedding 只有输入
        },
    }


    def __init__(self):
        # 账本结构：按模型分项
        self._usage = {
             "qwen3.8-flash": {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "cost": 0.0},
             "qwen3.7-text-embedding-flash": {"input_tokens": 0, "cost": 0.0}
        }

    def _record_usage(self, model,usage):
        #每次chat/embed成功后调用,把usage累加进账本
        if hasattr(usage, 'completion_tokens'):
            temp=usage.prompt_tokens_details
            cached_tokens=getattr(temp,"cached_tokens",0) or 0
            input_tokens=usage.prompt_tokens-cached_tokens
            output_tokens=usage.completion_tokens
            self._usage[model]["cached_tokens"]+=cached_tokens
            self._usage[model]["input_tokens"]+=input_tokens
            self._usage[model]["output_tokens"]+=output_tokens
            self._usage[model]["cost"]+=(input_tokens*self.PRICE[model]["input"]+cached_tokens*self.PRICE[model]["cached_input"]+output_tokens*self.PRICE[model]["output"])/1000000
        else:
            input_tokens=usage.prompt_tokens
            self._usage[model]["input_tokens"]+=input_tokens
            self._usage[model]["cost"]+=(input_tokens*self.PRICE[model]["input"])/1000000

    def get_cost(self)->dict:
        #返回累计总账:总token,总成本,按项目分项
        return self._usage



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

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError)
        ),
        wait=wait_exponential(multiplier=2, min=15, max=120),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def chat(self, messages, model, base_url=os.getenv("OPENAI_API_BASE_URL")) -> ChatResult:
        """发起一次对话，返回 ChatResult。"""
        client=self._get_client(base_url)
        response=client.chat.completions.create(
            model=model,
            messages=messages,
        )
        chatResult=ChatResult(response.choices[0].message.content,response.usage,response.model)
        self._record_usage(model,response.usage)
        return chatResult

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError)
        ),
        wait=wait_exponential(multiplier=2, min=15, max=120),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def chat_json (self, messages, model, base_url=os.getenv("OPENAI_API_BASE_URL")) -> dict:
        """发起一次对话并要求模型返回结构化 JSON，返回解析后的 dict。

        分两层保障拿到合法 dict：
        1. 请求带 response_format={"type": "json_object"}，从源头压降低模型吐非 JSON 的概率；
        2. 客户端 json.loads 校验，失败则追加「assistant 原文 + user 纠错指令」再重试 1 次，
           两次都失败则抛错，绝不返回半成品。

        注意：messages 是调用方传入的 list，这里会在原地 append 纠错消息，
        会修改调用方那份 messages（如需复用请传副本）。
        """
        client=self._get_client(base_url)
        for _ in range(2):
            # 每次重试都重新请求，走同一段代码，避免「第一次在循环外、第二次在 except 里」的错位
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            json_string = response.choices[0].message.content
            self._record_usage(model,response.usage)
            try:
                # 解析成功且是 dict 才算合格；是 list 等其他类型也视为失败
                data=json.loads(json_string)
                if not isinstance(data,dict):
                    raise ValueError(f"不是dict,实际是{type(data).__name__}")
                return data
            except (json.decoder.JSONDecodeError,ValueError) as e:
                # 解析失败：把模型上次的输出原文 + 纠错指令一起追加，让模型知道错在哪
                messages.append({
                    "role":"assistant",
                    "content":json_string
                })
                messages.append({
                    "role":"user",
                    "content": "你返回的不是合法JSON，请只返回JSON，不要其他内容"
                })
        # 循环跑完还没 return，说明重试 2 次都失败，抛错而不是静默返回
        raise ValueError("模型连续返回非法json")

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError)
        ),
        wait=wait_exponential(multiplier=2, min=15, max=120),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def embed(self,texts:list[str],model="qwen3.7-text-embedding-flash", base_url=os.getenv("OPENAI_API_BASE_URL"))->list[list[float]]:
        """把一批文字转向量,供后续相似度检索使用"""
        client=self._get_client(base_url)
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        self._record_usage(model,response.usage)
        return [item.embedding for item in response.data]
