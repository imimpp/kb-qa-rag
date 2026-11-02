"""实验 003：超时与重试实测

验证三件事：
  1) 超时（可重试错误）是否按指数退避重试，最终抛 RetryError
  2) 可重试错误在后续尝试成功时，是否"退避后成功"
  3) 鉴权失败（不可重试错误）是否零重试、立即失败

⚠️ 参数缩放说明：
为控制实验时长，本脚本用 multiplier=0.5 / min=0.5 / max=2 / attempts=3 的短退避参数，
复现生产配置（multiplier=2 / min=15 / max=120 / attempts=5）的同一套逻辑——
退避曲线形状一致，只是时间轴按比例缩小。生产代码见 function_calling_v2.py。

运行：python experiments/exp_003_retry.py
"""

import os
import time
import logging

from dotenv import load_dotenv
from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    AuthenticationError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

load_dotenv()
logging.basicConfig(level=logging.WARNING)

MODEL = "qwen3.8-flash"
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE_URL")
MESSAGES = [{"role": "user", "content": "你好"}]


class _DummyRequest:
    """仅用于构造 APITimeoutError 的占位 request（异常对象只需要一个 request 入参）"""

    method = "POST"
    url = "https://example.invalid"

# 记录每次退避实际睡了多久
slept = []


def record_sleep(retry_state):
    slept.append(round(retry_state.next_action.sleep, 2))


def make_client(timeout):
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=timeout, max_retries=0)


def make_retry_decorator():
    """与生产代码同构的 retry 配置，仅时间参数缩小"""
    return retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
        stop=stop_after_attempt(3),
        before_sleep=record_sleep,
    )


def run_case1_timeout():
    """用例 1：把 timeout 压到 1ms，人为触发出可重试的超时错误"""
    slept.clear()
    client = make_client(timeout=0.001)

    @make_retry_decorator()
    def call(messages):
        return client.chat.completions.create(model=MODEL, messages=messages)

    t0 = time.perf_counter()
    try:
        call(MESSAGES)
        print("用例1：意外成功（未触发超时）")
    except RetryError as e:
        cost = time.perf_counter() - t0
        inner = e.last_attempt.exception()
        print(
            f"用例1 超时+退避：退避 {len(slept)} 次 {slept}，"
            f"总耗时 {cost:.2f}s，最终抛 RetryError（底层异常 {type(inner).__name__}）"
        )


def run_case2_recover():
    """用例 2：前两次人为抛 APITimeoutError，第三次真实请求成功"""
    slept.clear()
    client = make_client(timeout=60)
    attempts = {"n": 0}

    @make_retry_decorator()
    def call(messages):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise APITimeoutError(request=_DummyRequest())
        return client.chat.completions.create(model=MODEL, messages=messages)

    t0 = time.perf_counter()
    try:
        resp = call(MESSAGES)
        cost = time.perf_counter() - t0
        print(
            f"用例2 退避后成功：共尝试 {attempts['n']} 次，退避 {len(slept)} 次 {slept}，"
            f"总耗时 {cost:.2f}s，最终拿到回答「{resp.choices[0].message.content[:20]}…」"
        )
    except RetryError as e:
        print(f"用例2 失败：3 次均未成功，{e}")


def run_case3_no_retry():
    """用例 3：错误 api_key，鉴权失败属不可重试错误，应零退避立即抛出"""
    slept.clear()
    bad = OpenAI(api_key="sk-invalid-key-for-test", base_url=BASE_URL, timeout=60, max_retries=0)

    @make_retry_decorator()
    def call(messages):
        return bad.chat.completions.create(model=MODEL, messages=messages)

    t0 = time.perf_counter()
    try:
        call(MESSAGES)
        print("用例3：意外成功（错误 key 竟然通过）")
    except AuthenticationError:
        cost = time.perf_counter() - t0
        print(f"用例3 不可重试：退避 {len(slept)} 次，耗时 {cost:.2f}s，直接抛 AuthenticationError（未重试）")
    except RetryError:
        cost = time.perf_counter() - t0
        print(f"用例3 异常：【不符合预期】鉴权错误被重试了 {len(slept)} 次，耗时 {cost:.2f}s")


if __name__ == "__main__":
    print("=" * 60)
    run_case1_timeout()
    print("=" * 60)
    run_case2_recover()
    print("=" * 60)
    run_case3_no_retry()
    print("=" * 60)
