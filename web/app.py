"""
公网回答服务入口：FastAPI 自托管静态页 + POST /api/ask。
链路：浏览器 -> 本服务 -> embed_index.search -> generation.build_messages
      -> LLMClient.chat -> generation.format_answer -> 响应 {answer, citations}。
"""
import os
from pathlib import Path

WEB = Path(__file__).resolve().parent
os.chdir(WEB)

# ↓↓ 以下由作者手写（fallback 闸）：缺 web/.env 即启动失败，不静默拿开发 key 跑通 ↓↓
assert (WEB / ".env").is_file(), "web/.env缺失,服务拒绝启动"
from dotenv import load_dotenv

load_dotenv(WEB / ".env", override=False)

from collections import defaultdict, deque
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import embed_index
from llm_client import LLMClient
from generation import build_messages, format_answer

PORT = int(os.getenv("PORT", 8000))

app = FastAPI()

RATE_LIMIT = 3
WINDOW_SEC = 60
MAX_CONCURRENT = 2
_hits = defaultdict(deque)
_hits_lock = threading.Lock()
_slots = threading.BoundedSemaphore(MAX_CONCURRENT)


class AskRequest(BaseModel):
    question: str


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")

@app.post("/api/ask")
def ask(payload: AskRequest, request: Request):
    question = payload.question.strip()
    if not question:
        return JSONResponse({"error":"先写一句话再提问"},status_code=400)
    if len(question) > 120:
        return JSONResponse({"error":"问题不能超过120字"},status_code=400)

    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _hits_lock:
        bucket = _hits[ip]
        while bucket and now -bucket[0]>WINDOW_SEC:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT:
            return JSONResponse({"error":"提问太频繁,请稍后再试"},status_code=429)
        bucket.append(now)

    if not _slots.acquire(blocking=False):
        return JSONResponse({"error":"当前提问较多,请稍后再试"},status_code=503)
    try:
        return answer_question(question)
    except Exception as e:
        return JSONResponse({"error":f"生成失败:{e}"},status_code=500)
    finally:
        _slots.release()

def answer_question(question: str) -> dict:
    """TODO（作者手写）：四步串联，返回 {"answer": str, "citations": list}。"""
    results = embed_index.search(question, k=5)
    messages = build_messages(question, results)
    chat_result = LLMClient().chat(messages, model="qwen3.8-flash")
    answer, citations = format_answer(chat_result, results)

    return {"answer":answer, "citations":citations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)