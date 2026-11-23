import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import embed_index
from llm_client import LLMClient
from generation import build_messages, format_answer

question = "企业如何办理参保登记以及需提供什么材料"
results = embed_index.search(question, k=5)
messages = build_messages(question, results)
chat_result = LLMClient().chat(messages,model="qwen3.8-flash")
answer, citations = format_answer(chat_result,results)
print(answer)
for citation in citations:
    print(citation)
