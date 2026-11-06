from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client=OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
)

response=client.chat.completions.create(
    model="qwen3.8-flash",
    messages=[{"role": "system","content":"你是一个有用的AI助手"},
              {"role":"user","content":"你好,请自我介绍一下"}
    ]
)
print(response.choices[0].message.content)
