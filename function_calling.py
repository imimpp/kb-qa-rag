import json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)

tools = [
    {
        "type": "function",
        "function":{
            "name":"calculate",
            "description":"Calculate your answer.",
            "parameters":{
                "type":"object",
                "properties":{
                    "expression":{
                        "type":"string",
                        "description":"要计算的数学表达式,例如'2+2'或'5*8'"
                    }
                },
                "required":["expression"]
            }
        }
    }
]

messages=[
    {
        "role":"user",
        "content":"帮我计算一下5+1等于多少?"
    }
]

#2.第一次请求:大模型决定是否调用工具
response=client.chat.completions.create(
    model="qwen3.8-flash",
    messages=messages,
    tools=tools,
)
response_message=response.choices[0].message

#3.检查大模型是否要求调用工具
if response_message.tool_calls:
    # 获取大模型想要调用的工具和参数
    tool_call=response_message.tool_calls[0]
    function_name=tool_call.function.name
    arguments=json.loads(tool_call.function.arguments)

    print(f"大模型决定调用工具:{function_name},参数是:{arguments}")

    #4.执行真正的本地工具
    if function_name == "calculate":
        result = eval(arguments["expression"])
        print("本地计算结果",result)

        #把工具执行结果还给大模型
        messages.append(response_message)
        messages.append(
            {
                "role":"tool",
                "tool_call_id":tool_call.id,
                "content":str(result)
            }
        )
        #6.第二次请求:大模型根据结果生成最终回答
        final_response=client.chat.completions.create(
            model="qwen3.8-flash",
            messages=messages,
        )
        final_response_message=final_response.choices[0].message
        print(final_response_message)
    else:
        print("大模型没有调用工具,直接回复了:",response_message)
