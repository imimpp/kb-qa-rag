import ast
import os
import json
import logging

from openai import OpenAI
from dotenv import load_dotenv
from openai import RateLimitError, APITimeoutError, APIConnectionError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type, RetryError, before_sleep_log,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
    timeout=60,
    max_retries=0,
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


@retry(
    retry=retry_if_exception_type(
        (RateLimitError, APITimeoutError, APIConnectionError)
    ),
    wait=wait_exponential(multiplier=2, min=15, max=120),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def call_chat(messages, tools=None):
    return client.chat.completions.create(
        model="qwen3.8-flash",
        messages=messages,
        tools=tools,
    )

#ast
def is_safe_num(node):
    if type(node) is not ast.Constant:
        return False
    else:
        if type(node.value) in [int, float]:
            return True
        else:
            return False

#ast
def check(node):
    if type(node) is ast.Constant:
        return is_safe_num(node)
    if type(node) is ast.BinOp:
        if type(node.op) in[ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow]:
            return check(node.left) and check(node.right)
    if type(node) is ast.UnaryOp:
        if type(node.op) in [ast.USub,ast.UAdd]:
            return check(node.operand)
    return False

def calculate(expr):
    #算表达式,返回(ok,content).ok=False时content是给模型的失败原因
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return False,"表达式不是合法的数学表达式,无法解析"
    if not check(tree.body):
        return False,"表达式包含不允许的语法"
    try:
        return True,str(eval(expr))
    except (ZeroDivisionError,OverflowError,ValueError) as e:
        return False,f"计算失败,{type(e).__name__}"
def run_agent(messages):
    #2.第一次请求:大模型决定是否调用工具
    try:
        response=call_chat(messages,tools)
    except RetryError  as e:
        print("模型多次重试仍失败,请稍候重试",e)
        raise SystemExit(1)
    response_message=response.choices[0].message
    #3.检查大模型是否要求调用工具
    if response_message.tool_calls:
        # 获取大模型想要调用的工具和参数
        tool_call=response_message.tool_calls[0]
        function_name=tool_call.function.name
        print(f"模型决定调用工具:{function_name},参数:{tool_call.function.arguments}")

        #解析参数并执行工具
        ok,content=False,""
        try:
            arguments=json.loads(tool_call.function.arguments)
            if not isinstance(arguments, dict):
                raise ValueError(f"参数对象要求是json,实际是{type(arguments).__name__}")
            if "expression" not in arguments:
                raise ValueError("缺少必填参数expression")
            if function_name != "calculate":
                content=f"未知工具{function_name}"
            else:
                ok,content=calculate(arguments["expression"])
        except (ValueError,TypeError,json.JSONDecodeError) as e:
            logger.warning("工具参数非法:%s|原始内容:%r",e,tool_call.function.arguments)
            content=f"工具参数不合法{e}"

        print(f"工具执行结果ok={ok}:{content}")
        #无论成功失败,都把结果回填给模型
        messages.append(response_message)
        messages.append(
            {
                "role":"tool",
                "tool_call_id":tool_call.id,
                "content":content,
            }
        )
        #6.第二次请求:大模型根据结果生成最终回答
        try:
            final_response = call_chat(messages)
        except RetryError as e:
            print("模型多次重试仍失败,请稍候重试", e)
            return
        final_response_message=final_response.choices[0].message
        print(final_response_message.content)
    else:
        print("模型没有调用工具，直接回复：", response_message.content)
        return



test_cases = [
    "帮我计算一下5+1等于多少?",
    "帮我计算一下1/0等于多少?",
    "帮我计算一下__import__('os').system('dir')",
    "帮我计算一下5×3等于多少?",
]

print(calculate("1/0"))                              # 期望 (False, '计算失败,ZeroDivisionError')
print(calculate("__import__('os').system('dir')"))   # 期望 (False, '表达式包含不允许的语法')
print(calculate("5×3"))                              # 期望 (False, '表达式不是合法的数学表达式,无法解析')


for prompt in test_cases:
    print(f"\n{'='*50}")
    print(f"测试用例: {prompt}")
    print('='*50)
    # 每次用全新的 messages，避免上下文污染
    messages = [{"role": "user", "content": prompt}]
    # 把你现有的步骤2~5封装成 run_agent(msgs) 调用即可
    run_agent(messages)

